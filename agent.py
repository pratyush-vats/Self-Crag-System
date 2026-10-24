"""Self-Correcting Agentic RAG engine (CRAG retrieval control + Self-RAG critique).

Pipeline (every node has its own failure handling, see ``Settings.fallback_map``)::

    analyse query -> retrieve (Chroma) -> grade documents -> route
        LOCAL  : use the relevant chunks
        HYBRID : relevant chunks + Tavily web results      (ambiguous)
        WEB    : Tavily web results only                   (all chunks irrelevant)
    -> generate -> groundedness check -> answer-quality check
        -> if either fails: re-draft with feedback (up to MAX_CORRECTION_RETRIES)
           (an incomplete answer also triggers a web search if it has not run yet)
    -> output

``SelfCRAGAgent.run`` is a generator of ``AgentEvent`` objects so a UI can render
every state transition and stream tokens live.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from enum import StrEnum
from typing import Any, Literal, TypeVar

import httpx
import ollama
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import LLMMode, ModelRouting, Settings, get_settings
from ingest import DatabaseUnavailable, Embedder, EmbeddingError, get_collection

log = logging.getLogger("selfcrag.agent")

ABSTAIN_PHRASE = "I could not find enough information"
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Schemas
# =============================================================================
class Stage(StrEnum):
    RETRIEVE = "retrieve"
    GRADE = "grade"
    WEB = "web"
    GENERATE = "generate"
    CHECK = "check"
    OUTPUT = "output"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class Route(StrEnum):
    LOCAL = "local"
    HYBRID = "hybrid"
    WEB = "web"
    NONE = "none"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryAnalysis(BaseModel):
    standalone_query: str = Field(description="The question rewritten so it makes sense without chat history")
    expanded_queries: list[str] = Field(default_factory=list, description="Alternative search queries")

    @field_validator("expanded_queries")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        return [q.strip() for q in value if q and q.strip()]


class DocumentGrade(BaseModel):
    index: int = Field(description="Passage number as given in the prompt")
    relevant: bool
    score: float = Field(ge=0.0, le=1.0, description="0 = unrelated, 1 = directly answers the question")
    reason: str = ""


class DocumentGradeBatch(BaseModel):
    grades: list[DocumentGrade]


class GroundednessGrade(BaseModel):
    grounded: bool = Field(description="True only if every factual claim is supported by the context")
    confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class AnswerGrade(BaseModel):
    answers_question: bool = Field(description="The answer directly addresses what was asked")
    complete: bool = Field(description="No important part of the question is left unanswered")
    confidence: float = Field(ge=0.0, le=1.0)
    missing_points: list[str] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    ref: str = ""
    origin: Literal["local", "web"] = "local"
    text: str
    source: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""
    url: str | None = None
    similarity: float | None = None
    relevant: bool | None = None
    grade_score: float | None = None
    grade_reason: str = ""
    confidence: float = 0.0

    @property
    def label(self) -> str:
        if self.origin == "web":
            return self.source or self.url or "web"
        pages = ""
        if self.page_start:
            pages = f", p.{self.page_start}" if self.page_start == self.page_end else f", pp.{self.page_start}-{self.page_end}"
        return f"{self.source}{pages}"


class AttemptRecord(BaseModel):
    attempt: int
    answer: str
    grounded: bool | None = None
    grounded_confidence: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    answers_question: bool | None = None
    complete: bool | None = None
    answer_confidence: float | None = None
    missing_points: list[str] = Field(default_factory=list)
    abstained: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.grounded and self.answers_question and self.complete)


class AgentResult(BaseModel):
    question: str
    answer: str
    route: Route = Route.NONE
    mode: LLMMode = "fast"
    models_used: list[str] = Field(default_factory=list)
    docs: list[RetrievedDoc] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    retries_used: int = 0
    verified: bool = False
    abstained: bool = False
    hallucination_check_passed: bool | None = None
    answer_quality_passed: bool | None = None
    web_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    standalone_query: str = ""
    expanded_queries: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    type: Literal["stage", "token", "draft_reset", "final"]
    stage: Stage | None = None
    status: StageStatus | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    token: str = ""
    result: AgentResult | None = None


class HealthStatus(BaseModel):
    ollama_ok: bool
    ollama_error: str | None = None
    installed_models: list[str] = Field(default_factory=list)
    missing_models: list[str] = Field(default_factory=list)
    chroma_ok: bool = False
    chroma_chunks: int = 0
    tavily_configured: bool = False


class AgentState(BaseModel):
    """Mutable state shared by the nodes of one run."""

    question: str
    history: list[ChatTurn] = Field(default_factory=list)
    routing: ModelRouting
    max_retries: int
    allow_web: bool
    standalone_query: str = ""
    expanded_queries: list[str] = Field(default_factory=list)
    candidates: list[RetrievedDoc] = Field(default_factory=list)
    web_docs: list[RetrievedDoc] = Field(default_factory=list)
    route: Route = Route.NONE
    web_attempted: bool = False
    web_disabled: bool = False
    low_confidence_context: bool = False
    models_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    stage_status: dict[str, str] = Field(default_factory=dict)
    started: float = Field(default_factory=time.perf_counter)

    def note_model(self, model: str | None) -> None:
        if model and model not in self.models_used:
            self.models_used.append(model)

    @property
    def relevant_local(self) -> list[RetrievedDoc]:
        return [d for d in self.candidates if d.relevant]

    @property
    def context_docs(self) -> list[RetrievedDoc]:
        if self.route == Route.WEB:
            docs = list(self.web_docs)
        elif self.route == Route.HYBRID:
            docs = [*self.relevant_local, *self.web_docs]
        else:
            docs = list(self.relevant_local) + list(self.web_docs)
        if not docs and self.low_confidence_context:
            docs = sorted(self.candidates, key=lambda d: d.similarity or 0, reverse=True)[:3]
        for number, doc in enumerate([d for d in docs if d.origin == "local"], start=1):
            doc.ref = f"L{number}"
        for number, doc in enumerate([d for d in docs if d.origin == "web"], start=1):
            doc.ref = f"W{number}"
        return docs

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


# =============================================================================
# Prompts
# =============================================================================
ANALYSIS_SYSTEM = """You are the query-analysis module of a retrieval system for a college first-semester syllabus.
1. Rewrite the user's latest question as a complete, standalone question (resolve pronouns using the chat history).
2. Produce up to {n} alternative search queries that could match the relevant syllabus text: synonyms, formal textbook terms, sub-topics, expanded abbreviations.
Return JSON only."""

GRADE_SYSTEM = """You are a strict relevance grader for a retrieval system.
For every numbered PASSAGE decide whether it contains information that helps answer the QUESTION.
- relevant=true only if the passage has facts, definitions, or steps that directly help answer it.
- score: 0.0 (unrelated) to 1.0 (directly answers it).
- Keyword overlap alone is NOT relevance.
Return one grade per passage, using the same index numbers. JSON only."""

GENERATE_SYSTEM = f"""You are a patient teaching assistant for a college first-semester syllabus.
Answer the student's QUESTION using ONLY the numbered CONTEXT passages.
Rules:
- Every factual statement must come from the context. Never use outside knowledge.
- After each statement cite the passage in square brackets, for example [L1] or [W2].
- Be clear and well organised. Use short paragraphs or bullet points; stay concise.
- If the context does not contain enough information, reply starting exactly with "{ABSTAIN_PHRASE}" and then say briefly what is missing."""

GROUNDED_SYSTEM = """You are a strict fact-checking grader.
Decide whether EVERY factual claim in the ANSWER is supported by the CONTEXT.
- Ignore citation markers like [L1] and wording/style differences.
- A claim that is not stated or directly implied by the context is unsupported, even if it is true in the real world.
- grounded=true only when there are no unsupported claims. List each unsupported claim briefly.
JSON only."""

ANSWER_SYSTEM = """You are a strict answer-quality grader.
Given the QUESTION and the ANSWER decide:
- answers_question: does the answer directly address what was asked?
- complete: is every important part of the question covered (no major gap)?
List what is missing in missing_points. Judge only the answer text, not whether it is correct. JSON only."""


# =============================================================================
# LLM client with retries, model fallback and structured output
# =============================================================================
class LLMError(RuntimeError):
    """Every model and retry failed."""


class OllamaUnavailable(LLMError):
    """The Ollama server cannot be reached."""


class StreamReset:
    """Sentinel: the stream failed part-way and restarted on another model."""


STREAM_RESET = StreamReset()
_TRANSIENT = (ConnectionError, TimeoutError, httpx.TransportError)


class LLMClient:
    def __init__(self, settings: Settings, client: ollama.Client | None = None) -> None:
        self.settings = settings
        self.client = client or ollama.Client(host=settings.ollama_base_url, timeout=settings.ollama_timeout_s)

    def _options(self, temperature: float, max_tokens: int | None = None) -> dict[str, Any]:
        return {
            "temperature": temperature,
            "num_ctx": self.settings.llm_num_ctx,
            "num_predict": max_tokens or self.settings.llm_max_new_tokens,
        }

    @staticmethod
    def _chain(model: str, fallbacks: Sequence[str]) -> list[str]:
        return list(dict.fromkeys([model, *fallbacks]))

    @staticmethod
    def _extract_json(content: str) -> str:
        content = content.strip()
        if content.startswith("{") or content.startswith("["):
            return content
        match = re.search(r"\{.*\}", content, re.S)
        return match.group(0) if match else content

    def list_models(self) -> list[str]:
        response = self.client.list()
        return [m.model for m in response.models if m.model]

    def structured(
        self,
        schema: type[T],
        *,
        model: str,
        system: str,
        user: str,
        fallbacks: Sequence[str] = (),
        temperature: float | None = None,
        max_tokens: int = 700,
    ) -> tuple[T, str]:
        """Call Ollama with a JSON schema and validate the reply with Pydantic."""
        temp = self.settings.llm_grader_temperature if temperature is None else temperature
        last: Exception | None = None
        saw_connection_error = False
        for candidate in self._chain(model, fallbacks):
            feedback = ""
            for attempt in range(1, self.settings.llm_max_attempts + 1):
                try:
                    response = self.client.chat(
                        model=candidate,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user + feedback}],
                        format=schema.model_json_schema(),
                        options=self._options(temp, max_tokens),
                        keep_alive=self.settings.ollama_keep_alive,
                        stream=False,
                    )
                    content = response.message.content or ""
                    return schema.model_validate_json(self._extract_json(content)), candidate
                except (ValidationError, json.JSONDecodeError) as exc:
                    last = exc
                    feedback = "\n\nYour previous reply was not valid for the required JSON schema. Reply again with ONLY valid JSON."
                except ollama.ResponseError as exc:
                    last = exc
                    log.warning("Ollama error on %s (%s): %s", candidate, exc.status_code, exc)
                    if exc.status_code == 404 or exc.status_code < 500:
                        break  # model not installed / bad request: try the next model
                    time.sleep(1.0 * attempt)
                except _TRANSIENT as exc:
                    last = exc
                    saw_connection_error = True
                    time.sleep(1.5 * attempt)
        if saw_connection_error and isinstance(last, _TRANSIENT):
            raise OllamaUnavailable(f"Cannot reach Ollama at {self.settings.ollama_base_url}: {last}") from last
        raise LLMError(f"Structured call failed on {self._chain(model, fallbacks)}: {last}") from last

    def stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        fallbacks: Sequence[str] = (),
        temperature: float | None = None,
    ) -> Iterator[str | StreamReset]:
        """Yield tokens; yields ``STREAM_RESET`` if a model dies mid-answer and the next one restarts."""
        temp = self.settings.llm_temperature if temperature is None else temperature
        last: Exception | None = None
        saw_connection_error = False
        emitted_any = False
        for candidate in self._chain(model, fallbacks):
            if emitted_any:
                yield STREAM_RESET
                emitted_any = False
            for attempt in range(1, self.settings.llm_max_attempts + 1):
                try:
                    produced = False
                    for part in self.client.chat(
                        model=candidate,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        options=self._options(temp),
                        keep_alive=self.settings.ollama_keep_alive,
                        stream=True,
                    ):
                        token = part.message.content or ""
                        if token:
                            produced = emitted_any = True
                            yield token
                    if produced:
                        self.last_stream_model = candidate
                        return
                    last = LLMError("empty response")
                except ollama.ResponseError as exc:
                    last = exc
                    log.warning("Ollama stream error on %s (%s): %s", candidate, exc.status_code, exc)
                    if exc.status_code == 404 or exc.status_code < 500:
                        break
                    time.sleep(1.0 * attempt)
                except _TRANSIENT as exc:
                    last = exc
                    saw_connection_error = True
                    time.sleep(1.5 * attempt)
                if emitted_any:
                    yield STREAM_RESET
                    emitted_any = False
        if saw_connection_error and isinstance(last, _TRANSIENT):
            raise OllamaUnavailable(f"Cannot reach Ollama at {self.settings.ollama_base_url}: {last}") from last
        raise LLMError(f"Generation failed on {self._chain(model, fallbacks)}: {last}") from last

    last_stream_model: str | None = None


# =============================================================================
# Web search (Tavily) with rate-limit handling
# =============================================================================
class WebSearchError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class WebSearcher:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient

            assert self.settings.tavily_api_key is not None
            self._client = TavilyClient(api_key=self.settings.tavily_api_key.get_secret_value())
        return self._client

    def search(self, query: str) -> list[RetrievedDoc]:
        s = self.settings
        last: Exception | None = None
        for attempt in range(1, s.web_max_attempts + 1):
            try:
                payload = self._get_client().search(
                    query=query,
                    max_results=s.web_max_results,
                    search_depth=s.web_search_depth,
                    timeout=s.web_timeout_s,
                )
                break
            except Exception as exc:  # noqa: BLE001 - classify provider errors by name/message
                last = exc
                text = f"{type(exc).__name__} {exc}".lower()
                if any(k in text for k in ("invalidapikey", "unauthorized", "forbidden", "401", "403", "api key")):
                    raise WebSearchError(f"Tavily rejected the API key: {exc}", fatal=True) from exc
                if any(k in text for k in ("usagelimit", "rate", "429", "quota", "limit")):
                    time.sleep(min(2**attempt, 10))
                else:
                    time.sleep(1.0 * attempt)
        else:
            raise WebSearchError(f"Tavily search failed after {s.web_max_attempts} attempts: {last}") from last

        docs: list[RetrievedDoc] = []
        for item in payload.get("results", []):
            score = float(item.get("score") or 0.0)
            content = (item.get("content") or "").strip()
            if not content or score < s.web_min_score:
                continue
            docs.append(
                RetrievedDoc(
                    origin="web",
                    text=content,
                    source=item.get("title") or item.get("url") or "web result",
                    url=item.get("url"),
                    similarity=None,
                    relevant=True,
                    grade_score=score,
                    confidence=score,
                )
            )
        return docs


# =============================================================================
# Helpers
# =============================================================================
def build_context(docs: Sequence[RetrievedDoc], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for doc in docs:
        header = f"[{doc.ref}] ({doc.label})" + (f" {doc.url}" if doc.url else "")
        body = doc.text.strip()
        room = max_chars - used - len(header) - 4
        if room < 200:
            break
        body = body[:room]
        parts.append(f"{header}\n{body}")
        used += len(header) + len(body) + 4
    return "\n\n".join(parts)


def is_abstention(text: str) -> bool:
    return text.strip().lower().startswith(ABSTAIN_PHRASE.lower())


def extractive_answer(docs: Sequence[RetrievedDoc]) -> str:
    """Last-resort answer when no LLM is reachable: quote the best passages."""
    if not docs:
        return f"{ABSTAIN_PHRASE} to answer this question."
    lines = ["The language model is currently unavailable, so here are the most relevant passages I found:\n"]
    for doc in docs[:3]:
        snippet = doc.text.strip().replace("\n", " ")
        lines.append(f"- **{doc.label}**: {snippet[:450]}{'...' if len(snippet) > 450 else ''}")
    return "\n".join(lines)


def _event(stage: Stage, status: StageStatus, message: str = "", **data: Any) -> AgentEvent:
    return AgentEvent(type="stage", stage=stage, status=status, message=message, data=data)


# =============================================================================
# The agent
# =============================================================================
class SelfCRAGAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        web: WebSearcher | None = None,
        collection_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.embedder = embedder or Embedder(self.settings)
        self.web = web or WebSearcher(self.settings)
        self._collection_factory = collection_factory or (lambda: get_collection(self.settings))
        self._web_disabled_for_session = False

    # ------------------------------------------------------------------ health
    def health(self, mode: LLMMode | None = None) -> HealthStatus:
        status = HealthStatus(ollama_ok=False, tavily_configured=self.settings.tavily_enabled)
        try:
            installed = self.llm.list_models()
            normalised = {m if ":" in m else f"{m}:latest" for m in installed}
            required = self.settings.required_models(mode)
            status.ollama_ok = True
            status.installed_models = sorted(installed)
            status.missing_models = [m for m in required if (m if ":" in m else f"{m}:latest") not in normalised]
        except Exception as exc:  # noqa: BLE001
            status.ollama_error = str(exc)
        try:
            status.chroma_chunks = self._collection_factory().count()
            status.chroma_ok = True
        except Exception:  # noqa: BLE001
            status.chroma_ok = False
        return status

    # --------------------------------------------------------------------- run
    def run(
        self,
        question: str,
        history: Sequence[ChatTurn] | None = None,
        mode: LLMMode | None = None,
        max_retries: int | None = None,
        allow_web: bool = True,
    ) -> Iterator[AgentEvent]:
        routing = self.settings.resolve_models(mode)
        state = AgentState(
            question=question.strip(),
            history=list(history or [])[-6:],
            routing=routing,
            max_retries=self.settings.max_correction_retries if max_retries is None else max(0, max_retries),
            allow_web=allow_web,
        )
        try:
            yield from self._pipeline(state)
        except Exception as exc:  # noqa: BLE001 - the UI must never crash
            log.exception("Unexpected agent failure")
            state.warnings.append(f"Unexpected error: {exc}")
            yield AgentEvent(
                type="final",
                result=self._result(
                    state,
                    "Sorry, something went wrong while processing your question. Please try again.",
                    verified=False,
                ),
            )

    # ---------------------------------------------------------------- pipeline
    def _pipeline(self, state: AgentState) -> Generator[AgentEvent, None, None]:
        if not state.question:
            yield AgentEvent(type="final", result=self._result(state, "Please type a question.", verified=False))
            return

        # 1. Query analysis + retrieval ------------------------------------------------
        yield self._mark(state, Stage.RETRIEVE, StageStatus.RUNNING, "Analysing and expanding the query...")
        self._analyse(state)
        yield self._mark(
            state,
            Stage.RETRIEVE,
            StageStatus.RUNNING,
            f"Searching vector database with {1 + len(state.expanded_queries)} queries...",
            queries=[state.standalone_query, *state.expanded_queries],
        )
        t0 = time.perf_counter()
        retrieval_failed = False
        try:
            self._retrieve(state)
        except (DatabaseUnavailable, EmbeddingError, OSError) as exc:
            retrieval_failed = True
            state.warnings.append(f"Vector retrieval failed ({exc}); falling back to web search.")
            log.warning("Retrieval failed: %s", exc)
        state.timings["retrieve"] = time.perf_counter() - t0
        yield self._mark(
            state,
            Stage.RETRIEVE,
            StageStatus.FAILED if retrieval_failed else StageStatus.DONE,
            "Vector retrieval failed" if retrieval_failed else f"Retrieved {len(state.candidates)} candidate chunks",
            count=len(state.candidates),
        )

        # 2. Document grading -----------------------------------------------------------
        if state.candidates:
            yield self._mark(state, Stage.GRADE, StageStatus.RUNNING, f"Grading {len(state.candidates)} chunks...")
            t0 = time.perf_counter()
            method = self._grade(state)
            state.timings["grade"] = time.perf_counter() - t0
            n_rel = len(state.relevant_local)
            yield self._mark(
                state,
                Stage.GRADE,
                StageStatus.DONE,
                f"{n_rel}/{len(state.candidates)} chunks relevant (graded by {method})",
                relevant=n_rel,
                method=method,
            )
        else:
            yield self._mark(state, Stage.GRADE, StageStatus.SKIPPED, "Nothing to grade")

        # 3. CRAG routing -----------------------------------------------------------------
        self._decide_route(state)
        if state.route in (Route.WEB, Route.HYBRID):
            yield from self._web_stage(state, reason=f"route={state.route.value}")
        else:
            yield self._mark(state, Stage.WEB, StageStatus.SKIPPED, "Local context is sufficient")

        # 4. Make sure we have *something* to answer from --------------------------------
        if not state.context_docs:
            best = max((d.similarity or 0 for d in state.candidates), default=0)
            if state.candidates and best >= self.settings.similarity_threshold:
                state.low_confidence_context = True
                state.warnings.append("No chunk was graded relevant; answering from the closest chunks with low confidence.")
            else:
                answer = f"{ABSTAIN_PHRASE} in the syllabus documents or on the web to answer this question reliably."
                yield self._mark(state, Stage.GENERATE, StageStatus.SKIPPED, "No evidence available")
                yield self._mark(state, Stage.CHECK, StageStatus.SKIPPED, "Nothing to verify")
                yield self._mark(state, Stage.OUTPUT, StageStatus.DONE, "Abstained: no supporting evidence")
                yield AgentEvent(type="final", result=self._result(state, answer, verified=True, abstained=True))
                return

        # 5. Generate + self-correct --------------------------------------------------------
        answer = yield from self._correction_loop(state)

        # 6. Output -------------------------------------------------------------------------
        best_attempt = self._best_attempt(state)
        final_text = best_attempt.answer if best_attempt else answer
        verified = bool(best_attempt and best_attempt.passed)
        abstained = bool(best_attempt and best_attempt.abstained)
        if not verified and not abstained:
            state.warnings.append("The answer could not be fully verified against the sources. Please double-check it.")
        yield self._mark(state, Stage.OUTPUT, StageStatus.DONE, "Verified answer ready" if verified else "Answer ready (unverified)")
        yield AgentEvent(type="final", result=self._result(state, final_text, verified=verified, abstained=abstained))

    # ------------------------------------------------------------------- nodes
    def _mark(self, state: AgentState, stage: Stage, status: StageStatus, message: str = "", **data: Any) -> AgentEvent:
        state.stage_status[stage.value] = status.value
        return _event(stage, status, message, **data)

    def _analyse(self, state: AgentState) -> None:
        state.standalone_query = state.question
        n = self.settings.max_query_expansions
        if n == 0:
            return
        history = "\n".join(f"{t.role}: {t.content[:400]}" for t in state.history) or "(none)"
        user = f"CHAT HISTORY:\n{history}\n\nLATEST QUESTION:\n{state.question}"
        t0 = time.perf_counter()
        try:
            analysis, model = self.llm.structured(
                QueryAnalysis,
                model=state.routing.analysis,
                system=ANALYSIS_SYSTEM.format(n=n),
                user=user,
                fallbacks=self.settings.llm_fallbacks(state.routing.analysis),
                max_tokens=300,
            )
            state.note_model(model)
            if analysis.standalone_query.strip():
                state.standalone_query = analysis.standalone_query.strip()
            seen = {state.standalone_query.lower(), state.question.lower()}
            for query in analysis.expanded_queries:
                if query.lower() not in seen and len(state.expanded_queries) < n:
                    state.expanded_queries.append(query)
                    seen.add(query.lower())
        except LLMError as exc:  # fallback: original question, no expansion
            state.warnings.append(f"Query analysis skipped ({exc}).")
            log.warning("Query analysis failed: %s", exc)
        state.timings["analyse"] = time.perf_counter() - t0

    def _retrieve(self, state: AgentState) -> None:
        queries = [state.standalone_query, *state.expanded_queries]
        collection = self._collection_factory()
        if collection.count() == 0:
            raise DatabaseUnavailable("the vector database is empty - run ingestion first")
        vectors = self.embedder.embed_queries(queries)
        result = collection.query(
            query_embeddings=vectors,
            n_results=min(self.settings.retrieval_top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        merged: dict[str, RetrievedDoc] = {}
        for q_ids, q_docs, q_metas, q_dists in zip(
            result["ids"], result["documents"], result["metadatas"], result["distances"], strict=True
        ):
            for doc_id, text, meta, distance in zip(q_ids, q_docs, q_metas, q_dists, strict=True):
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))  # Chroma cosine distance = 1 - cos_sim
                existing = merged.get(doc_id)
                if existing is None or similarity > (existing.similarity or 0):
                    meta = meta or {}
                    merged[doc_id] = RetrievedDoc(
                        origin="local",
                        text=text,
                        source=str(meta.get("source", "")),
                        page_start=meta.get("page_start"),
                        page_end=meta.get("page_end"),
                        section=str(meta.get("section", "")),
                        similarity=similarity,
                        confidence=similarity,
                    )
        ranked = sorted(merged.values(), key=lambda d: d.similarity or 0, reverse=True)
        state.candidates = ranked[: self.settings.candidate_pool]

    def _grade(self, state: AgentState) -> str:
        """Grade chunks with the LLM; fall back to the cosine threshold per chunk."""
        passages = "\n\n".join(f"PASSAGE {i}:\n{d.text[:600]}" for i, d in enumerate(state.candidates))
        user = f"QUESTION:\n{state.standalone_query}\n\n{passages}"
        grades: dict[int, DocumentGrade] = {}
        method = "llm"
        try:
            batch, model = self.llm.structured(
                DocumentGradeBatch,
                model=state.routing.grading,
                system=GRADE_SYSTEM,
                user=user,
                fallbacks=self.settings.llm_fallbacks(state.routing.grading),
                max_tokens=900,
            )
            state.note_model(model)
            grades = {g.index: g for g in batch.grades if 0 <= g.index < len(state.candidates)}
        except LLMError as exc:
            state.warnings.append(f"LLM document grading failed ({exc}); used similarity threshold instead.")
            log.warning("Grading failed: %s", exc)
            method = "similarity-threshold"

        threshold = self.settings.similarity_threshold
        for index, doc in enumerate(state.candidates):
            grade = grades.get(index)
            similarity = doc.similarity or 0.0
            if grade is not None:
                doc.relevant = grade.relevant
                doc.grade_score = grade.score
                doc.grade_reason = grade.reason
                doc.confidence = round(0.7 * grade.score + 0.3 * similarity, 4)
            else:  # heuristic fallback (also used for passages the LLM skipped)
                doc.relevant = similarity >= threshold
                doc.grade_score = similarity
                doc.grade_reason = "cosine similarity heuristic"
                doc.confidence = similarity
                if method == "llm":
                    method = "llm+heuristic"
        return method

    def _decide_route(self, state: AgentState) -> None:
        relevant = state.relevant_local
        best = max((d.confidence for d in relevant), default=0.0)
        if len(relevant) >= self.settings.min_relevant_docs and best >= self.settings.confidence_threshold:
            state.route = Route.LOCAL
        elif relevant:
            state.route = Route.HYBRID  # ambiguous in CRAG terms
        else:
            state.route = Route.WEB  # incorrect in CRAG terms

    def _web_available(self, state: AgentState) -> bool:
        return state.allow_web and self.settings.tavily_enabled and not self._web_disabled_for_session

    def _web_stage(self, state: AgentState, reason: str) -> Generator[AgentEvent, None, None]:
        state.web_attempted = True
        if not self._web_available(state):
            why = "disabled in the sidebar" if not state.allow_web else (
                "disabled for this session" if self._web_disabled_for_session else "no TAVILY_API_KEY configured"
            )
            if state.route == Route.WEB:
                state.warnings.append(f"Web fallback was needed but is unavailable ({why}).")
            yield self._mark(state, Stage.WEB, StageStatus.SKIPPED, f"Web search unavailable ({why})")
            return
        yield self._mark(state, Stage.WEB, StageStatus.RUNNING, f"Searching the web ({reason})...")
        t0 = time.perf_counter()
        try:
            docs = self.web.search(state.standalone_query)
            state.web_docs = docs
            yield self._mark(state, Stage.WEB, StageStatus.DONE, f"{len(docs)} web results kept", count=len(docs))
        except WebSearchError as exc:
            if exc.fatal:
                self._web_disabled_for_session = True
            state.warnings.append(f"Web search failed: {exc}")
            yield self._mark(state, Stage.WEB, StageStatus.FAILED, "Web search failed; continuing with local context")
        state.timings["web"] = state.timings.get("web", 0.0) + (time.perf_counter() - t0)

    # --------------------------------------------------------- self-correction
    def _correction_loop(self, state: AgentState) -> Generator[AgentEvent, None, str]:
        s = self.settings
        max_attempts = 1 + state.max_retries
        feedback = ""
        previous = ""
        attempt = 0
        last_answer = ""
        while True:
            attempt += 1
            docs = state.context_docs
            context = build_context(docs, s.max_context_chars)
            label = "Drafting answer..." if attempt == 1 else f"Re-drafting answer (retry {attempt - 1}/{state.max_retries})..."
            yield self._mark(state, Stage.GENERATE, StageStatus.RUNNING, label, attempt=attempt)
            if attempt > 1:
                yield AgentEvent(type="draft_reset")

            t0 = time.perf_counter()
            draft = yield from self._generate(state, context, feedback, previous)
            state.timings["generate"] = state.timings.get("generate", 0.0) + (time.perf_counter() - t0)
            if draft is None:  # all models failed -> extractive fallback
                draft = extractive_answer(docs)
                state.warnings.append("Generation failed on every model; returned an extractive answer.")
                yield AgentEvent(type="draft_reset")
                yield AgentEvent(type="token", token=draft)
                state.attempts.append(AttemptRecord(attempt=attempt, answer=draft))
                yield self._mark(state, Stage.GENERATE, StageStatus.FAILED, "LLM unavailable: extractive answer used")
                yield self._mark(state, Stage.CHECK, StageStatus.SKIPPED, "Cannot verify without an LLM")
                return draft
            last_answer = draft
            yield self._mark(state, Stage.GENERATE, StageStatus.DONE, f"Draft {attempt} written ({len(draft)} chars)")

            record = AttemptRecord(attempt=attempt, answer=draft, abstained=is_abstention(draft))
            state.attempts.append(record)

            if record.abstained:
                if not state.web_attempted and self._web_available(state) and attempt < max_attempts:
                    state.route = Route.HYBRID if state.relevant_local else Route.WEB
                    yield from self._web_stage(state, reason="model found the local context insufficient")
                    if state.web_docs:
                        feedback, previous = "", ""
                        continue
                yield self._mark(state, Stage.CHECK, StageStatus.SKIPPED, "Model abstained; nothing to verify")
                return draft

            yield self._mark(state, Stage.CHECK, StageStatus.RUNNING, "Checking groundedness and answer quality...")
            t0 = time.perf_counter()
            grounded, quality = self._check(state, context, draft)
            state.timings["check"] = state.timings.get("check", 0.0) + (time.perf_counter() - t0)
            if grounded is not None:
                record.grounded = grounded.grounded
                record.grounded_confidence = grounded.confidence
                record.unsupported_claims = grounded.unsupported_claims
            if quality is not None:
                record.answers_question = quality.answers_question
                record.complete = quality.complete
                record.answer_confidence = quality.confidence
                record.missing_points = quality.missing_points

            if grounded is None or quality is None:
                state.warnings.append("A verification step could not run, so the answer is unverified.")
                yield self._mark(state, Stage.CHECK, StageStatus.FAILED, "Verification unavailable", attempt=attempt)
                return draft

            yield self._mark(
                state,
                Stage.CHECK,
                StageStatus.DONE,
                f"Grounded: {'yes' if grounded.grounded else 'NO'} | Answers question: {'yes' if quality.answers_question and quality.complete else 'NO'}",
                attempt=attempt,
                grounded=grounded.grounded,
                answers=quality.answers_question and quality.complete,
            )
            if record.passed:
                return draft
            if attempt >= max_attempts:
                state.warnings.append(f"Self-correction stopped after {state.max_retries} retries without passing every check.")
                return draft
            if state.elapsed() > s.agent_time_budget_s:
                state.warnings.append("Time budget reached; stopped self-correction early.")
                return draft

            # incomplete answer -> CRAG-style escalation to the web (once)
            incomplete = not (quality.answers_question and quality.complete)
            if incomplete and not state.web_attempted and self._web_available(state):
                state.route = Route.HYBRID if state.relevant_local else Route.WEB
                yield from self._web_stage(state, reason="answer was incomplete")
            feedback = self._feedback_text(grounded, quality)
            previous = draft

    def _generate(
        self, state: AgentState, context: str, feedback: str, previous: str
    ) -> Generator[AgentEvent, None, str | None]:
        user = f"CONTEXT:\n{context}\n\nQUESTION:\n{state.standalone_query}"
        if feedback and previous:
            user += (
                f"\n\nPREVIOUS DRAFT:\n{previous}\n\nA reviewer found these problems with the previous draft:\n{feedback}\n"
                "Write an improved answer that fixes them. Remove anything the context does not support and "
                "add missing points only if the context supports them."
            )
        text = ""
        try:
            for token in self.llm.stream(
                model=state.routing.generation,
                system=GENERATE_SYSTEM,
                user=user,
                fallbacks=self.settings.llm_fallbacks(state.routing.generation),
            ):
                if isinstance(token, StreamReset):
                    text = ""
                    yield AgentEvent(type="draft_reset")
                    continue
                text += token
                yield AgentEvent(type="token", token=token)
        except LLMError as exc:
            log.warning("Generation failed: %s", exc)
            state.warnings.append(f"Generation error: {exc}")
            return text.strip() or None
        state.note_model(self.llm.last_stream_model or state.routing.generation)
        return text.strip() or None

    def _check(
        self, state: AgentState, context: str, answer: str
    ) -> tuple[GroundednessGrade | None, AnswerGrade | None]:
        model = state.routing.grading
        fallbacks = self.settings.llm_fallbacks(model)
        grounded: GroundednessGrade | None = None
        quality: AnswerGrade | None = None
        try:
            grounded, used = self.llm.structured(
                GroundednessGrade,
                model=model,
                system=GROUNDED_SYSTEM,
                user=f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
                fallbacks=fallbacks,
                max_tokens=500,
            )
            state.note_model(used)
        except LLMError as exc:
            log.warning("Groundedness check failed: %s", exc)
        try:
            quality, used = self.llm.structured(
                AnswerGrade,
                model=model,
                system=ANSWER_SYSTEM,
                user=f"QUESTION:\n{state.standalone_query}\n\nANSWER:\n{answer}",
                fallbacks=fallbacks,
                max_tokens=400,
            )
            state.note_model(used)
        except LLMError as exc:
            log.warning("Answer-quality check failed: %s", exc)
        return grounded, quality

    @staticmethod
    def _feedback_text(grounded: GroundednessGrade, quality: AnswerGrade) -> str:
        lines: list[str] = []
        if not grounded.grounded:
            claims = "; ".join(grounded.unsupported_claims[:5]) or "some statements are not backed by the context"
            lines.append(f"- UNSUPPORTED CLAIMS (remove or fix): {claims}")
        if not (quality.answers_question and quality.complete):
            missing = "; ".join(quality.missing_points[:5]) or "the answer does not fully address the question"
            lines.append(f"- INCOMPLETE (cover if the context allows): {missing}")
        return "\n".join(lines)

    @staticmethod
    def _best_attempt(state: AgentState) -> AttemptRecord | None:
        if not state.attempts:
            return None

        def score(a: AttemptRecord) -> tuple[int, int, float]:
            return (
                int(a.passed),
                int(bool(a.grounded)),
                (a.grounded_confidence or 0.0) + (a.answer_confidence or 0.0),
            )

        return max(state.attempts, key=lambda a: (score(a), a.attempt))

    # ------------------------------------------------------------------ result
    def _result(self, state: AgentState, answer: str, *, verified: bool, abstained: bool = False) -> AgentResult:
        best = self._best_attempt(state)
        _ = state.context_docs  # (re)assigns the [L1]/[W1] reference labels
        shown = [*state.candidates, *state.web_docs]  # the UI lists every graded chunk and web hit
        hallucination = None
        quality = None
        if best is not None:
            hallucination = best.grounded
            quality = (best.answers_question and best.complete) if best.answers_question is not None else None
        if abstained:
            hallucination, quality = True, None
        return AgentResult(
            question=state.question,
            answer=answer,
            route=state.route,
            mode=state.routing.mode,
            models_used=state.models_used,
            docs=shown,
            attempts=state.attempts,
            retries_used=max(0, len(state.attempts) - 1),
            verified=verified,
            abstained=abstained,
            hallucination_check_passed=hallucination,
            answer_quality_passed=quality,
            web_used=bool(state.web_docs),
            warnings=state.warnings,
            timings={k: round(v, 2) for k, v in state.timings.items()},
            elapsed_s=round(state.elapsed(), 2),
            standalone_query=state.standalone_query,
            expanded_queries=state.expanded_queries,
        )
