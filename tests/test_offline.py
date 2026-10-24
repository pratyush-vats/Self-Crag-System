"""Offline tests: no Ollama, ChromaDB server, Tavily or internet required.

Run:  pytest -q tests
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (  # noqa: E402
    ABSTAIN_PHRASE,
    AnswerGrade,
    DocumentGrade,
    DocumentGradeBatch,
    GroundednessGrade,
    LLMError,
    QueryAnalysis,
    RetrievedDoc,
    Route,
    SelfCRAGAgent,
    StageStatus,
    WebSearchError,
)
from config import Settings  # noqa: E402
from ingest import (  # noqa: E402
    PageText,
    chunk_document,
    clean_pages,
    discover_documents,
    extract_archive,
)


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        kaggle_input_dir=tmp_path / "no-kaggle",
        tavily_api_key="tvly-test",
        max_correction_retries=2,
        llm_max_attempts=1,
    )


class FakeLLM:
    """Scriptable stand-in for LLMClient."""

    def __init__(self, relevant=(0, 1), grounded=(True,), complete=(True,), answers=("Answer text [L1].",), fail_stream=False):
        self.relevant = set(relevant)
        self.grounded = list(grounded)
        self.complete = list(complete)
        self.answers = list(answers)
        self.fail_stream = fail_stream
        self.stream_calls = 0
        self.check_calls = 0
        self.prompts: list[str] = []
        self.last_stream_model = "fake-llm"

    def structured(self, schema, *, model, system, user, **_):
        if schema is QueryAnalysis:
            return QueryAnalysis(standalone_query="what is a stack", expanded_queries=["stack data structure LIFO"]), model
        if schema is DocumentGradeBatch:
            n = user.count("PASSAGE ")
            grades = [DocumentGrade(index=i, relevant=i in self.relevant, score=0.9 if i in self.relevant else 0.1) for i in range(n)]
            return DocumentGradeBatch(grades=grades), model
        if schema is GroundednessGrade:
            i = min(self.check_calls, len(self.grounded) - 1)
            ok = self.grounded[i]
            return GroundednessGrade(grounded=ok, confidence=0.9, unsupported_claims=[] if ok else ["made-up claim"]), model
        if schema is AnswerGrade:
            i = min(self.check_calls, len(self.complete) - 1)
            self.check_calls += 1
            ok = self.complete[i]
            return AnswerGrade(answers_question=ok, complete=ok, confidence=0.8, missing_points=[] if ok else ["missing detail"]), model
        raise AssertionError(schema)

    def stream(self, *, model, system, user, **_):
        if self.fail_stream:
            raise LLMError("ollama down")
        self.prompts.append(user)
        text = self.answers[min(self.stream_calls, len(self.answers) - 1)]
        self.stream_calls += 1
        for word in text.split(" "):
            yield word + " "

    def list_models(self):
        return ["llama3.1:8b", "nomic-embed-text:latest"]


class FakeEmbedder:
    def embed_queries(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeCollection:
    def __init__(self, similarities=(0.82, 0.74, 0.30), empty=False):
        self.similarities = similarities
        self.empty = empty

    def count(self):
        return 0 if self.empty else len(self.similarities)

    def query(self, query_embeddings, n_results, include):
        ids, docs, metas, dists = [], [], [], []
        for _ in query_embeddings:
            ids.append([f"id{i}" for i in range(len(self.similarities))])
            docs.append([f"Stack is a LIFO structure number {i}." for i in range(len(self.similarities))])
            metas.append([{"source": "unit1.pdf", "page_start": i + 1, "page_end": i + 1, "section": "Unit 1"} for i in range(len(self.similarities))])
            dists.append([1 - s for s in self.similarities])
        return {"ids": ids, "documents": docs, "metadatas": metas, "distances": dists}


class FakeWeb:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    def search(self, query):
        self.calls += 1
        if self.error:
            raise self.error
        return [RetrievedDoc(origin="web", text="Web says a stack is LIFO.", source="Example", url="https://e.com", relevant=True, confidence=0.8, grade_score=0.8)]


def make_agent(settings, llm=None, collection=None, web=None, db_error: Exception | None = None):
    def factory():
        if db_error:
            raise db_error
        return collection or FakeCollection()

    return SelfCRAGAgent(settings, llm=llm or FakeLLM(), embedder=FakeEmbedder(), web=web or FakeWeb(), collection_factory=factory)


def run(agent, question="What is a stack?", **kw):
    events = list(agent.run(question, **kw))
    assert events[-1].type == "final"
    return events, events[-1].result


# ----------------------------------------------------------------------------- ingestion
WORDS = ["", "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"]


def test_cleaning_strips_headers_footers_and_rewraps():
    pages = [
        PageText(
            page_number=i,
            text=(
                "College of Engineering - Semester 1\n"
                "Unit 1 Introduction\n"
                f"{WORDS[i]} covers one idea in detail and is unique to this page.\n"
                "A stack is a linear data struc-\nture that follows LIFO order.\n"
                f"Extra explanatory line about {WORDS[i]} and queues.\n"
                f"Another distinct sentence regarding {WORDS[i + 1]} and arrays.\n"
                f"Page {i}\nConfidential syllabus 2026"
            ),
        )
        for i in range(1, 7)
    ]
    cleaned = clean_pages(pages)
    text = cleaned[0].text
    assert "College of Engineering" not in text
    assert "Confidential" not in text
    assert "Page 1" not in text
    assert "data structure that follows LIFO order." in text  # de-hyphenated and re-joined
    assert "Alpha covers one idea" in text  # unique body text is kept


def test_chunking_respects_size_and_keeps_metadata(settings):
    settings = settings.model_copy(update={"chunk_size": 300, "chunk_overlap": 60})
    paragraph = " ".join(f"Sentence number {i} explains an important syllabus concept clearly." for i in range(40))
    pages = [PageText(page_number=1, text="Unit 2 Memory\n\n" + paragraph), PageText(page_number=2, text=paragraph)]
    chunks, _ = chunk_document(pages, "unit2.pdf", "hash", settings)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 + 5 for c in chunks)
    assert chunks[0].section == "Unit 2 Memory"
    assert chunks[-1].page_end == 2
    assert len({c.id for c in chunks}) == len(chunks)


def test_noise_chunks_are_dropped(settings):
    pages = [PageText(page_number=1, text="12 34 56 78 90 12 34 56 78 90 12 34 56 78 90 12 34 56 78 90 12 34 56 78 90")]
    chunks, dropped = chunk_document(pages, "n.pdf", "h", settings)
    assert chunks == [] and dropped == 1


def test_zip_extraction_blocks_zip_slip_and_skips_junk(settings, tmp_path):
    settings.raw_dir.mkdir(parents=True)
    archive = settings.raw_dir / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("syllabus/unit1.pdf", b"%PDF-1.4 fake")
        zf.writestr("../evil.pdf", b"%PDF-1.4 evil")
        zf.writestr("images/photo.png", b"png")
    target, written, already = extract_archive(archive, settings)
    assert written == 1 and not already
    assert (target / "syllabus" / "unit1.pdf").exists()
    assert not (tmp_path / "data" / "evil.pdf").exists()
    assert not (target / "images").exists()
    _, _, already_again = extract_archive(archive, settings)
    assert already_again
    assert [p.name for p in discover_documents(settings)] == ["unit1.pdf"]


# ----------------------------------------------------------------------------- agent
def test_local_route_passes_both_checks(settings):
    llm = FakeLLM()
    events, result = run(make_agent(settings, llm=llm))
    assert result.route == Route.LOCAL
    assert result.verified and result.hallucination_check_passed and result.answer_quality_passed
    assert result.retries_used == 0
    stages = {e.stage.value: e.status for e in events if e.type == "stage"}
    assert stages["web"] == StageStatus.SKIPPED
    assert stages["output"] == StageStatus.DONE
    assert any(e.type == "token" for e in events)
    assert [d.relevant for d in result.docs] == [True, True, False]


def test_self_correction_redrafts_until_grounded(settings):
    llm = FakeLLM(grounded=(False, True), complete=(True, True), answers=("Bad draft [L1].", "Good draft [L1]."))
    events, result = run(make_agent(settings, llm=llm))
    assert result.verified and result.retries_used == 1
    assert result.answer.strip() == "Good draft [L1]."
    assert any(e.type == "draft_reset" for e in events)
    assert "made-up claim" in llm.prompts[1]  # feedback reached the second prompt
    assert result.attempts[0].grounded is False and result.attempts[1].grounded is True


def test_retry_limit_returns_best_draft_with_warning(settings):
    llm = FakeLLM(grounded=(False,), complete=(True,))
    _, result = run(make_agent(settings, llm=llm), max_retries=1)
    assert not result.verified
    assert result.retries_used == 1
    assert result.hallucination_check_passed is False
    assert any("could not be fully verified" in w for w in result.warnings)


def test_all_chunks_irrelevant_routes_to_web(settings):
    web = FakeWeb()
    _, result = run(make_agent(settings, llm=FakeLLM(relevant=()), web=web))
    assert result.route == Route.WEB and result.web_used and web.calls == 1
    assert any(d.origin == "web" for d in result.docs)


def test_ambiguous_context_uses_hybrid(settings):
    settings = settings.model_copy(update={"confidence_threshold": 0.99})
    _, result = run(make_agent(settings, web=FakeWeb()))
    assert result.route == Route.HYBRID and result.web_used


def test_incomplete_answer_escalates_to_web_once(settings):
    web = FakeWeb()
    llm = FakeLLM(grounded=(True, True), complete=(False, True), answers=("Partial [L1].", "Full [L1] [W1]."))
    _, result = run(make_agent(settings, llm=llm, web=web))
    assert web.calls == 1 and result.web_used
    assert result.verified and result.retries_used == 1


def test_database_failure_falls_back_to_web(settings):
    from ingest import DatabaseUnavailable

    web = FakeWeb()
    events, result = run(make_agent(settings, web=web, db_error=DatabaseUnavailable("db gone")))
    assert result.route == Route.WEB and web.calls == 1
    assert any("Vector retrieval failed" in w for w in result.warnings)
    assert result.answer.strip()


def test_abstains_when_no_evidence_and_no_web(settings):
    settings = settings.model_copy(update={"tavily_api_key": None})
    agent = make_agent(settings, llm=FakeLLM(relevant=()), collection=FakeCollection(similarities=(0.2, 0.1)))
    _, result = run(agent)
    assert result.abstained
    assert result.answer.startswith(ABSTAIN_PHRASE)


def test_web_rate_limit_failure_keeps_local_answer(settings):
    settings = settings.model_copy(update={"confidence_threshold": 0.99})
    _, result = run(make_agent(settings, web=FakeWeb(error=WebSearchError("429 rate limit"))))
    assert result.route == Route.HYBRID and not result.web_used
    assert result.verified
    assert any("Web search failed" in w for w in result.warnings)


def test_generation_failure_returns_extractive_answer(settings):
    _, result = run(make_agent(settings, llm=FakeLLM(fail_stream=True)))
    assert "currently unavailable" in result.answer
    assert not result.verified
    assert any("extractive" in w for w in result.warnings)


def test_empty_question_is_handled(settings):
    _, result = run(make_agent(settings), question="   ")
    assert "type a question" in result.answer.lower()


def test_health_reports_missing_models(settings):
    status = make_agent(settings).health("hybrid")
    assert status.ollama_ok
    assert "qwen2.5:14b" in status.missing_models
    assert status.chroma_ok and status.tavily_configured


def test_model_routing(settings):
    assert settings.resolve_models("fast").grading == "llama3.1:8b"
    hybrid = settings.resolve_models("hybrid")
    assert (hybrid.generation, hybrid.grading) == ("llama3.1:8b", "qwen2.5:14b")
    assert settings.resolve_models("rigor").generation == "qwen2.5:14b"
    assert settings.llm_fallbacks("qwen2.5:14b") == ["llama3.1:8b"]


def test_chunk_overlap_validation(tmp_path):
    with pytest.raises(ValueError):
        Settings(data_dir=tmp_path, chunk_size=300, chunk_overlap=300)
