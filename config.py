"""Centralised configuration for the Self-CRAG project.

Every tunable value lives here and can be overridden from the environment or a
``.env`` file (see ``.env.example``).  Nothing else in the code base hard-codes
paths, model names, thresholds or API keys.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent

# fast   : llama3.1:8b does everything (default, fits a 6 GB GPU)
# hybrid : llama3.1:8b drafts answers, qwen2.5:14b grades them
# rigor  : qwen2.5:14b does everything (final test cases; part of it runs on CPU)
LLMMode = Literal["fast", "hybrid", "rigor"]

# What every node does when it fails.  The agent implements exactly these
# strategies; the mapping is kept here so it is documented and overridable.
DEFAULT_FALLBACK_MAP: dict[str, str] = {
    "query_analysis": "use the original question unchanged",
    "embedding": "retry with backoff, then route to Tavily web search",
    "vector_db": "treat as empty retrieval and route to Tavily web search",
    "document_grading": "fall back to cosine-similarity threshold",
    "web_search_rate_limit": "exponential backoff, then continue with local context only",
    "web_search_auth_error": "disable web search for the session and continue locally",
    "generation": "retry, then primary model, then extractive answer from the chunks",
    "eval_model_missing_or_oom": "downgrade grading to the primary model",
    "groundedness_check": "mark the answer as unverified and stop the loop",
    "answer_check": "mark the answer as unverified and stop the loop",
    "no_evidence": "abstain instead of guessing",
}


class ModelRouting(BaseModel):
    """Which Ollama model each agent node uses for the current mode."""

    mode: LLMMode
    analysis: str
    generation: str
    grading: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ paths
    data_dir: Path = PROJECT_ROOT / "data"
    kaggle_input_dir: Path = Path("/kaggle/input")
    extra_data_dirs: str = Field(default="", description="comma separated extra folders to scan")
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    chroma_collection: str = "syllabus_chunks"
    max_extract_gb: float = Field(default=20.0, gt=0, description="zip-bomb guard")

    # ----------------------------------------------------------------- ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_s: float = Field(default=240.0, gt=0)
    ollama_keep_alive: str = "10m"
    llm_num_ctx: int = Field(default=4096, ge=1024)
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_grader_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_max_new_tokens: int = Field(default=900, ge=64)
    llm_max_attempts: int = Field(default=2, ge=1, le=5, description="tries per model per call")

    embedding_model: str = "nomic-embed-text"
    embed_batch_size: int = Field(default=32, ge=1, le=256)

    primary_model: str = "llama3.1:8b"
    # NOTE: "qwen2.4" does not exist; the 14B Qwen release is qwen2.5:14b.
    eval_model: str = "qwen2.5:14b"
    llm_mode: LLMMode = "fast"

    # --------------------------------------------------------------- chunking
    chunk_size: int = Field(default=1000, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    min_chunk_chars: int = Field(default=80, ge=1)
    min_alpha_ratio: float = Field(default=0.45, ge=0, le=1)

    # -------------------------------------------------------------- retrieval
    retrieval_top_k: int = Field(default=6, ge=1, le=50)
    candidate_pool: int = Field(default=8, ge=1, le=50)
    max_query_expansions: int = Field(default=3, ge=0, le=6)
    similarity_threshold: float = Field(default=0.45, ge=0, le=1)
    confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    min_relevant_docs: int = Field(default=1, ge=1)
    max_context_chars: int = Field(default=6500, ge=1000)

    # -------------------------------------------------------------------- web
    tavily_api_key: SecretStr | None = None
    web_search_enabled: bool = True
    web_max_results: int = Field(default=4, ge=1, le=10)
    web_search_depth: Literal["basic", "advanced"] = "basic"
    web_timeout_s: float = Field(default=20.0, gt=0)
    web_max_attempts: int = Field(default=3, ge=1, le=6)
    web_min_score: float = Field(default=0.25, ge=0, le=1)

    # -------------------------------------------------------- self-correction
    max_correction_retries: int = Field(default=2, ge=0, le=6)
    agent_time_budget_s: float = Field(default=420.0, gt=0)

    # --------------------------------------------------------------------- ui
    app_port: int = 8501
    app_password: SecretStr | None = None
    log_level: str = "INFO"

    fallback_map: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_FALLBACK_MAP))

    # ------------------------------------------------------------- validators
    @field_validator("tavily_api_key", "app_password", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("data_dir", "chroma_dir", mode="after")
    @classmethod
    def _anchor_relative(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.candidate_pool < self.retrieval_top_k:
            self.candidate_pool = self.retrieval_top_k
        return self

    # ------------------------------------------------------------- properties
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def extra_dirs(self) -> list[Path]:
        return [Path(p.strip()).expanduser() for p in self.extra_data_dirs.split(",") if p.strip()]

    @property
    def data_search_dirs(self) -> list[Path]:
        """Folders scanned recursively for PDFs (local data/ and Kaggle layouts)."""
        candidates = [
            self.extracted_dir,
            self.raw_dir,
            self.kaggle_input_dir,
            *self.extra_dirs,
        ]
        seen: set[Path] = set()
        result: list[Path] = []
        for folder in candidates:
            try:
                resolved = folder.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                result.append(folder)
        return result

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.web_search_enabled and self.tavily_api_key and self.tavily_api_key.get_secret_value())

    # ---------------------------------------------------------------- helpers
    def resolve_models(self, mode: LLMMode | None = None) -> ModelRouting:
        mode = mode or self.llm_mode
        if mode == "rigor":
            return ModelRouting(mode=mode, analysis=self.eval_model, generation=self.eval_model, grading=self.eval_model)
        if mode == "hybrid":
            return ModelRouting(mode=mode, analysis=self.primary_model, generation=self.primary_model, grading=self.eval_model)
        return ModelRouting(mode="fast", analysis=self.primary_model, generation=self.primary_model, grading=self.primary_model)

    def llm_fallbacks(self, model: str) -> list[str]:
        """Models to try after ``model`` fails (not pulled, out of memory, ...)."""
        return [] if model == self.primary_model else [self.primary_model]

    def required_models(self, mode: LLMMode | None = None) -> list[str]:
        routing = self.resolve_models(mode)
        names = [self.embedding_model, routing.analysis, routing.generation, routing.grading]
        return list(dict.fromkeys(names))


def ensure_directories(settings: Settings) -> None:
    for folder in (settings.data_dir, settings.raw_dir, settings.extracted_dir, settings.chroma_dir):
        folder.mkdir(parents=True, exist_ok=True)


def configure_logging(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "chromadb", "pdfminer", "pdfplumber"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
