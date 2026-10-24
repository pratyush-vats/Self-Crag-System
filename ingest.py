"""Dataset ingestion: ZIP/Kaggle folder -> PDF text -> clean -> chunk -> embed -> ChromaDB.

Run from the command line::

    python ingest.py              # incremental (unchanged files are skipped)
    python ingest.py --reset      # wipe the collection and rebuild
    python ingest.py --skip-extract

The same ``run_ingestion`` function powers the "Re-ingest" button in the app.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import chromadb
import ollama
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, Field

from config import Settings, configure_logging, ensure_directories, get_settings

log = logging.getLogger("selfcrag.ingest")

ProgressFn = Callable[[float, str], None]
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# =============================================================================
# Schemas
# =============================================================================
class PageText(BaseModel):
    page_number: int
    text: str


class Chunk(BaseModel):
    id: str
    text: str
    embed_text: str
    source: str
    page_start: int
    page_end: int
    section: str = ""
    chunk_index: int
    doc_hash: str


class IngestReport(BaseModel):
    archives_extracted: list[str] = Field(default_factory=list)
    files_found: int = 0
    files_ingested: int = 0
    files_skipped_unchanged: int = 0
    files_failed: list[str] = Field(default_factory=list)
    chunks_added: int = 0
    chunks_dropped_noise: int = 0
    chunks_total_in_db: int = 0
    warnings: list[str] = Field(default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.files_ingested} ingested, {self.files_skipped_unchanged} unchanged, "
            f"{len(self.files_failed)} failed | +{self.chunks_added} chunks "
            f"({self.chunks_total_in_db} in DB) in {self.elapsed_s:.1f}s"
        )


class DBStatus(BaseModel):
    ok: bool
    chunks: int = 0
    documents: int = 0
    error: str | None = None


class DatabaseUnavailable(RuntimeError):
    """ChromaDB cannot be opened or the collection does not exist."""


class EmbeddingError(RuntimeError):
    """Ollama could not produce embeddings."""


# =============================================================================
# 1. ZIP handling and file discovery
# =============================================================================
def find_archives(settings: Settings) -> list[Path]:
    found: dict[Path, Path] = {}
    if settings.raw_dir.exists():
        for path in settings.raw_dir.rglob("*.zip"):
            found[path.resolve()] = path
    if settings.data_dir.exists():
        for path in settings.data_dir.glob("*.zip"):
            found[path.resolve()] = path
    return sorted(found.values())


def _safe_folder_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "archive"


def extract_archive(zip_path: Path, settings: Settings) -> tuple[Path, int, bool]:
    """Extract supported files from ``zip_path``.

    Returns ``(target_dir, files_written, already_extracted)``.  Protects against
    zip-slip (``../`` paths) and zip bombs (``max_extract_gb``).
    """
    target = settings.extracted_dir / _safe_folder_name(zip_path.stem)
    stat = zip_path.stat()
    stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    marker = target / ".extracted"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == stamp:
        return target, 0, True

    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    limit = int(settings.max_extract_gb * 1024**3)
    total = 0
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            if Path(info.filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            total += info.file_size
            if total > limit:
                raise ValueError(f"{zip_path.name}: extracted size exceeds MAX_EXTRACT_GB={settings.max_extract_gb}")
            destination = (target / info.filename).resolve()
            if not destination.is_relative_to(root):
                log.warning("Skipping unsafe path in zip: %s", info.filename)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            written += 1
    marker.write_text(stamp, encoding="utf-8")
    return target, written, False


def discover_documents(settings: Settings) -> list[Path]:
    """Recursively collect PDFs (and .txt/.md) from every configured folder."""
    chroma_root = settings.chroma_dir.resolve()
    found: dict[Path, Path] = {}
    for folder in settings.data_search_dirs:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved.is_relative_to(chroma_root):
                continue
            found.setdefault(resolved, path)
    return sorted(found.values(), key=lambda p: str(p).lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_name(path: Path, settings: Settings) -> str:
    resolved = path.resolve()
    for base in (settings.extracted_dir, settings.raw_dir, settings.data_dir, settings.kaggle_input_dir):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return path.name


# =============================================================================
# 2. Text extraction
# =============================================================================
def _extract_with_pdfplumber(path: Path) -> list[PageText]:
    import pdfplumber

    pages: list[PageText] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the document
                log.warning("%s p.%d: pdfplumber failed (%s)", path.name, number, exc)
                text = ""
            pages.append(PageText(page_number=number, text=text))
            page.flush_cache()
    return pages


def _extract_with_pypdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("PDF is password protected") from exc
    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("%s p.%d: pypdf failed (%s)", path.name, number, exc)
            text = ""
        pages.append(PageText(page_number=number, text=text))
    return pages


def extract_pages(path: Path) -> tuple[list[PageText], str]:
    """Return ``(pages, extractor_name)``; falls back from pdfplumber to pypdf."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [PageText(page_number=1, text=path.read_text(encoding="utf-8", errors="ignore"))], "text"

    pages: list[PageText] = []
    try:
        pages = _extract_with_pdfplumber(path)
        extractor = "pdfplumber"
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: pdfplumber failed (%s); trying pypdf", path.name, exc)
        extractor = "pypdf"

    if sum(len(p.text.strip()) for p in pages) < 50:
        pages = _extract_with_pypdf(path)
        extractor = "pypdf"
    return pages, extractor


# =============================================================================
# 3. Cleaning
# =============================================================================
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\ufeff\u00ad]")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?[-\u2013\u2014]?\s*\d{1,4}\s*(?:(?:of|/)\s*\d{1,4})?\s*[-\u2013\u2014]?\s*$", re.I)
_DOT_LEADER_RE = re.compile(r"(?:\.\s?){5,}")
_BULLET_RE = re.compile(r"^\s*(?:[-\u2022\u25cf\u25aa\u25e6*]|\d{1,2}[.)]|[a-zA-Z][.)]|\([a-zA-Z0-9]{1,3}\))\s+")
_HEADING_WORD_RE = re.compile(r"^(unit|module|chapter|section|topic|semester|course|syllabus|part|lab|experiment)\b", re.I)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z]")


def _line_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def detect_boilerplate(
    pages: Sequence[PageText], edge_lines: int = 2, min_ratio: float = 0.4, min_pages: int = 4, min_lines: int = 8
) -> set[str]:
    """Find running headers/footers: short lines at page edges repeated on many pages.

    Very short pages (fewer than ``min_lines`` lines, e.g. slides) are ignored so real
    content is never mistaken for a header.
    """
    if len(pages) < min_pages:
        return set()
    counter: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        if len(lines) < min_lines:
            continue
        edges = lines[:edge_lines] + lines[-edge_lines:]
        counter.update({_line_key(ln) for ln in edges if len(ln) <= 120})
    threshold = max(3, int(len(pages) * min_ratio))
    return {key for key, count in counter.items() if key and count >= threshold}


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90 or line.endswith((".", ",", ";")):
        return False
    if _HEADING_WORD_RE.match(line) or _NUMBERED_HEADING_RE.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    if line.isupper():
        return True
    words = [w for w in line.split() if w[0].isalpha()]
    if 1 <= len(words) <= 10 and sum(w[0].isupper() for w in words) / len(words) >= 0.8:
        return True
    return False


def reflow_paragraphs(text: str) -> str:
    """Join hard-wrapped lines into paragraphs while keeping headings and bullets."""
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in text.split("\n"):
        if not line:
            flush()
        elif is_heading(line):
            flush()
            paragraphs.append(line)
        elif _BULLET_RE.match(line):
            flush()
            current.append(line)
        else:
            current.append(line)
    flush()
    return "\n\n".join(p for p in paragraphs if p.strip())


def clean_page_text(text: str, boilerplate: set[str] | None = None) -> str:
    boilerplate = boilerplate or set()
    text = unicodedata.normalize("NFKC", text)  # also expands ligatures (fi, fl)
    text = _CONTROL_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    kept: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
        if not line:
            kept.append("")
            continue
        if _PAGE_NUMBER_RE.match(line):
            continue
        if len(line) <= 120 and _line_key(line) in boilerplate:
            continue
        if _DOT_LEADER_RE.search(line):  # table-of-contents leader lines
            continue
        if len(line) > 3 and sum(c.isalnum() for c in line) / len(line) < 0.25:
            continue  # lines of pure symbols / rules / artifacts
        kept.append(line)
    joined = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", "\n".join(kept))  # de-hyphenate wrapped words
    return reflow_paragraphs(joined)


def clean_pages(pages: Sequence[PageText]) -> list[PageText]:
    boilerplate = detect_boilerplate(pages)
    cleaned = [PageText(page_number=p.page_number, text=clean_page_text(p.text, boilerplate)) for p in pages]
    return [p for p in cleaned if p.text.strip()]


# =============================================================================
# 4. Chunking on semantic boundaries
# =============================================================================
class _Atom(NamedTuple):
    text: str
    page: int
    section: str
    para_break: bool


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def _atomize(paragraph: str, limit: int) -> list[str]:
    """Split a paragraph into pieces no longer than ``limit`` (sentences, then words)."""
    if len(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    for sentence in _SENTENCE_RE.split(paragraph):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        buffer = ""
        for word in sentence.split():
            if buffer and len(buffer) + len(word) + 1 > limit:
                pieces.append(buffer)
                buffer = word
            else:
                buffer = f"{buffer} {word}".strip()
        if buffer:
            pieces.append(buffer)
    return pieces


def _render(atoms: Sequence[_Atom]) -> str:
    text = ""
    for atom in atoms:
        if text:
            text += "\n\n" if atom.para_break else " "
        text += atom.text
    return text


def _alpha_ratio(text: str) -> float:
    return sum(c.isalpha() for c in text) / len(text) if text else 0.0


def chunk_document(pages: Sequence[PageText], source: str, doc_hash: str, settings: Settings) -> tuple[list[Chunk], int]:
    """Pack paragraphs/sentences into ~``chunk_size`` char chunks with overlap.

    Returns ``(chunks, number_of_chunks_dropped_as_noise)``.
    """
    size, overlap = settings.chunk_size, settings.chunk_overlap
    atoms: list[_Atom] = []
    section = ""
    for page in pages:
        for paragraph in page.text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if is_heading(paragraph):
                section = paragraph
            for index, piece in enumerate(_atomize(paragraph, size)):
                atoms.append(_Atom(piece, page.page_number, section, index == 0))

    raw_chunks: list[list[_Atom]] = []
    current: list[_Atom] = []
    for atom in atoms:
        if current and len(_render([*current, atom])) > size:
            raw_chunks.append(current)
            tail: list[_Atom] = []
            total = 0
            for previous in reversed(current):
                if total + len(previous.text) + 1 > overlap:
                    break
                tail.insert(0, previous)
                total += len(previous.text) + 1
            while tail and len(_render([*tail, atom])) > size:
                tail.pop(0)
            current = tail
        current.append(atom)
    if current:
        raw_chunks.append(current)

    chunks: list[Chunk] = []
    dropped = 0
    for group in raw_chunks:
        text = _render(group).strip()
        if len(text) < settings.min_chunk_chars or _alpha_ratio(text) < settings.min_alpha_ratio:
            dropped += 1
            continue
        index = len(chunks)
        section_name = group[0].section
        embed_text = f"{section_name}\n{text}" if section_name and not text.startswith(section_name) else text
        chunks.append(
            Chunk(
                id=hashlib.sha1(f"{doc_hash}:{index}".encode()).hexdigest()[:24],
                text=text,
                embed_text=embed_text,
                source=source,
                page_start=min(a.page for a in group),
                page_end=max(a.page for a in group),
                section=section_name,
                chunk_index=index,
                doc_hash=doc_hash,
            )
        )
    return chunks, dropped


# =============================================================================
# 5. Embeddings (Ollama) and ChromaDB
# =============================================================================
class Embedder:
    """Thin wrapper over ``ollama.Client.embed`` with batching and retries."""

    def __init__(self, settings: Settings, client: ollama.Client | None = None) -> None:
        self.settings = settings
        self.client = client or ollama.Client(host=settings.ollama_base_url, timeout=settings.ollama_timeout_s)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.embed(
                    model=self.settings.embedding_model,
                    input=texts,
                    keep_alive=self.settings.ollama_keep_alive,
                )
                vectors = [list(v) for v in response.embeddings]
                if len(vectors) != len(texts):
                    raise EmbeddingError("embedding count mismatch")
                return vectors
            except ollama.ResponseError as exc:
                last = exc
                if exc.status_code == 404:
                    raise EmbeddingError(
                        f"Embedding model '{self.settings.embedding_model}' is not installed. "
                        f"Run: ollama pull {self.settings.embedding_model}"
                    ) from exc
                time.sleep(1.5 * attempt)
            except (ConnectionError, TimeoutError, OSError) as exc:
                last = exc
                time.sleep(1.5 * attempt)
        raise EmbeddingError(f"Ollama embedding failed: {last}") from last

    def embed_documents(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Embed with the ``search_document:`` prefix. Failed items are ``None``."""
        prefixed = [f"search_document: {t}" for t in texts]
        out: list[list[float] | None] = []
        batch = self.settings.embed_batch_size
        for start in range(0, len(prefixed), batch):
            part = prefixed[start : start + batch]
            try:
                out.extend(self._embed(part))
            except EmbeddingError as exc:
                if "not installed" in str(exc):
                    raise
                log.warning("Batch embed failed (%s); retrying one by one", exc)
                for text in part:
                    try:
                        out.append(self._embed([text])[0])
                    except EmbeddingError:
                        out.append(None)
        return out

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([f"search_query: {t}" for t in texts])

    def ping(self) -> None:
        self._embed(["search_query: ping"])


_CLIENTS: dict[str, chromadb.api.ClientAPI] = {}


def chroma_client(settings: Settings) -> chromadb.api.ClientAPI:
    key = str(settings.chroma_dir)
    if key not in _CLIENTS:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = chromadb.PersistentClient(
            path=key, settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        )
    return _CLIENTS[key]


def get_collection(settings: Settings, create: bool = False):
    """Open the persistent collection (cosine space). Raises DatabaseUnavailable."""
    try:
        client = chroma_client(settings)
        if create:
            return client.get_or_create_collection(
                name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
        return client.get_collection(name=settings.chroma_collection, embedding_function=None)
    except Exception as exc:  # noqa: BLE001
        raise DatabaseUnavailable(f"ChromaDB collection unavailable: {exc}") from exc


def get_db_status(settings: Settings | None = None) -> DBStatus:
    settings = settings or get_settings()
    try:
        collection = get_collection(settings)
        count = collection.count()
        sources: set[str] = set()
        if count:
            metas = collection.get(include=["metadatas"])["metadatas"] or []
            sources = {m.get("source", "") for m in metas if m}
        return DBStatus(ok=True, chunks=count, documents=len(sources))
    except DatabaseUnavailable as exc:
        return DBStatus(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return DBStatus(ok=False, error=str(exc))


# =============================================================================
# 6. Orchestration
# =============================================================================
def _ingest_one(path: Path, collection, embedder: Embedder, settings: Settings, report: IngestReport) -> str:
    """Ingest a single file. Returns 'ingested' or 'unchanged'."""
    source = source_name(path, settings)
    doc_hash = file_sha256(path)

    existing = collection.get(where={"source": source}, include=["metadatas"], limit=1)
    if existing["ids"]:
        if (existing["metadatas"][0] or {}).get("doc_hash") == doc_hash:
            return "unchanged"
        collection.delete(where={"source": source})

    pages, extractor = extract_pages(path)
    cleaned = clean_pages(pages)
    if not cleaned:
        report.warnings.append(f"{source}: no extractable text (scanned PDF? OCR is not included)")
        return "ingested"

    chunks, dropped = chunk_document(cleaned, source, doc_hash, settings)
    report.chunks_dropped_noise += dropped
    if not chunks:
        report.warnings.append(f"{source}: all chunks were filtered as noise")
        return "ingested"

    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    good = [(c, v) for c, v in zip(chunks, vectors, strict=True) if v is not None]
    if len(good) < len(chunks):
        report.warnings.append(f"{source}: {len(chunks) - len(good)} chunks could not be embedded")

    batch = 128
    for start in range(0, len(good), batch):
        part = good[start : start + batch]
        collection.upsert(
            ids=[c.id for c, _ in part],
            documents=[c.text for c, _ in part],
            embeddings=[v for _, v in part],
            metadatas=[
                {
                    "source": c.source,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "doc_hash": c.doc_hash,
                    "extractor": extractor,
                }
                for c, _ in part
            ],
        )
    report.chunks_added += len(good)
    return "ingested"


def run_ingestion(
    settings: Settings | None = None,
    *,
    reset: bool = False,
    extract: bool = True,
    progress: ProgressFn | None = None,
) -> IngestReport:
    settings = settings or get_settings()
    ensure_directories(settings)
    notify: ProgressFn = progress or (lambda _p, _m: None)
    report = IngestReport()
    started = time.perf_counter()

    if extract:
        notify(0.0, "Looking for zip archives...")
        for archive in find_archives(settings):
            try:
                notify(0.02, f"Extracting {archive.name} ...")
                target, written, already = extract_archive(archive, settings)
                if not already:
                    report.archives_extracted.append(f"{archive.name} -> {target.name} ({written} files)")
            except (zipfile.BadZipFile, RuntimeError, ValueError, OSError) as exc:
                report.warnings.append(f"Could not extract {archive.name}: {exc}")
                log.error("Extraction failed for %s: %s", archive, exc)

    files = discover_documents(settings)
    report.files_found = len(files)
    if not files:
        report.warnings.append(
            "No PDFs found. Put your dataset .zip in data/raw/ (or extract the PDFs anywhere under data/extracted/)."
        )
        report.elapsed_s = time.perf_counter() - started
        return report

    embedder = Embedder(settings)
    notify(0.04, "Checking embedding model...")
    embedder.ping()

    client = chroma_client(settings)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
    collection = get_collection(settings, create=True)

    for index, path in enumerate(files, start=1):
        notify(0.05 + 0.93 * (index - 1) / len(files), f"[{index}/{len(files)}] {path.name}")
        try:
            outcome = _ingest_one(path, collection, embedder, settings, report)
            if outcome == "unchanged":
                report.files_skipped_unchanged += 1
            else:
                report.files_ingested += 1
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate per-file failures
            log.exception("Failed to ingest %s", path)
            report.files_failed.append(f"{path.name}: {exc}")

    report.chunks_total_in_db = collection.count()
    report.elapsed_s = time.perf_counter() - started
    notify(1.0, "Done")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the syllabus dataset into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="delete the collection and rebuild it")
    parser.add_argument("--skip-extract", action="store_true", help="do not unzip archives in data/raw")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    def show(fraction: float, message: str) -> None:
        print(f"[{fraction * 100:5.1f}%] {message}", flush=True)

    try:
        report = run_ingestion(settings, reset=args.reset, extract=not args.skip_extract, progress=show)
    except EmbeddingError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    print("\n" + report.summary())
    for line in report.archives_extracted:
        print("  extracted:", line)
    for line in report.warnings:
        print("  warning:", line)
    for line in report.files_failed:
        print("  FAILED:", line)
    return 0 if report.chunks_total_in_db else 1


if __name__ == "__main__":
    sys.exit(main())
