"""Streamlit front-end for the Self-CRAG agent.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hmac
from pathlib import Path

import streamlit as st

from agent import AgentResult, ChatTurn, SelfCRAGAgent
from config import configure_logging, ensure_directories, get_settings
from ingest import EmbeddingError, get_db_status, run_ingestion

settings = get_settings()
configure_logging(settings)
ensure_directories(settings)

st.set_page_config(page_title="Self-CRAG Syllabus Assistant", page_icon="🎓", layout="wide")

STAGES: list[tuple[str, str, str]] = [
    ("retrieve", "🔎", "Parse & Retrieve"),
    ("grade", "📋", "Document Grading"),
    ("web", "🌐", "Web Fallback"),
    ("generate", "✍️", "Generation"),
    ("check", "🛡️", "Self-Correction Check"),
    ("output", "✅", "Output"),
]
STYLE = {
    "pending": ("#6b7280", "rgba(107,114,128,0.12)"),
    "running": ("#2563eb", "rgba(37,99,235,0.18)"),
    "done": ("#16a34a", "rgba(22,163,74,0.16)"),
    "skipped": ("#9ca3af", "rgba(156,163,175,0.12)"),
    "failed": ("#dc2626", "rgba(220,38,38,0.16)"),
}
MODE_LABELS = {
    "fast": "⚡ Fast - llama3.1:8b does everything",
    "hybrid": "⚖️ Hybrid - 8B answers, 14B grades",
    "rigor": "🧪 Testcase - qwen2.5:14b does everything",
}


# --------------------------------------------------------------------------- helpers
def render_pipeline(states: dict[str, str], note: str = "") -> str:
    chips = []
    for key, icon, label in STAGES:
        status = states.get(key, "pending")
        color, background = STYLE[status]
        pulse = " ⏳" if status == "running" else ""
        chips.append(
            f'<span style="display:inline-block;margin:2px 0;padding:4px 10px;border-radius:14px;'
            f'border:1px solid {color};background:{background};color:{color};font-size:0.82rem;'
            f'font-weight:600;white-space:nowrap">{icon} {label}{pulse}</span>'
        )
    arrow = '<span style="color:#9ca3af;margin:0 4px">→</span>'
    note_html = f'<div style="margin-top:4px;font-size:0.8rem;opacity:0.75">{note}</div>' if note else ""
    return f'<div style="line-height:2">{arrow.join(chips)}</div>{note_html}'


@st.cache_resource(show_spinner=False)
def get_agent() -> SelfCRAGAgent:
    return SelfCRAGAgent(settings)


@st.cache_data(ttl=20, show_spinner=False)
def cached_health(mode: str) -> dict:
    return get_agent().health(mode).model_dump()  # type: ignore[arg-type]


@st.cache_data(ttl=20, show_spinner=False)
def cached_db_status() -> dict:
    return get_db_status(settings).model_dump()


def check_password() -> bool:
    secret = settings.app_password.get_secret_value() if settings.app_password else ""
    if not secret or st.session_state.get("authed"):
        return True
    st.title("🔒 Self-CRAG Syllabus Assistant")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), secret.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong password")
    return False


def yes_no(value: bool | None) -> str:
    return "—" if value is None else ("✅ PASS" if value else "❌ FAIL")


def do_ingest(reset: bool) -> None:
    bar = st.sidebar.progress(0.0, text="Starting ingestion...")
    try:
        report = run_ingestion(
            settings,
            reset=reset,
            progress=lambda fraction, message: bar.progress(min(max(fraction, 0.0), 1.0), text=message),
        )
        st.session_state["ingest_report"] = report.model_dump()
    except EmbeddingError as exc:
        st.session_state["ingest_report"] = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        st.session_state["ingest_report"] = {"error": f"Ingestion crashed: {exc}"}
    finally:
        bar.empty()
        cached_db_status.clear()
        cached_health.clear()


# --------------------------------------------------------------------------- main
if not check_password():
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("ingest_report", None)

# ---- sidebar -----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control panel")
    mode = st.radio(
        "LLM mode",
        options=list(MODE_LABELS),
        format_func=MODE_LABELS.get,
        index=list(MODE_LABELS).index(settings.llm_mode),
        help="Fast fits a 6 GB GPU. Testcase mode runs the 14B model, partly on CPU, so it is slower.",
    )
    routing = settings.resolve_models(mode)  # type: ignore[arg-type]
    st.markdown(
        f"**Active LLM**  \n🧠 Generation: `{routing.generation}`  \n🧪 Grading: `{routing.grading}`  \n"
        f"🧬 Embeddings: `{settings.embedding_model}`"
    )
    max_retries = st.slider("Max self-correction retries", 0, 5, settings.max_correction_retries)
    allow_web = st.toggle(
        "Web fallback (Tavily)",
        value=settings.tavily_enabled,
        disabled=not settings.tavily_enabled,
        help="Set TAVILY_API_KEY in .env to enable.",
    )

    st.divider()
    st.subheader("🛡️ Self-correction metrics")
    last: AgentResult | None = (
        AgentResult.model_validate(st.session_state["last_result"]) if st.session_state["last_result"] else None
    )
    if last is None:
        st.caption("Ask a question to see the metrics.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Hallucination check", yes_no(last.hallucination_check_passed))
        c2.metric("Answer quality", yes_no(last.answer_quality_passed))
        c3, c4 = st.columns(2)
        c3.metric("Retries used", last.retries_used)
        c4.metric("Route", last.route.value.upper())
        st.caption(
            f"Verified: {'yes' if last.verified else 'no'} · Abstained: {'yes' if last.abstained else 'no'} · "
            f"{last.elapsed_s:.1f}s · models: {', '.join(last.models_used) or '—'}"
        )
        for number, attempt in enumerate(last.attempts, start=1):
            with st.expander(f"Attempt {number}: " + ("✅ passed" if attempt.passed else "↻ needed correction" if not attempt.abstained else "abstained"), expanded=False):
                st.write(f"Grounded: {yes_no(attempt.grounded)} (conf {attempt.grounded_confidence or 0:.2f})")
                st.write(f"Answers question: {yes_no(attempt.answers_question)} · complete: {yes_no(attempt.complete)}")
                if attempt.unsupported_claims:
                    st.write("**Unsupported claims**")
                    for claim in attempt.unsupported_claims:
                        st.write(f"- {claim}")
                if attempt.missing_points:
                    st.write("**Missing points**")
                    for point in attempt.missing_points:
                        st.write(f"- {point}")

        local_docs = [d for d in last.docs if d.origin == "local"]
        web_docs = [d for d in last.docs if d.origin == "web"]
        with st.expander(f"📄 Retrieved chunks ({len(local_docs)})"):
            if last.expanded_queries:
                st.caption("Queries: " + " | ".join([last.standalone_query, *last.expanded_queries]))
            for doc in local_docs:
                badge = "🟢 relevant" if doc.relevant else "🔴 irrelevant"
                st.markdown(f"**{doc.label}** — {badge}")
                st.progress(min(max(doc.confidence, 0.0), 1.0), text=f"confidence {doc.confidence:.2f} · cosine {doc.similarity or 0:.2f}")
                if doc.grade_reason:
                    st.caption(doc.grade_reason)
                st.text(doc.text[:700] + ("..." if len(doc.text) > 700 else ""))
                st.divider()
        with st.expander(f"🌐 Web fallback sources ({len(web_docs)})"):
            if not web_docs:
                st.caption("Web search was not used for this answer.")
            for doc in web_docs:
                st.markdown(f"**[{doc.source}]({doc.url})**" if doc.url else f"**{doc.source}**")
                st.caption(f"Tavily score {doc.confidence:.2f}")
                st.text(doc.text[:500] + ("..." if len(doc.text) > 500 else ""))

    st.divider()
    st.subheader("🗄️ Database")
    health = cached_health(mode)
    db = cached_db_status()
    st.markdown(
        f"{'🟢' if health['ollama_ok'] else '🔴'} Ollama  \n"
        f"{'🟢' if db['ok'] else '🔴'} ChromaDB — {db.get('chunks', 0)} chunks / {db.get('documents', 0)} documents  \n"
        f"{'🟢' if health['tavily_configured'] else '⚪'} Tavily web search"
    )
    if not health["ollama_ok"]:
        st.error("Ollama is not reachable. Start it with `ollama serve`.")
    elif health["missing_models"]:
        st.warning("Missing models: " + ", ".join(f"`ollama pull {m}`" for m in health["missing_models"]))
    if not db["ok"] or db.get("chunks", 0) == 0:
        st.warning("The knowledge base is empty. Put the dataset zip in data/raw/ and press Re-ingest.")

    if st.button("🔄 Re-ingest Kaggle data", use_container_width=True, help="Unzips new archives and adds new/changed PDFs"):
        do_ingest(reset=False)
    if st.button("🗑️ Rebuild from scratch", use_container_width=True, help="Deletes the vector store first"):
        do_ingest(reset=True)
    if st.button("↻ Refresh status", use_container_width=True):
        cached_db_status.clear()
        cached_health.clear()
        st.rerun()

    report = st.session_state["ingest_report"]
    if report:
        if "error" in report:
            st.error(report["error"])
        else:
            st.success(
                f"{report['files_ingested']} ingested, {report['files_skipped_unchanged']} unchanged, "
                f"+{report['chunks_added']} chunks ({report['chunks_total_in_db']} total)"
            )
            for line in report["archives_extracted"]:
                st.caption("📦 " + line)
            for line in [*report["warnings"], *report["files_failed"]]:
                st.caption("⚠️ " + line)

    with st.expander("📦 Add dataset via browser"):
        uploads = st.file_uploader("Zip or PDF files", type=["zip", "pdf"], accept_multiple_files=True)
        if uploads and st.button("Save to data/raw"):
            for upload in uploads:
                target = settings.raw_dir / Path(upload.name).name
                target.write_bytes(upload.getbuffer())
            st.success(f"Saved {len(uploads)} file(s). Now press Re-ingest.")

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_result"] = None
        st.rerun()

# ---- chat area ----------------------------------------------------------------------
st.title("🎓 Self-Correcting Agentic RAG")
st.caption("Syllabus assistant · local vector search → document grading → web fallback → answer → hallucination & relevance checks")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("stages"):
            st.markdown(render_pipeline(message["stages"]), unsafe_allow_html=True)
        st.markdown(message["content"])
        for warning in message.get("warnings", []):
            st.warning(warning)

prompt = st.chat_input("Ask about your syllabus...")
if prompt:
    history = [ChatTurn(role=m["role"], content=m["content"]) for m in st.session_state["messages"][-6:]]
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pipeline_slot = st.empty()
        answer_slot = st.empty()
        states = {key: "pending" for key, _, _ in STAGES}
        note = "Starting..."
        text = ""
        final: AgentResult | None = None
        pipeline_slot.markdown(render_pipeline(states, note), unsafe_allow_html=True)

        for event in get_agent().run(prompt, history=history, mode=mode, max_retries=max_retries, allow_web=allow_web):  # type: ignore[arg-type]
            if event.type == "stage" and event.stage and event.status:
                states[event.stage.value] = event.status.value
                note = event.message
                pipeline_slot.markdown(render_pipeline(states, note), unsafe_allow_html=True)
            elif event.type == "token":
                text += event.token
                answer_slot.markdown(text + "▌")
            elif event.type == "draft_reset":
                text = ""
                answer_slot.markdown("_Self-correction: re-drafting the answer..._")
            elif event.type == "final" and event.result:
                final = event.result

        if final is not None:
            pipeline_slot.markdown(render_pipeline(states), unsafe_allow_html=True)
            answer_slot.markdown(final.answer)
            st.session_state["messages"].append(
                {"role": "assistant", "content": final.answer, "stages": states, "warnings": final.warnings}
            )
            st.session_state["last_result"] = final.model_dump(mode="json")
    st.rerun()
