import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from docquery.config import (
    DEFAULT_API_URL,
    ENV_FILE,
    MAX_FILE_MB,
    MAX_HISTORY_QUESTIONS,
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_FILES,
)

load_dotenv(ENV_FILE, encoding="utf-8-sig")
API_URL = (os.getenv("DOCQUERY_API_URL") or DEFAULT_API_URL).rstrip("/")

ANSWER_MODE = "Ask Gemini"
SEARCH_MODE = "Find passages"
SCORE_HELP = "Cosine similarity ranges from -1 to 1. Higher scores mean the passage is closer in meaning to the question. Use the passage text to check the answer."

APP_STYLE = """
<style>
.stMainBlockContainer { max-width: 56rem; padding-top: 4.5rem; }
.st-key-docquery_intro {
    background: linear-gradient(115deg, #e8f5ef 0%, #f6faf8 100%);
    border: 1px solid #d4e8dd;
    border-radius: 1.15rem;
    padding: clamp(1rem, 3vw, 2rem);
}
.st-key-docquery_intro h1 { font-size: clamp(2rem, 5vw, 3.05rem); padding: 0; color: #182928; letter-spacing: -0.04em; }
.st-key-docquery_intro p { color: #36554b; }
.st-key-docquery_empty { text-align: center; padding: 1.15rem; }
.st-key-docquery_composer { background: rgb(18 103 91 / 5%); }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stText"] { white-space: pre-wrap; overflow-wrap: anywhere; }
[class*="st-key-trash_"] button [data-testid="stMarkdownContainer"] {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
}
@media (max-width: 600px) {
    .stMainBlockContainer { padding-top: 4.25rem; }
}
</style>
"""


class ApiError(Exception):
    """An API failure that can be shown in the UI."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def call_api(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=120.0, **kwargs)
    except httpx.TimeoutException as exc:
        raise ApiError("The API took too long to respond. Refresh the document list before retrying.") from exc
    except httpx.RequestError as exc:
        raise ApiError("Cannot reach the DocQuery API. Check that FastAPI is running.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(f"The API returned an unreadable response (HTTP {response.status_code}).") from exc
    if response.is_error:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = detail if isinstance(detail, str) else f"Request failed (HTTP {response.status_code})."
        raise ApiError(message, response.status_code)
    return payload


def ask_question(question: str, generate: bool, previous_questions: list[str]) -> dict:
    payload = {
        "question": question,
        "generate": generate,
        "previous_questions": previous_questions if generate else [],
    }
    try:
        return call_api("POST", "/ask", json=payload)
    except ApiError as exc:
        if not generate or exc.status_code not in {429, 502, 503, 504}:
            raise
        result = call_api("POST", "/ask", json={"question": question, "generate": False})
        result["generation_notice"] = f"{exc} Showing matching passages."
        return result


def render_passages(sources: list[dict], label: str, *, expanded: bool = False) -> None:
    with st.expander(label, expanded=expanded):
        for source in sources:
            with st.container(border=True, gap="xsmall"):
                st.text(source["document_name"])
                st.caption(
                    f"Page {source['page']} · Similarity {source['similarity']:.3f}",
                    help=SCORE_HELP,
                )
                st.text(source["text"])


def render_result(result: dict) -> None:
    if notice := result.get("generation_notice"):
        st.warning(notice)
    sources = result["sources"]
    if result["status"] == "retrieved":
        if sources:
            render_passages([sources[0]], f"Best match · Page {sources[0]['page']}", expanded=True)
            sources = sources[1:]
        else:
            st.text("No matching passages found.")
    else:
        st.text(result["answer"])
        citations = set(result["citation_ids"])
        cited = [source for source in sources if source["source_id"] in citations]
        if cited:
            label = f"Cited source · Page {cited[0]['page']}" if len(cited) == 1 else f"Cited sources ({len(cited)})"
            render_passages(cited, label)
            sources = [source for source in sources if source["source_id"] not in citations]
    if sources:
        label = "More matches" if result["status"] in {"answered", "retrieved"} else "Related passages"
        render_passages(sources, f"{label} ({len(sources)})")


def upload_pdfs() -> None:
    if notice := st.session_state.pop("upload_notice", None):
        st.success(notice)
    upload_version = st.session_state.get("upload_version", 0)
    files = st.file_uploader(
        "Add PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_upload_{upload_version}",
        help=f"Add up to {MAX_UPLOAD_FILES} text-based PDFs at a time. Each can be up to {MAX_FILE_MB} MB.",
    )
    if len(files) > MAX_UPLOAD_FILES:
        st.warning(f"Choose up to {MAX_UPLOAD_FILES} PDFs at a time.")
        return
    if not files or not st.button("Add to workspace", type="primary", width="stretch"):
        return

    try:
        with st.spinner("Preparing your PDFs…"):
            uploads = [("files", (file.name, file.getvalue(), "application/pdf")) for file in files]
            result = call_api("POST", "/upload", files=uploads)
    except ApiError as exc:
        st.error(str(exc))
        return

    messages = []
    if result["indexed"]:
        count = len(result["indexed"])
        messages.append("1 PDF is ready to search." if count == 1 else f"{count} PDFs are ready to search.")
    if result["duplicates"]:
        count = len(result["duplicates"])
        messages.append("1 PDF was already added." if count == 1 else f"{count} PDFs were already added.")
    st.session_state.upload_notice = " ".join(messages)
    st.session_state.upload_version = upload_version + 1
    st.rerun()


def remove_pdfs(document: dict | None = None) -> None:
    path = f"/documents/{document['id']}" if document else "/documents"
    try:
        call_api("DELETE", path)
    except ApiError as exc:
        st.error(str(exc))
        return

    if document:
        st.session_state.document_notice = f"Removed {document['name']}."
    else:
        st.session_state.chat = []
        st.session_state.document_notice = "Removed all PDFs."
    st.rerun()


def sidebar() -> tuple[list[dict], bool]:
    with st.sidebar:
        if st.button("New chat", icon=":material/add:", width="stretch"):
            st.session_state.chat = []

        st.header("Documents")
        upload_pdfs()
        if notice := st.session_state.pop("document_notice", None):
            st.success(notice)

        try:
            documents = call_api("GET", "/documents")
            health = call_api("GET", "/health")
        except ApiError as exc:
            st.error(str(exc))
            return [], False

        if not documents:
            st.caption("No PDFs added yet.")
            return documents, health["gemini_configured"]

        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.markdown("**Ready to search**", width="content")
            with st.popover("Remove all", type="tertiary"):
                st.caption("Remove all PDFs from the workspace and reset this chat?")
                if st.button("Remove all PDFs", width="stretch"):
                    remove_pdfs()

        for document in documents:
            with st.container(border=True, gap="xsmall"):
                details, actions = st.columns([5, 1], gap="small", vertical_alignment="center")
                with details:
                    st.text(document["name"])
                    pages = document["pages"]
                    st.caption("1 page" if pages == 1 else f"{pages} pages")
                with actions:
                    label = f"Remove {document['name']}"
                    if st.button(label, icon=":material/delete:", help=label,
                                 type="tertiary", key=f"trash_{document['id']}"):
                        remove_pdfs(document)

        return documents, health["gemini_configured"]


def render_composer(has_documents: bool, gemini_configured: bool) -> tuple[str | None, bool]:
    if not has_documents:
        return None, False

    with st.container(border=True, key="docquery_composer", gap="xsmall"):
        if gemini_configured:
            mode = st.radio(
                "Response mode", [ANSWER_MODE, SEARCH_MODE], horizontal=True,
                label_visibility="collapsed", key="response_mode",
            )
            generate = mode == ANSWER_MODE
        else:
            generate = False

        if generate:
            st.caption("Gemini receives your question, recent questions, and matching passages.")
        else:
            st.caption("Local search shows matching text and page numbers.")

        question = st.chat_input(
            "Ask about your PDFs" if generate else "Search your PDFs",
            key="document_question", max_chars=MAX_QUESTION_CHARS,
        )
        return question, generate


def main() -> None:
    st.set_page_config(page_title="DocQuery – AI Document Assistant", page_icon=":material/description:")
    st.html(APP_STYLE)
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.container(key="docquery_intro", gap="xsmall"):
        st.title("DocQuery", anchor=False)
        st.markdown("Ask your PDFs and check the source.")

    documents, gemini_configured = sidebar()
    conversation = st.container()
    submitted, generate = render_composer(bool(documents), gemini_configured)
    with conversation:
        for turn in st.session_state.chat:
            with st.chat_message("user", avatar=":material/person:"):
                st.text(turn["question"])
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                if "error" in turn:
                    st.error(turn["error"])
                else:
                    render_result(turn["result"])

        if not st.session_state.chat and not (submitted or "").strip():
            with st.container(key="docquery_empty", border=True):
                if documents:
                    st.subheader("What would you like to find?")
                    st.markdown("Try a question about a policy, deadline, or detail in your PDFs.")
                else:
                    st.subheader("Add a PDF to get started")
                    st.markdown("Open **Documents**, choose a PDF, and select **Add to workspace**.")

    if not submitted or not (question := submitted.strip()):
        return

    previous_questions = [
        turn["question"] for turn in st.session_state.chat
        if turn.get("result", {}).get("status") == "answered"
    ][-MAX_HISTORY_QUESTIONS:]
    turn = {"question": question}
    with conversation:
        with st.chat_message("user", avatar=":material/person:"):
            st.text(question)
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            try:
                with st.spinner("Finding evidence and drafting an answer…" if generate else "Finding matching passages…"):
                    turn["result"] = ask_question(question, generate, previous_questions)
                render_result(turn["result"])
            except ApiError as exc:
                turn["error"] = str(exc)
                st.error(turn["error"])
    st.session_state.chat.append(turn)


if __name__ == "__main__":
    main()
