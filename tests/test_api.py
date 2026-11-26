import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from google import genai
from streamlit.testing.v1 import AppTest

from docquery.api import create_app
from docquery.embeddings import EMBEDDING_DIM
from docquery.gemini import GeneratedAnswer, GeminiGenerator
from docquery.service import DocQuery


class FixtureEmbedder:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            text = text.lower()
            dimension = 0 if "vacation" in text else 1 if "remote" in text else 2
            vectors[row, dimension] = 1
        return vectors


def pdf_upload(filename: str, *pages: str) -> tuple:
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            page.insert_text((50, 50), text)
        return "files", (filename, pdf.tobytes(), "application/pdf")


@pytest.fixture
def client():
    service = DocQuery(FixtureEmbedder(), GeminiGenerator("", "unused"), min_similarity=0.30)
    with TestClient(create_app(service)) as client:
        yield client


def test_upload_retrieve_list_and_clear_without_gemini(client):
    handbook = pdf_upload("handbook.pdf", "Remote work is allowed.", "Vacation: 20 days.")
    travel = pdf_upload("travel.pdf", "Travel: receipts due in 30 days.")

    assert client.get("/health").json() == {"status": "ok", "gemini_configured": False}
    empty = client.post("/ask", json={"question": "What is the vacation policy?"}).json()
    assert empty["status"] == "insufficient_evidence" and empty["sources"] == []
    uploaded = client.post("/upload", files=[handbook, travel, handbook]).json()
    assert [item["name"] for item in uploaded["indexed"]] == ["handbook.pdf", "travel.pdf"]
    assert uploaded["duplicates"] == ["handbook.pdf"]
    assert client.post("/upload", files=[handbook]).json()["indexed"] == []
    assert client.get("/documents").json() == uploaded["indexed"]

    payload = {"question": "How much vacation is included?", "top_k": 1, "generate": False}
    result = client.post("/ask", json=payload).json()
    assert result["status"] == "retrieved" and result["answer"] is None
    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert (source["document_name"], source["page"], source["similarity"]) == ("handbook.pdf", 2, 1.0)

    assert client.post("/ask", json={**payload, "generate": True}).status_code == 503
    assert client.delete("/documents").json() == {"deleted_documents": 2}
    assert client.get("/documents").json() == []
    assert client.post("/ask", json=payload).json()["sources"] == []


def test_removing_one_pdf_leaves_the_other_searchable(client):
    handbook = pdf_upload("handbook.pdf", "Vacation allowance.", "Remote work policy.")
    travel = pdf_upload("travel.pdf", "Travel receipts are due in 30 days.")
    first, second = client.post("/upload", files=[handbook, travel]).json()["indexed"]

    route = f"/documents/{first['id']}"
    assert client.delete(route).json() == {"deleted_documents": 1}
    assert client.delete(route).status_code == 404
    assert client.get("/documents").json() == [second]

    payload = {"question": "When are travel receipts due?", "generate": False}
    sources = client.post("/ask", json=payload).json()["sources"]
    assert [(source["document_id"], source["similarity"]) for source in sources] == [(second["id"], 1.0)]
    assert client.post("/upload", files=[handbook]).json()["indexed"] == [first]


def test_invalid_upload_leaves_batch_uncommitted(client):
    existing = pdf_upload("existing.pdf", "Business travel policy.")
    previous = client.post("/upload", files=[existing]).json()["indexed"]
    valid = pdf_upload("good.pdf", "Vacation policy.")
    invalid = "files", ("bad.pdf", b"not a PDF", "application/pdf")
    blank = pdf_upload("blank.pdf", "")

    for rejected in (invalid, blank):
        response = client.post("/upload", files=[valid, rejected])
        assert response.status_code == 422
        assert client.get("/documents").json() == previous
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "Vacation?", "top_k": 0}).status_code == 422


def test_evidence_threshold_and_citation_validation():
    generator = Mock(spec=GeminiGenerator)
    service = DocQuery(FixtureEmbedder(), generator, min_similarity=0.30)
    with TestClient(create_app(service)) as client:
        client.post("/upload", files=[pdf_upload("staff.pdf", "Remote work.", "Vacation: 20 days.")])
        unrelated = client.post("/ask", json={"question": "Where are the planets?"}).json()
        assert unrelated["status"] == "insufficient_evidence"
        generator.generate.assert_not_called()

        payload = {"question": "How much vacation?", "previous_questions": ["What is the policy?"]}
        generator.generate.return_value = GeneratedAnswer(answer="20 days.", citation_ids=[1, 1])
        answer = client.post("/ask", json=payload).json()
        assert answer["status"] == "answered" and answer["citation_ids"] == [1]
        question, evidence, history = generator.generate.call_args.args
        assert question == payload["question"] and history == payload["previous_questions"]
        assert [(source.source_id, source.page) for source in evidence] == [(1, 2)]

        for output in (
            GeneratedAnswer(answer="", citation_ids=[]),
            GeneratedAnswer(answer="Unsupported.", citation_ids=[2]),
        ):
            generator.generate.return_value = output
            refused = client.post("/ask", json=payload).json()
            assert refused["status"] == "insufficient_evidence" and refused["citation_ids"] == []


def test_gemini_request_uses_structured_answer_and_text_only(monkeypatch):
    def gemini_request(request):
        assert request.url.path.endswith("/models/test-model:generateContent")
        body = json.loads(request.content)
        config = body["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert set(config["responseSchema"]["properties"]) == {"answer", "citation_ids"}
        assert json.loads(body["contents"][0]["parts"][0]["text"]) == {
            "question": "How much vacation?",
            "previous_questions": [],
            "passages": [{"id": 1, "text": "Vacation: 20 days."}],
        }
        return httpx.Response(200, json={"candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": '{"answer":"20 days.","citation_ids":[1]}'}],
            },
            "finishReason": "STOP",
        }]})

    real_client = genai.Client

    def client_with_fake_transport(**kwargs):
        kwargs["http_options"].client_args = {"transport": httpx.MockTransport(gemini_request)}
        return real_client(**kwargs)

    monkeypatch.setattr("docquery.gemini.genai.Client", client_with_fake_transport)

    service = DocQuery(FixtureEmbedder(), GeminiGenerator("test-key", "test-model"), 0.30)
    with TestClient(create_app(service)) as client:
        client.post("/upload", files=[pdf_upload("staff.pdf", "Vacation: 20 days.")])
        response = client.post("/ask", json={"question": "How much vacation?"})

    assert response.json()["answer"] == "20 days."


@pytest.mark.parametrize(
    ("gemini_configured", "generation_fails"), [(False, False), (True, False), (True, True)]
)
def test_streamlit_upload_chat_and_clear(client, monkeypatch, gemini_configured, generation_fails):
    _, (name, contents, _) = pdf_upload("staff.pdf", "Remote work.", "Vacation: 20 days.")
    selected_file = SimpleNamespace(name=name, getvalue=lambda: contents)
    # AppTest cannot attach files to the uploader. Supply the selected file for the UI action.
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: [selected_file] if kwargs["key"] == "pdf_upload_0" else [])

    generation_requests = []

    def local_request(method, url, **kwargs):
        path = httpx.URL(url).raw_path.decode()
        kwargs.pop("timeout", None)
        if path == "/health" and gemini_configured:
            return httpx.Response(200, json={"status": "ok", "gemini_configured": True})
        if path == "/ask":
            generate = kwargs["json"]["generate"]
            generation_requests.append(generate)
            if generate:
                assert generation_fails
                return httpx.Response(502, json={"detail": "Gemini is temporarily unavailable."})
        return client.request(method, path, **kwargs)

    monkeypatch.setattr(httpx, "request", local_request)
    app = AppTest.from_file("streamlit_app.py").run()
    assert not app.chat_input
    assert any(item.value == "Add a PDF to get started" for item in app.subheader)
    next(button for button in app.button if button.label == "Add to workspace").click().run()
    assert not app.exception and not app.chat_input[0].disabled
    assert not any(button.label == "Add to workspace" for button in app.button)
    if gemini_configured:
        assert app.radio[0].value == "Ask Gemini"
        if not generation_fails:
            app.radio[0].set_value("Find passages").run()
    else:
        assert not app.radio
    app.chat_input[0].set_value("What is the vacation allowance?").run()
    assert not app.exception and len(app.chat_message) == 2
    assert not any(item.value == "What would you like to find?" for item in app.subheader)
    assert any("Page 2" in source.label for source in app.expander)
    assert generation_requests == ([True, False] if generation_fails else [False])
    if generation_fails:
        assert "Showing matching passages." in app.warning[0].value
    app.run()
    assert len(app.chat_message) == 2
    next(button for button in app.button if button.label == "New chat").click().run()
    assert not app.chat_message
    assert client.get("/documents").json()[0]["name"] == "staff.pdf"


def test_streamlit_separates_cited_sources_from_other_matches():
    app = AppTest.from_string("""
from streamlit_app import render_result
render_result({
    "status": "answered",
    "answer": "20 days of paid leave.",
    "citation_ids": [2],
    "sources": [
        {"source_id": 1, "document_name": "staff.pdf", "page": 1, "similarity": 0.65, "text": "Remote work."},
        {"source_id": 2, "document_name": "staff.pdf", "page": 2, "similarity": 0.62, "text": "Vacation policy."},
    ],
})
""").run()
    assert not app.exception
    assert app.expander[0].label.endswith("Page 2")
    assert app.expander[1].label == "More matches (1)"


def test_sidebar_document_actions(client, monkeypatch):
    first, second = client.post("/upload", files=[
        pdf_upload("staff.pdf", "Vacation allowance."),
        pdf_upload("travel.pdf", "Travel receipts are due in 30 days."),
    ]).json()["indexed"]

    def local_request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        return client.request(method, httpx.URL(url).raw_path.decode(), **kwargs)

    monkeypatch.setattr(httpx, "request", local_request)
    app = AppTest.from_file("streamlit_app.py").run()
    assert not app.exception
    assert app.sidebar.button[0].label == "New chat"
    assert app.sidebar.button(key=f"trash_{first['id']}").label == "Remove staff.pdf"
    app.sidebar.button(key=f"trash_{first['id']}").click().run()
    assert not app.exception
    assert client.get("/documents").json() == [second]
    assert any(item.value == "travel.pdf" for item in app.sidebar.text)
    assert not any(item.value == "staff.pdf" for item in app.sidebar.text)

    next(button for button in app.sidebar.button if button.label == "Remove all PDFs").click().run()
    assert not app.exception
    assert client.get("/documents").json() == []
    assert not app.chat_input
