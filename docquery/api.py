import os
import unicodedata
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import DEFAULT_GEMINI_MODEL, ENV_FILE, MAX_FILE_BYTES, MAX_FILE_MB, MAX_UPLOAD_FILES
from .embeddings import MiniLMEmbedder
from .gemini import GeminiGenerator, GenerationError
from .pdf import InvalidPdf
from .schemas import AskRequest, AskResponse, ClearResponse, DocumentInfo, HealthResponse, UploadResponse
from .service import DocQuery, EmbeddingUnavailable, IndexCapacityExceeded

MODEL_ERROR = "Could not use the local embedding model. Check the MiniLM model download and retry."


def read_upload(file: UploadFile) -> tuple[str, bytes]:
    basename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char for char in basename if not unicodedata.category(char).startswith("C")).strip()
    if not name.lower().endswith(".pdf") or len(name) > 180:
        raise HTTPException(422, "Each upload needs a PDF filename of at most 180 characters.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(415, f"{name}: expected a PDF upload.")
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{name}: PDFs may be up to {MAX_FILE_MB} MB each.")
    return name, data


def create_app(service: DocQuery | None = None) -> FastAPI:
    if service is None:
        load_dotenv(ENV_FILE, encoding="utf-8-sig")
        min_similarity = float(os.getenv("DOCQUERY_MIN_SIMILARITY", "0.30"))
        if not 0 <= min_similarity <= 1:
            raise ValueError("DOCQUERY_MIN_SIMILARITY must be between 0 and 1.")
        generator = GeminiGenerator(
            os.getenv("GEMINI_API_KEY", "").strip(),
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
        )
        service = DocQuery(MiniLMEmbedder(), generator, min_similarity)

    app = FastAPI(title="DocQuery", description="PDF retrieval and grounded question answering.")

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(gemini_configured=service.generator.configured)

    @app.post("/upload")
    def upload(files: Annotated[list[UploadFile], File()]) -> UploadResponse:
        if not 1 <= len(files) <= MAX_UPLOAD_FILES:
            raise HTTPException(422, f"Upload between 1 and {MAX_UPLOAD_FILES} PDFs at a time.")
        try:
            return service.index_documents([read_upload(file) for file in files])
        except InvalidPdf as exc:
            raise HTTPException(422, str(exc)) from exc
        except IndexCapacityExceeded as exc:
            raise HTTPException(413, "The index is full. Clear the corpus or upload fewer pages.") from exc
        except EmbeddingUnavailable as exc:
            raise HTTPException(503, MODEL_ERROR) from exc

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        try:
            return service.ask(request)
        except EmbeddingUnavailable as exc:
            raise HTTPException(503, MODEL_ERROR) from exc
        except GenerationError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.get("/documents")
    def documents() -> list[DocumentInfo]:
        return service.list_documents()

    @app.delete("/documents")
    def clear() -> ClearResponse:
        return ClearResponse(deleted_documents=service.clear_documents())

    @app.delete("/documents/{document_id}")
    def remove_document(document_id: str) -> ClearResponse:
        if not service.remove_document(document_id):
            raise HTTPException(404, "That PDF is no longer in the workspace.")
        return ClearResponse(deleted_documents=1)

    return app


app = create_app()
