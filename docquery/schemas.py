from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from .config import MAX_HISTORY_QUESTIONS, MAX_QUESTION_CHARS

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
]


class AskRequest(BaseModel):
    question: Question
    top_k: int = Field(default=4, ge=1, le=8)
    generate: bool = True
    previous_questions: list[Question] = Field(
        default_factory=list, max_length=MAX_HISTORY_QUESTIONS
    )


class Source(BaseModel):
    source_id: int
    document_id: str
    document_name: str
    page: int
    text: str
    similarity: float


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "retrieved"]
    answer: str | None = None
    sources: list[Source] = Field(default_factory=list)
    citation_ids: list[int] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    id: str
    name: str
    pages: int
    chunks: int


class UploadResponse(BaseModel):
    indexed: list[DocumentInfo]
    duplicates: list[str]


class ClearResponse(BaseModel):
    deleted_documents: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    gemini_configured: bool
