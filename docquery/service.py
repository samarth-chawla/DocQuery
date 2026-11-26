from hashlib import sha256
from threading import Lock

from .config import MAX_INDEX_CHUNKS
from .embeddings import MiniLMEmbedder
from .gemini import GeminiGenerator
from .pdf import Chunk, InvalidPdf, chunk_pages, extract_pages
from .retrieval import VectorIndex
from .schemas import AskRequest, AskResponse, DocumentInfo, UploadResponse

NO_ANSWER = "I couldn't find enough evidence in the indexed PDFs to answer that question."


class EmbeddingUnavailable(Exception):
    """The local model could not load or encode text."""


class IndexCapacityExceeded(Exception):
    """Adding this batch would exceed the in-memory index capacity."""


class DocQuery:
    def __init__(
        self, embedder: MiniLMEmbedder, generator: GeminiGenerator, min_similarity: float
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.min_similarity = min_similarity
        self.index = VectorIndex()
        self.documents: dict[str, DocumentInfo] = {}
        # Serialize access to the PDF parser, local model and mutable FAISS index.
        self.lock = Lock()

    def index_documents(self, uploads: list[tuple[str, bytes]]) -> UploadResponse:
        with self.lock:
            new_documents: dict[str, DocumentInfo] = {}
            chunks: list[Chunk] = []
            duplicates: list[str] = []

            for name, data in uploads:
                document_id = sha256(data).hexdigest()
                if document_id in self.documents or document_id in new_documents:
                    duplicates.append(name)
                    continue
                try:
                    pages = extract_pages(data)
                except InvalidPdf as exc:
                    raise InvalidPdf(f"{name}: {exc}") from exc
                try:
                    document_chunks = chunk_pages(pages, document_id, name, self.embedder.count_tokens)
                except Exception as exc:
                    raise EmbeddingUnavailable from exc
                if len(self.index.chunks) + len(chunks) + len(document_chunks) > MAX_INDEX_CHUNKS:
                    raise IndexCapacityExceeded
                chunks.extend(document_chunks)
                new_documents[document_id] = DocumentInfo(
                    id=document_id, name=name, pages=len(pages), chunks=len(document_chunks)
                )

            # Commit the whole batch after extraction, validation and embedding succeed.
            if chunks:
                self.index.add(chunks, self.embed([chunk.text for chunk in chunks]))
                self.documents.update(new_documents)
            return UploadResponse(indexed=list(new_documents.values()), duplicates=duplicates)

    def embed(self, texts: list[str]):
        try:
            return self.embedder.encode(texts)
        except Exception as exc:
            raise EmbeddingUnavailable from exc

    def list_documents(self) -> list[DocumentInfo]:
        with self.lock:
            return list(self.documents.values())

    def remove_document(self, document_id: str) -> bool:
        with self.lock:
            if document_id not in self.documents:
                return False
            self.index.remove_document(document_id)
            del self.documents[document_id]
            return True

    def clear_documents(self) -> int:
        with self.lock:
            count = len(self.documents)
            self.index = VectorIndex()
            self.documents.clear()
            return count

    def ask(self, request: AskRequest) -> AskResponse:
        with self.lock:
            sources = []
            if self.documents:
                question_vector = self.embed([request.question])
                sources = self.index.search(question_vector, request.top_k)

        if not request.generate:
            return AskResponse(status="retrieved", sources=sources)

        evidence = [source for source in sources if source.similarity >= self.min_similarity]
        if evidence:
            # Generation uses a snapshot of the retrieved passages outside the index lock.
            result = self.generator.generate(request.question, evidence, request.previous_questions)
            allowed_ids = {source.source_id for source in evidence}
            answer = result.answer.strip()
            if answer and result.citation_ids and set(result.citation_ids) <= allowed_ids:
                return AskResponse(
                    status="answered",
                    answer=answer,
                    sources=sources,
                    citation_ids=list(dict.fromkeys(result.citation_ids)),
                )

        return AskResponse(status="insufficient_evidence", answer=NO_ANSWER, sources=sources)
