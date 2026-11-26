import faiss
import numpy as np

from .embeddings import EMBEDDING_DIM
from .pdf import Chunk
from .schemas import Source


class VectorIndex:
    def __init__(self) -> None:
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape != (len(chunks), EMBEDDING_DIM):
            raise ValueError("Expected one 384-dimensional vector per chunk.")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def remove_document(self, document_id: str) -> None:
        rows = np.array(
            [row for row, chunk in enumerate(self.chunks) if chunk.document_id == document_id],
            dtype=np.int64,
        )
        if rows.size:
            remaining = [chunk for chunk in self.chunks if chunk.document_id != document_id]
            # IndexFlat compacts the vector rows. Keep chunk metadata in the same order.
            self.index.remove_ids(rows)
            self.chunks = remaining

    def search(self, question_vector: np.ndarray, top_k: int) -> list[Source]:
        count = min(top_k, len(self.chunks))
        if not count:
            return []

        scores, rows = self.index.search(question_vector, count)
        sources = []
        for rank, (score, row) in enumerate(zip(scores[0], rows[0]), start=1):
            # FAISS returns row positions. The matching Chunk stores the source metadata.
            chunk = self.chunks[row]
            sources.append(
                Source(
                    source_id=rank,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page=chunk.page,
                    text=chunk.text,
                    similarity=float(np.clip(score, -1, 1)),
                )
            )
        return sources
