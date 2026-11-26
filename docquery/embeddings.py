from functools import cached_property

import numpy as np

from .config import MODEL_CACHE

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class MiniLMEmbedder:
    @cached_property
    def model(self):
        # Load once on first use so the API can start without a model download.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            MODEL_NAME, device="cpu", cache_folder=str(MODEL_CACHE), trust_remote_code=False
        )

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def encode(self, texts: list[str]) -> np.ndarray:
        # FAISS uses float32. Unit-length vectors turn inner products into cosine scores.
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
