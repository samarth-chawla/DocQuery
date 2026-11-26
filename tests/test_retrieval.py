import os

import numpy as np
import pymupdf
import pytest

from docquery.embeddings import EMBEDDING_DIM, MiniLMEmbedder
from docquery.pdf import Chunk, chunk_pages, extract_pages
from docquery.retrieval import VectorIndex


def test_split_pages_overlaps_without_crossing_page_boundaries():
    words = [f"word{number}" for number in range(320)]
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for offset in range(0, len(words), 8):
            page.insert_text((40, 40 + 12 * (offset // 8)), " ".join(words[offset:offset + 8]))
        pdf.new_page()  # A blank page must still count toward the physical page number.
        pdf.new_page().insert_text((40, 40), "Another page.")
        pages = extract_pages(pdf.tobytes())

    chunks = chunk_pages(pages, "demo", "demo.pdf", lambda text: len(text.split()))
    first_page_chunks = [chunk for chunk in chunks if chunk.page == 1]
    assert len(first_page_chunks) > 1
    assert set(first_page_chunks[0].text.split()) & set(first_page_chunks[1].text.split())
    assert all(len(chunk.text.split()) <= 200 for chunk in chunks)
    assert [chunk.text for chunk in chunks if chunk.page == 3] == ["Another page."]
    assert {chunk.page for chunk in chunks} == {1, 3}
    assert all(chunk.document_id == "demo" and chunk.document_name == "demo.pdf" for chunk in chunks)


def test_faiss_returns_expected_cosine_scores_and_metadata():
    index = VectorIndex()
    vectors = np.zeros((3, EMBEDDING_DIM), dtype=np.float32)
    vectors[0, :2] = [0.6, 0.8]
    vectors[1, :2] = [1, 0]
    vectors[2, :2] = [-1, 0]
    index.add(
        [Chunk("a", "policy.pdf", 1, "Remote work"), Chunk("a", "policy.pdf", 2, "Vacation"),
         Chunk("b", "budget.pdf", 1, "Travel")],
        vectors,
    )

    results = index.search(vectors[1:2], top_k=8)
    assert [source.similarity for source in results] == pytest.approx([1, 0.6, -1])
    assert [(source.document_name, source.page) for source in results] == [
        ("policy.pdf", 2), ("policy.pdf", 1), ("budget.pdf", 1)
    ]
    assert len(index.search(vectors[1:2], top_k=1)) == 1


@pytest.mark.skipif(os.getenv("DOCQUERY_TEST_MODEL") != "1", reason="set DOCQUERY_TEST_MODEL=1")
def test_real_minilm_embeddings_match_a_paraphrase():
    embedder = MiniLMEmbedder()
    vectors = embedder.encode(
        [
            "Employees receive twenty days of paid vacation per year.",
            "Tomatoes grow best in warm weather and full sunlight.",
            "How much annual leave does the company offer?",
        ]
    )
    assert vectors.shape == (3, EMBEDDING_DIM)
    assert vectors.dtype == np.float32 and vectors.flags.c_contiguous
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1, 1], abs=1e-5)

    index = VectorIndex()
    index.add(
        [Chunk("staff", "staff.pdf", 2, "Vacation policy"), Chunk("plants", "plants.pdf", 1, "Gardening")],
        vectors[:2],
    )
    assert index.search(vectors[2:], top_k=1)[0].page == 2

    long_page = extract_pages(sample_pdf("Annual leave and company holidays. " * 40))
    chunks = chunk_pages(long_page, "staff", "staff.pdf", embedder.count_tokens)
    assert len(chunks) > 1
    assert all(embedder.count_tokens(chunk.text) <= embedder.model.max_seq_length - 2 for chunk in chunks)


def sample_pdf(text: str) -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox((40, 40, 550, 800), text)
        return pdf.tobytes()
