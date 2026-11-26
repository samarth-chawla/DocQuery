import re
from collections.abc import Callable
from dataclasses import dataclass

import pymupdf

from .config import MAX_PDF_PAGES, MAX_PDF_TEXT_CHARS


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    document_id: str
    document_name: str
    page: int
    text: str


class InvalidPdf(ValueError):
    """An upload cannot be indexed as a text PDF."""


def extract_pages(data: bytes) -> list[Page]:
    if not data.startswith(b"%PDF-"):
        raise InvalidPdf("The file does not have a PDF signature.")

    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise InvalidPdf("Password-protected PDFs are unsupported.")
            if pdf.page_count > MAX_PDF_PAGES:
                raise InvalidPdf(f"PDFs may contain up to {MAX_PDF_PAGES} pages.")

            pages = []
            text_chars = 0
            for number, page in enumerate(pdf, start=1):
                # Keep paragraph breaks for the splitter and normalize layout whitespace.
                text = re.sub(r"[^\S\n]+", " ", page.get_text(sort=True)).strip()
                text_chars += len(text)
                if text_chars > MAX_PDF_TEXT_CHARS:
                    raise InvalidPdf("The PDF contains too much extracted text.")
                pages.append(Page(number, text))
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise InvalidPdf("The PDF could not be read.") from exc

    if not any(page.text for page in pages):
        raise InvalidPdf("No selectable text was found. Scanned PDFs need OCR first.")
    return pages


def chunk_pages(
    pages: list[Page],
    document_id: str,
    document_name: str,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    # Import on first upload because LangChain also loads the embedding package.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Aim below MiniLM's 256-token limit. Overlap helps preserve boundary context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=count_tokens,
    )
    return [
        Chunk(document_id, document_name, page.number, text)
        for page in pages
        for text in splitter.split_text(page.text)
    ]
