from __future__ import annotations

import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_page_text(doc: fitz.Document, page_num: int) -> str:
    page = doc.load_page(page_num)
    text = page.get_text("text") or ""
    return text.strip()

