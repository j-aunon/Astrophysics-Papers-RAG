from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def render_page_to_png(
    doc: fitz.Document,
    *,
    page_num: int,
    out_path: Path,
    dpi: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page = doc.load_page(page_num)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(out_path.as_posix())
    return out_path

