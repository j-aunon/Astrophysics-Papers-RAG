from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfInfo:
    path: Path
    file_hash: str
    num_pages: int


def iter_pdf_paths(path_or_dir: str) -> Iterable[Path]:
    path = Path(path_or_dir)
    if path.is_dir():
        for pdf in sorted(path.rglob("*.pdf")):
            yield pdf
        return
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
        return
    raise FileNotFoundError(f"PDF path not found: {path_or_dir}")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf_info(pdf_path: Path) -> PdfInfo:
    try:
        file_hash = sha256_file(pdf_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to hash file: {pdf_path}") from exc

    try:
        doc = fitz.open(pdf_path)
        num_pages = int(doc.page_count)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to open PDF (corrupt or unsupported): {pdf_path}") from exc

    return PdfInfo(path=pdf_path, file_hash=file_hash, num_pages=num_pages)

