from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedFigure:
    figure_uid: str
    figure_id: str
    page_num: int
    image_path: Path
    bbox: tuple[float, float, float, float] | None


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def extract_figures_from_page(
    doc: fitz.Document,
    *,
    doc_id: int,
    page_num: int,
    out_dir: Path,
) -> list[ExtractedFigure]:
    """
    Extracts raster images found on the page.

    Note: PDFs vary widely; this uses PyMuPDF's text dictionary blocks to get bbox when available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    page = doc.load_page(page_num)
    data = page.get_text("dict")
    blocks = data.get("blocks", []) if isinstance(data, dict) else []

    extracted: list[ExtractedFigure] = []
    for block_idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if block.get("type") != 1:
            continue
        xref = block.get("xref")
        bbox_raw = block.get("bbox")
        bbox = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
        if not isinstance(xref, int):
            continue

        try:
            image = doc.extract_image(xref)
            img_bytes: bytes = image.get("image", b"")
            img_ext = str(image.get("ext", "png"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract image xref=%s on page %s: %s", xref, page_num + 1, exc)
            continue

        if not img_bytes:
            continue

        content_hash = _sha1_bytes(img_bytes)
        figure_id = f"f{block_idx}"
        figure_uid = f"{doc_id}:{page_num + 1}:{figure_id}:{content_hash[:10]}"
        out_path = out_dir / f"{figure_uid.replace(':', '_')}.{img_ext}"
        try:
            out_path.write_bytes(img_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write figure image: %s (%s)", out_path, exc)
            continue

        extracted.append(
            ExtractedFigure(
                figure_uid=figure_uid,
                figure_id=figure_id,
                page_num=page_num + 1,
                image_path=out_path,
                bbox=bbox,
            )
        )

    return extracted
