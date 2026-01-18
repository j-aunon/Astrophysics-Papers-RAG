from __future__ import annotations

import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from ..config import AppConfig
from ..db import sqlite as db
from .image_extract import extract_figures_from_page
from .ocr import ocr_image
from .page_render import render_page_to_png
from .pdf_loader import load_pdf_info
from .text_extract import extract_page_text
from .vlm_caption import Qwen3VLCaptioner

logger = logging.getLogger(__name__)


def ingest_pdf(
    cfg: AppConfig,
    *,
    pdf_path: Path,
    force: bool = False,
    enable_vlm: bool = True,
) -> db.DocumentRecord:
    """
    Ingest a single PDF into SQLite and write page/figure artifacts to disk.

    Incremental behavior:
    - If file hash is unchanged AND ingested_at is present, skip unless force=True.
    """
    pdf_info = load_pdf_info(pdf_path)

    conn = db.connect(cfg.paths.sqlite_path)
    db.ensure_schema(conn)

    existing = db.get_document_by_path(conn, pdf_info.path.as_posix())
    if existing and (not force) and existing.file_hash == pdf_info.file_hash:
        row = conn.execute(
            "SELECT ingested_at FROM documents WHERE doc_id = ?", (existing.doc_id,)
        ).fetchone()
        if row is not None and row["ingested_at"]:
            logger.info("Skipping ingestion (unchanged): %s", pdf_info.path)
            conn.close()
            return existing

    doc_rec = db.upsert_document(
        conn,
        file_path=pdf_info.path.as_posix(),
        file_name=pdf_info.path.name,
        file_hash=pdf_info.file_hash,
        num_pages=pdf_info.num_pages,
    )

    pages_dir = Path(cfg.paths.pages_dir) / f"doc_{doc_rec.doc_id}"
    figures_dir = Path(cfg.paths.figures_dir) / f"doc_{doc_rec.doc_id}"
    pages_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    captioner = None
    if enable_vlm:
        captioner = Qwen3VLCaptioner(
            cfg.models.vlm_model, device=cfg.models.device, dtype=cfg.models.dtype
        )

    max_pages = cfg.ingestion.max_pages or pdf_info.num_pages
    max_pages = min(max_pages, pdf_info.num_pages)

    try:
        doc = fitz.open(pdf_info.path)
    except Exception as exc:  # noqa: BLE001
        conn.close()
        raise RuntimeError(f"Failed to open PDF (corrupt or unsupported): {pdf_info.path}") from exc

    try:
        for page_num0 in tqdm(range(max_pages), desc=f"Ingest {pdf_info.path.name}", unit="page"):
            text = ""
            try:
                text = extract_page_text(doc, page_num0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Text extraction failed for page %s: %s", page_num0 + 1, exc)

            rendered_path = None
            try:
                rendered_path = render_page_to_png(
                    doc,
                    page_num=page_num0,
                    out_path=pages_dir / f"page_{page_num0 + 1:04d}.png",
                    dpi=cfg.ingestion.render_dpi,
                ).as_posix()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Page render failed for page %s: %s", page_num0 + 1, exc)

            db.upsert_page(
                conn,
                doc_id=doc_rec.doc_id,
                page_num=page_num0 + 1,
                extracted_text=text,
                rendered_image_path=rendered_path,
            )

            figures = []
            try:
                figures = extract_figures_from_page(
                    doc, doc_id=doc_rec.doc_id, page_num=page_num0, out_dir=figures_dir
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Figure extraction failed for page %s: %s", page_num0 + 1, exc)

            for fig in figures:
                ocr_text = ocr_image(fig.image_path)

                vlm_caption = None
                entities_json = None
                bullets_json = None
                if captioner is not None:
                    vlm = captioner.caption(fig.image_path)
                    if vlm is not None:
                        vlm_caption = vlm.caption
                        entities_json = json.dumps(vlm.entities, ensure_ascii=False)
                        bullets_json = json.dumps(vlm.bullets, ensure_ascii=False)

                db.upsert_figure(
                    conn,
                    figure_uid=fig.figure_uid,
                    figure_id=fig.figure_id,
                    doc_id=doc_rec.doc_id,
                    page_num=fig.page_num,
                    image_path=fig.image_path.as_posix(),
                    bbox=fig.bbox,
                    ocr_text=ocr_text,
                    vlm_caption=vlm_caption,
                    vlm_entities_json=entities_json,
                    vlm_bullets_json=bullets_json,
                )

        db.mark_ingested(conn, doc_id=doc_rec.doc_id)
        db.commit(conn)
        return doc_rec
    finally:
        doc.close()
        conn.close()
