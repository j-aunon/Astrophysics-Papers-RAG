from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..db import sqlite as db
from ..lang_policy import enforce_english_output
from ..retriever.hybrid import HybridResults

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextPackage:
    text_items: list[dict[str, Any]]
    figure_items: list[dict[str, Any]]
    context_text: str


def build_context(cfg, *, results: HybridResults, max_text_items: int = 12, max_pages: int = 8) -> ContextPackage:
    conn = db.connect(cfg.paths.sqlite_path)
    db.ensure_schema(conn)

    # Prefer fused ranking, but fall back to raw text results.
    chunk_uids: list[str] = []
    pages: set[tuple[int, int]] = set()

    for fused in results.fused:
        if fused.source == "text" and len(chunk_uids) < max_text_items:
            uid = str(fused.payload.get("chunk_uid") or fused.key)
            if uid and uid not in chunk_uids:
                chunk_uids.append(uid)
            doc_id = fused.payload.get("doc_id")
            page_num = fused.payload.get("page_num")
            if doc_id and page_num:
                pages.add((int(doc_id), int(page_num)))
        elif fused.source == "visual" and len(pages) < max_pages:
            doc_id = fused.payload.get("doc_id")
            page_num = fused.payload.get("page_num")
            if doc_id and page_num:
                pages.add((int(doc_id), int(page_num)))

    if not chunk_uids:
        for r in results.text[:max_text_items]:
            chunk_uids.append(r.chunk_uid)
            doc_id = r.metadata.get("doc_id")
            page_num = r.metadata.get("page_num")
            if doc_id and page_num:
                pages.add((int(doc_id), int(page_num)))

    text_items = _fetch_chunks(conn, chunk_uids)
    figure_items = _fetch_figures_for_pages(conn, sorted(pages)[:max_pages])

    text_evidence = "\n".join(
        [
            f"- [{item['doc_id']}:{item['page_num']}:{item['chunk_id']}] {item['text']}"
            for item in text_items
        ]
    )
    figure_evidence = "\n".join(
        [
            _format_figure_evidence(item)
            for item in figure_items
        ]
    )

    context_text = enforce_english_output(
        f"TEXT EVIDENCE:\n{text_evidence}\n\nFIGURE EVIDENCE:\n{figure_evidence}\n"
    )
    conn.close()
    return ContextPackage(text_items=text_items, figure_items=figure_items, context_text=context_text)


def _fetch_chunks(conn: sqlite3.Connection, chunk_uids: list[str]) -> list[dict[str, Any]]:
    if not chunk_uids:
        return []
    rows = conn.execute(
        f"""
        SELECT chunk_uid, chunk_id, doc_id, page_num, section_name, text
        FROM text_chunks
        WHERE chunk_uid IN ({",".join(["?"] * len(chunk_uids))})
        """,
        chunk_uids,
    ).fetchall()
    by_uid = {str(r["chunk_uid"]): r for r in rows}
    out: list[dict[str, Any]] = []
    for uid in chunk_uids:
        r = by_uid.get(uid)
        if not r:
            continue
        out.append(
            {
                "chunk_uid": str(r["chunk_uid"]),
                "chunk_id": str(r["chunk_id"]),
                "doc_id": int(r["doc_id"]),
                "page_num": int(r["page_num"]),
                "section_name": str(r["section_name"] or ""),
                "text": enforce_english_output(str(r["text"])),
            }
        )
    return out


def _fetch_figures_for_pages(conn: sqlite3.Connection, pages: list[tuple[int, int]]) -> list[dict[str, Any]]:
    if not pages:
        return []
    clauses = " OR ".join(["(doc_id = ? AND page_num = ?)"] * len(pages))
    params: list[Any] = []
    for doc_id, page_num in pages:
        params.extend([doc_id, page_num])
    rows = conn.execute(
        f"""
        SELECT figure_uid, figure_id, doc_id, page_num, image_path, ocr_text, vlm_caption, vlm_entities_json, vlm_bullets_json
        FROM figures
        WHERE {clauses}
        ORDER BY doc_id ASC, page_num ASC
        """,
        params,
    ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        entities = []
        bullets = []
        try:
            if r["vlm_entities_json"]:
                entities = json.loads(str(r["vlm_entities_json"]))
            if r["vlm_bullets_json"]:
                bullets = json.loads(str(r["vlm_bullets_json"]))
        except Exception:  # noqa: BLE001
            pass
        out.append(
            {
                "figure_uid": str(r["figure_uid"]),
                "figure_id": str(r["figure_id"]),
                "doc_id": int(r["doc_id"]),
                "page_num": int(r["page_num"]),
                "image_path": str(r["image_path"]),
                "ocr_text": enforce_english_output(str(r["ocr_text"] or "")),
                "vlm_caption": enforce_english_output(str(r["vlm_caption"] or "")),
                "vlm_entities": [enforce_english_output(str(x)) for x in (entities or [])],
                "vlm_bullets": [enforce_english_output(str(x)) for x in (bullets or [])],
            }
        )
    return out


def _format_figure_evidence(item: dict[str, Any]) -> str:
    cite = f"[{item['doc_id']}:{item['page_num']}:{item['figure_id']}]"
    caption = item.get("vlm_caption") or ""
    ocr = item.get("ocr_text") or ""
    bullets = item.get("vlm_bullets") or []
    bullet_lines = "\n".join([f"  - {b}" for b in bullets]) if bullets else "  - (No VLM bullets available.)"
    return (
        f"- {cite} caption: {caption}\n"
        f"  OCR: {ocr}\n"
        f"{bullet_lines}"
    )

