from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from ..config import AppConfig
from ..db import sqlite as db
from .chroma_client import get_client, get_text_collection
from .chunking import chunk_text_section_aware
from .colpali_index import ColPaliPageIndex
from .text_embed import TextEmbedder

logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200


def index_document_text(cfg: AppConfig, *, doc_id: int, force: bool = False) -> None:
    conn = db.connect(cfg.paths.sqlite_path)
    db.ensure_schema(conn)

    doc_row = conn.execute(
        "SELECT file_hash, text_indexed_at, text_indexed_hash FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if doc_row is None:
        conn.close()
        raise ValueError(f"Unknown doc_id: {doc_id}")

    if (not force) and doc_row["text_indexed_at"] and doc_row["text_indexed_hash"] == doc_row["file_hash"]:
        logger.info("Skipping text indexing (already indexed): doc_id=%s", doc_id)
        conn.close()
        return

    pages = conn.execute(
        "SELECT page_num, extracted_text FROM pages WHERE doc_id = ? ORDER BY page_num ASC",
        (doc_id,),
    ).fetchall()

    embedder = TextEmbedder(cfg.models.text_embedding_model, device=cfg.models.device)
    chroma = get_client(cfg.paths.chroma_dir)
    collection = get_text_collection(chroma, name="text_chunks")

    # Remove previous embeddings for this doc_id (idempotent rebuild).
    try:
        collection.delete(where={"doc_id": int(doc_id)})
    except Exception:  # noqa: BLE001
        # Older Chroma versions may fail if nothing matches; ignore.
        pass

    total_chunks = 0
    for row in tqdm(pages, desc=f"Chunk+embed doc_id={doc_id}", unit="page"):
        page_num = int(row["page_num"])
        text = str(row["extracted_text"] or "")

        chunks = chunk_text_section_aware(
            doc_id=doc_id,
            page_num=page_num,
            text=text,
            max_chars=CHUNK_MAX_CHARS,
            overlap_chars=CHUNK_OVERLAP_CHARS,
        )
        db.replace_chunks(
            conn,
            doc_id=doc_id,
            page_num=page_num,
            chunks=[
                {
                    "chunk_uid": c.chunk_uid,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "section_name": c.section_name,
                    "text": c.text,
                    "text_offset_start": c.text_offset_start,
                    "text_offset_end": c.text_offset_end,
                }
                for c in chunks
            ],
        )

        if not chunks:
            continue

        chunk_uids = [c.chunk_uid for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts, batch_size=16)
        metadatas = [
            {
                "doc_id": int(doc_id),
                "page_num": int(page_num),
                "chunk_id": c.chunk_id,
                "chunk_uid": c.chunk_uid,
                "chunk_index": int(c.chunk_index),
                "section_name": c.section_name or "",
                "text_offset": int(c.text_offset_start),
            }
            for c in chunks
        ]
        collection.add(
            ids=chunk_uids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )
        total_chunks += len(chunks)

    db.mark_text_indexed(conn, doc_id=doc_id)
    db.commit(conn)
    conn.close()
    logger.info("Text indexing complete: doc_id=%s chunks=%s", doc_id, total_chunks)


def index_document_visual(cfg: AppConfig, *, doc_id: int, force: bool = False) -> None:
    conn = db.connect(cfg.paths.sqlite_path)
    db.ensure_schema(conn)

    doc_row = conn.execute(
        "SELECT file_hash, visual_indexed_at, visual_indexed_hash FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if doc_row is None:
        conn.close()
        raise ValueError(f"Unknown doc_id: {doc_id}")

    if (not force) and doc_row["visual_indexed_at"] and doc_row["visual_indexed_hash"] == doc_row["file_hash"]:
        logger.info("Skipping visual indexing (already indexed): doc_id=%s", doc_id)
        conn.close()
        return

    page_rows = conn.execute(
        "SELECT page_num, rendered_image_path FROM pages WHERE doc_id = ? ORDER BY page_num ASC",
        (doc_id,),
    ).fetchall()
    page_images: list[Path] = []
    for r in page_rows:
        p = r["rendered_image_path"]
        if p:
            page_images.append(Path(str(p)))

    index = ColPaliPageIndex(Path(cfg.paths.colpali_dir))
    index.build_for_document(doc_id=doc_id, page_image_paths=page_images)

    db.mark_visual_indexed(conn, doc_id=doc_id)
    db.commit(conn)
    conn.close()
    logger.info("Visual indexing complete: doc_id=%s pages=%s", doc_id, len(page_images))
