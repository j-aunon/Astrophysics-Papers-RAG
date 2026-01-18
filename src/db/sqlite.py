from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(sqlite_path: str) -> sqlite3.Connection:
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
          doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
          file_path TEXT NOT NULL UNIQUE,
          file_name TEXT NOT NULL,
          file_hash TEXT NOT NULL,
          num_pages INTEGER NOT NULL,
          ingested_at TEXT,
          text_indexed_at TEXT,
          text_indexed_hash TEXT,
          visual_indexed_at TEXT,
          visual_indexed_hash TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pages (
          page_id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_id INTEGER NOT NULL,
          page_num INTEGER NOT NULL,
          extracted_text TEXT,
          rendered_image_path TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(doc_id, page_num),
          FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS text_chunks (
          chunk_pk INTEGER PRIMARY KEY AUTOINCREMENT,
          chunk_uid TEXT NOT NULL UNIQUE,
          chunk_id TEXT NOT NULL,
          doc_id INTEGER NOT NULL,
          page_num INTEGER NOT NULL,
          chunk_index INTEGER NOT NULL,
          section_name TEXT,
          text TEXT NOT NULL,
          text_offset_start INTEGER,
          text_offset_end INTEGER,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(doc_id, page_num, chunk_id),
          FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS figures (
          figure_pk INTEGER PRIMARY KEY AUTOINCREMENT,
          figure_uid TEXT NOT NULL UNIQUE,
          figure_id TEXT NOT NULL,
          doc_id INTEGER NOT NULL,
          page_num INTEGER NOT NULL,
          image_path TEXT NOT NULL,
          bbox_x0 REAL,
          bbox_y0 REAL,
          bbox_x1 REAL,
          bbox_y1 REAL,
          ocr_text TEXT,
          vlm_caption TEXT,
          vlm_entities_json TEXT,
          vlm_bullets_json TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(doc_id, page_num, figure_id),
          FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON text_chunks(doc_id, page_num);
        CREATE INDEX IF NOT EXISTS idx_figures_doc_page ON figures(doc_id, page_num);
        """
    )
    _ensure_columns(
        conn,
        table="documents",
        columns={
            "ingested_at": "TEXT",
            "text_indexed_at": "TEXT",
            "text_indexed_hash": "TEXT",
            "visual_indexed_at": "TEXT",
            "visual_indexed_hash": "TEXT",
        },
    )
    conn.commit()


def _ensure_columns(
    conn: sqlite3.Connection, *, table: str, columns: dict[str, str]
) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        if row and row["name"]
    }
    for name, col_type in columns.items():
        if name in existing:
            continue
        logger.info("Adding missing column %s.%s (%s)", table, name, col_type)
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type};")


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: int
    file_path: str
    file_name: str
    file_hash: str
    num_pages: int


def upsert_document(
    conn: sqlite3.Connection, *, file_path: str, file_name: str, file_hash: str, num_pages: int
) -> DocumentRecord:
    now = utc_now_iso()
    row = conn.execute(
        "SELECT doc_id, file_hash, num_pages FROM documents WHERE file_path = ?",
        (file_path,),
    ).fetchone()

    if row is None:
        conn.execute(
            """
            INSERT INTO documents(file_path, file_name, file_hash, num_pages, created_at, updated_at)
            VALUES(?,?,?,?,?,?)
            """,
            (file_path, file_name, file_hash, num_pages, now, now),
        )
        conn.commit()
    else:
        conn.execute(
            """
            UPDATE documents
            SET file_name = ?, file_hash = ?, num_pages = ?, updated_at = ?
            WHERE file_path = ?
            """,
            (file_name, file_hash, num_pages, now, file_path),
        )
        conn.commit()

    out = conn.execute(
        """
        SELECT doc_id, file_path, file_name, file_hash, num_pages
        FROM documents
        WHERE file_path = ?
        """,
        (file_path,),
    ).fetchone()
    assert out is not None
    return DocumentRecord(
        doc_id=int(out["doc_id"]),
        file_path=str(out["file_path"]),
        file_name=str(out["file_name"]),
        file_hash=str(out["file_hash"]),
        num_pages=int(out["num_pages"]),
    )


def mark_ingested(conn: sqlite3.Connection, *, doc_id: int) -> None:
    now = utc_now_iso()
    conn.execute(
        "UPDATE documents SET ingested_at = ?, updated_at = ? WHERE doc_id = ?",
        (now, now, doc_id),
    )


def mark_text_indexed(conn: sqlite3.Connection, *, doc_id: int) -> None:
    now = utc_now_iso()
    conn.execute(
        "UPDATE documents SET text_indexed_at = ?, text_indexed_hash = file_hash, updated_at = ? WHERE doc_id = ?",
        (now, now, doc_id),
    )


def mark_visual_indexed(conn: sqlite3.Connection, *, doc_id: int) -> None:
    now = utc_now_iso()
    conn.execute(
        "UPDATE documents SET visual_indexed_at = ?, visual_indexed_hash = file_hash, updated_at = ? WHERE doc_id = ?",
        (now, now, doc_id),
    )


def get_document_by_path(conn: sqlite3.Connection, file_path: str) -> DocumentRecord | None:
    row = conn.execute(
        """
        SELECT doc_id, file_path, file_name, file_hash, num_pages
        FROM documents
        WHERE file_path = ?
        """,
        (file_path,),
    ).fetchone()
    if row is None:
        return None
    return DocumentRecord(
        doc_id=int(row["doc_id"]),
        file_path=str(row["file_path"]),
        file_name=str(row["file_name"]),
        file_hash=str(row["file_hash"]),
        num_pages=int(row["num_pages"]),
    )


def upsert_page(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    page_num: int,
    extracted_text: str | None,
    rendered_image_path: str | None,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO pages(doc_id, page_num, extracted_text, rendered_image_path, created_at, updated_at)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(doc_id, page_num) DO UPDATE SET
          extracted_text=excluded.extracted_text,
          rendered_image_path=excluded.rendered_image_path,
          updated_at=excluded.updated_at
        """,
        (doc_id, page_num, extracted_text, rendered_image_path, now, now),
    )


def replace_chunks(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    page_num: int,
    chunks: Iterable[dict],
) -> None:
    now = utc_now_iso()
    conn.execute("DELETE FROM text_chunks WHERE doc_id = ? AND page_num = ?", (doc_id, page_num))
    conn.executemany(
        """
        INSERT INTO text_chunks(
          chunk_uid, chunk_id, doc_id, page_num, chunk_index, section_name, text,
          text_offset_start, text_offset_end, created_at, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                c["chunk_uid"],
                c["chunk_id"],
                doc_id,
                page_num,
                int(c["chunk_index"]),
                c.get("section_name"),
                c["text"],
                c.get("text_offset_start"),
                c.get("text_offset_end"),
                now,
                now,
            )
            for c in chunks
        ],
    )


def upsert_figure(
    conn: sqlite3.Connection,
    *,
    figure_uid: str,
    figure_id: str,
    doc_id: int,
    page_num: int,
    image_path: str,
    bbox: tuple[float, float, float, float] | None,
    ocr_text: str | None,
    vlm_caption: str | None,
    vlm_entities_json: str | None,
    vlm_bullets_json: str | None,
) -> None:
    now = utc_now_iso()
    bbox_x0 = bbox[0] if bbox else None
    bbox_y0 = bbox[1] if bbox else None
    bbox_x1 = bbox[2] if bbox else None
    bbox_y1 = bbox[3] if bbox else None

    conn.execute(
        """
        INSERT INTO figures(
          figure_uid, figure_id, doc_id, page_num, image_path,
          bbox_x0, bbox_y0, bbox_x1, bbox_y1,
          ocr_text, vlm_caption, vlm_entities_json, vlm_bullets_json,
          created_at, updated_at
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(figure_uid) DO UPDATE SET
          image_path=excluded.image_path,
          bbox_x0=excluded.bbox_x0,
          bbox_y0=excluded.bbox_y0,
          bbox_x1=excluded.bbox_x1,
          bbox_y1=excluded.bbox_y1,
          ocr_text=excluded.ocr_text,
          vlm_caption=excluded.vlm_caption,
          vlm_entities_json=excluded.vlm_entities_json,
          vlm_bullets_json=excluded.vlm_bullets_json,
          updated_at=excluded.updated_at
        """,
        (
            figure_uid,
            figure_id,
            doc_id,
            page_num,
            image_path,
            bbox_x0,
            bbox_y0,
            bbox_x1,
            bbox_y1,
            ocr_text,
            vlm_caption,
            vlm_entities_json,
            vlm_bullets_json,
            now,
            now,
        ),
    )


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()
