from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .db import sqlite as db
from .ingest.pdf_loader import iter_pdf_paths
from .ingest.pipeline import ingest_pdf
from .index.pipeline import index_document_text, index_document_visual
from .lang_policy import LanguagePolicyError, enforce_startup_language_policy
from .logging_utils import configure_logging
from .rag.answer_llm import QwenAnswerGenerator
from .rag.build_context import build_context
from .retriever.hybrid import retrieve_hybrid

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astro-rag",
        description="Local multimodal RAG over astrophysics PDFs (SQLite + ChromaDB + ColPali + Qwen).",
    )
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file.")

    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest PDFs into SQLite and extract artifacts.")
    ingest.add_argument("--pdf", required=True, help="Path to a PDF file or a directory of PDFs.")
    ingest.add_argument("--force", action="store_true", help="Re-ingest even if unchanged.")
    ingest.add_argument(
        "--disable-vlm",
        action="store_true",
        help="Disable Qwen3-VL figure captioning (still extracts images + OCR).",
    )

    index = sub.add_parser("index", help="Index ingested PDFs (text into Chroma; pages into ColPali).")
    index.add_argument("--pdf", required=True, help="Path to a PDF file or a directory of PDFs.")
    index.add_argument("--force", action="store_true", help="Re-index even if already indexed.")
    index.add_argument("--skip-text", action="store_true", help="Skip text indexing.")
    index.add_argument("--skip-visual", action="store_true", help="Skip visual indexing (ColPali).")

    query = sub.add_parser("query", help="Ask a question and get a cited answer.")
    query.add_argument("--question", required=True, help="Scientific question (English).")
    query.add_argument("--json", action="store_true", help="Output raw JSON payload to stdout.")

    health = sub.add_parser("health", help="Validate local runtime dependencies.")
    health.add_argument("--verbose", action="store_true", help="Print additional diagnostics.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    configure_logging("INFO", cfg.paths.logs_dir)

    try:
        enforce_startup_language_policy(cfg.language.output_language)
    except LanguagePolicyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.cmd == "ingest":
        return _cmd_ingest(cfg, args)
    if args.cmd == "index":
        return _cmd_index(cfg, args)
    if args.cmd == "query":
        return _cmd_query(cfg, args)
    if args.cmd == "health":
        return _cmd_health(cfg, args)

    print("Unknown command.", file=sys.stderr)
    return 2


def _cmd_ingest(cfg, args) -> int:
    pdfs = list(iter_pdf_paths(args.pdf))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 2

    for pdf in pdfs:
        ingest_pdf(cfg, pdf_path=pdf, force=bool(args.force), enable_vlm=not bool(args.disable_vlm))

    print(f"Ingested PDFs: {len(pdfs)}")
    return 0


def _cmd_index(cfg, args) -> int:
    pdfs = list(iter_pdf_paths(args.pdf))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 2

    for pdf in pdfs:
        doc = ingest_pdf(cfg, pdf_path=pdf, force=False, enable_vlm=True)
        if not args.skip_text:
            index_document_text(cfg, doc_id=doc.doc_id, force=bool(args.force))
        if not args.skip_visual:
            index_document_visual(cfg, doc_id=doc.doc_id, force=bool(args.force))

    print(f"Indexed PDFs: {len(pdfs)}")
    return 0


def _cmd_query(cfg, args) -> int:
    question = str(args.question).strip()
    if not question:
        print("Question must be non-empty.", file=sys.stderr)
        return 2

    results = retrieve_hybrid(cfg, question=question)
    ctx = build_context(cfg, results=results)

    generator = QwenAnswerGenerator(cfg.models.llm_model, device=cfg.models.device, dtype=cfg.models.dtype)
    answer = generator.generate(question=question, context_text=ctx.context_text)

    payload = {
        "question": question,
        "answer": answer.answer_text,
        "llm_model": answer.model_name,
        "text_evidence": ctx.text_items,
        "figure_evidence": ctx.figure_items,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(answer.answer_text)
    return 0


def _cmd_health(cfg, args) -> int:
    problems: list[str] = []

    # Paths
    for p in (
        Path(cfg.paths.data_dir),
        Path(cfg.paths.chroma_dir),
        Path(cfg.paths.pages_dir),
        Path(cfg.paths.figures_dir),
        Path(cfg.paths.colpali_dir),
    ):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"Failed to create directory {p}: {exc}")

    # SQLite schema
    try:
        conn = db.connect(cfg.paths.sqlite_path)
        db.ensure_schema(conn)
        conn.close()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"SQLite error: {exc}")

    # Tesseract
    try:
        import pytesseract

        _ = pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001
        problems.append(
            "Tesseract is not available (required for OCR). Install system package 'tesseract-ocr'. "
            f"Details: {exc}"
        )

    if problems:
        print("Health check failed:", file=sys.stderr)
        for pr in problems:
            print(f"- {pr}", file=sys.stderr)
        return 2

    print("Health check passed.")
    if args.verbose:
        print(f"SQLite: {cfg.paths.sqlite_path}")
        print(f"Chroma: {cfg.paths.chroma_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
