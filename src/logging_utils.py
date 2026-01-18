from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging(log_level: str, logs_dir: str | None = None) -> None:
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logs_dir:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(logs_dir) / "astro_rag.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

    # Silence overly chatty libs by default.
    for noisy in ("chromadb", "urllib3", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")

