from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cli import main


def _build_args() -> list[str]:
    p = argparse.ArgumentParser(description="Ingest and index PDFs (convenience wrapper).")
    p.add_argument("--pdf-dir", required=True, help="Directory containing PDFs.")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    p.add_argument("--force", action="store_true", help="Re-ingest and re-index.")
    args = p.parse_args()

    argv = ["--config", args.config, "index", "--pdf", args.pdf_dir]
    if args.force:
        argv.append("--force")
    return argv


if __name__ == "__main__":
    raise SystemExit(main(_build_args()))
