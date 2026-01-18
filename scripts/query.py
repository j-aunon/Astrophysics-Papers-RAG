from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cli import main


def _build_args() -> list[str]:
    p = argparse.ArgumentParser(description="Query the local multimodal RAG (convenience wrapper).")
    p.add_argument("--question", required=True, help="Scientific question (English).")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    p.add_argument("--json", action="store_true", help="Output JSON payload.")
    args = p.parse_args()

    argv = ["--config", args.config, "query", "--question", args.question]
    if args.json:
        argv.append("--json")
    return argv


if __name__ == "__main__":
    raise SystemExit(main(_build_args()))
