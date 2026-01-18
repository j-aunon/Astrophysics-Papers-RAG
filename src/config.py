from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .lang_policy import enforce_startup_language_policy


@dataclass(frozen=True)
class LanguageConfig:
    output_language: str = "en"


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    sqlite_path: str = "data/metadata.sqlite3"
    chroma_dir: str = "data/chroma"
    pages_dir: str = "data/pages"
    figures_dir: str = "data/figures"
    colpali_dir: str = "data/colpali"
    logs_dir: str = "logs"


@dataclass(frozen=True)
class IngestionConfig:
    render_dpi: int = 200
    max_pages: int | None = None


@dataclass(frozen=True)
class ModelsConfig:
    text_embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vlm_model: str = "Qwen/Qwen3-VL"
    device: str = "auto"  # auto|cpu|cuda
    dtype: str = "auto"  # auto|float16|bfloat16|float32


@dataclass(frozen=True)
class AppConfig:
    language: LanguageConfig = LanguageConfig()
    paths: PathsConfig = PathsConfig()
    ingestion: IngestionConfig = IngestionConfig()
    models: ModelsConfig = ModelsConfig()


def _deep_get(dct: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = dct
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_config(config_path: str | None = None) -> AppConfig:
    """
    Loads configuration from YAML (default: ./config.yaml) with safe defaults.

    Environment-variable overrides are intentionally minimal to keep behavior predictable.
    """
    cfg_path = Path(config_path or "config.yaml")
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    config = AppConfig(
        language=LanguageConfig(
            output_language=str(_deep_get(raw, "language.output_language", "en")),
        ),
        paths=PathsConfig(
            data_dir=str(_deep_get(raw, "paths.data_dir", "data")),
            sqlite_path=str(_deep_get(raw, "paths.sqlite_path", "data/metadata.sqlite3")),
            chroma_dir=str(_deep_get(raw, "paths.chroma_dir", "data/chroma")),
            pages_dir=str(_deep_get(raw, "paths.pages_dir", "data/pages")),
            figures_dir=str(_deep_get(raw, "paths.figures_dir", "data/figures")),
            colpali_dir=str(_deep_get(raw, "paths.colpali_dir", "data/colpali")),
            logs_dir=str(_deep_get(raw, "paths.logs_dir", "logs")),
        ),
        ingestion=IngestionConfig(
            render_dpi=int(_deep_get(raw, "ingestion.render_dpi", 200)),
            max_pages=_deep_get(raw, "ingestion.max_pages", None),
        ),
        models=ModelsConfig(
            text_embedding_model=str(_deep_get(raw, "models.text_embedding_model", "BAAI/bge-m3")),
            llm_model=str(_deep_get(raw, "models.llm_model", "Qwen/Qwen2.5-7B-Instruct")),
            vlm_model=str(_deep_get(raw, "models.vlm_model", "Qwen/Qwen3-VL")),
            device=str(_deep_get(raw, "models.device", "auto")),
            dtype=str(_deep_get(raw, "models.dtype", "auto")),
        ),
    )

    enforce_startup_language_policy(config.language.output_language)
    Path(config.paths.data_dir).mkdir(parents=True, exist_ok=True)
    return config
