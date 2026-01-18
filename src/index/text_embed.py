from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBatch:
    ids: list[str]
    embeddings: np.ndarray  # (N, D)


class TextEmbedder:
    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'. Install it to generate embeddings."
            ) from exc

        device = None if self.device == "auto" else self.device
        self._model = SentenceTransformer(self.model_name, device=device)

    def embed(self, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
        self._load()
        assert self._model is not None
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

