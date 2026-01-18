from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualPageResult:
    doc_id: int
    page_num: int
    score: float


class ColPaliPageIndex:
    """
    ColPali page-level candidate generator.

    This is intentionally implemented as an optional component because ColPali packaging and model
    availability can vary by environment. When unavailable, visual retrieval returns no results.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_for_document(self, *, doc_id: int, page_image_paths: list[Path]) -> None:
        """
        Build/update the ColPali index for a single document.

        Stub behavior: logs an instruction and does nothing.
        """
        logger.warning(
            "ColPali indexing is not configured in this environment. "
            "Install ColPali and implement the embedding backend to enable visual retrieval."
        )
        _ = (doc_id, page_image_paths)

    def query(self, *, query: str, top_k: int) -> list[VisualPageResult]:
        logger.warning(
            "ColPali retrieval is disabled. Returning no visual candidates. "
            "Install ColPali and configure the visual index to enable this feature."
        )
        _ = (query, top_k)
        return []

