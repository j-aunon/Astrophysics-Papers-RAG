from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..index.chroma_client import ChromaTextResult, get_client, get_text_collection
from ..index.colpali_index import ColPaliPageIndex, VisualPageResult
from ..index.text_embed import TextEmbedder
from .rrf import FusedResult, rrf_merge

logger = logging.getLogger(__name__)

TEXT_TOP_K = 12
VISUAL_TOP_K = 6
RRF_K = 60
W_TEXT = 1.0
W_VISUAL = 0.8


@dataclass(frozen=True)
class HybridResults:
    text: list[ChromaTextResult]
    visual_pages: list[VisualPageResult]
    fused: list[FusedResult]


def retrieve_hybrid(cfg: AppConfig, *, question: str) -> HybridResults:
    embedder = TextEmbedder(cfg.models.text_embedding_model, device=cfg.models.device)
    query_vec = embedder.embed([question], batch_size=1)[0].astype(np.float32)

    chroma = get_client(cfg.paths.chroma_dir)
    collection = get_text_collection(chroma, name="text_chunks")

    q = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=TEXT_TOP_K,
        include=["documents", "metadatas", "distances", "ids"],
    )

    text_results: list[ChromaTextResult] = []
    ids = (q.get("ids") or [[]])[0]
    docs = (q.get("documents") or [[]])[0]
    metas = (q.get("metadatas") or [[]])[0]
    dists = (q.get("distances") or [[]])[0]
    for chunk_uid, doc, meta, dist in zip(ids, docs, metas, dists):
        score = float(1.0 - float(dist)) if dist is not None else 0.0
        meta_dict = dict(meta or {})
        chunk_id = str(meta_dict.get("chunk_id") or "")
        text_results.append(
            ChromaTextResult(
                chunk_uid=str(chunk_uid),
                chunk_id=chunk_id,
                score=score,
                metadata=meta_dict,
                document=str(doc),
            )
        )

    visual_index = ColPaliPageIndex(Path(cfg.paths.colpali_dir))
    try:
        visual_pages = visual_index.query(query=question, top_k=VISUAL_TOP_K)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Visual retrieval failed; continuing with text-only retrieval: %s", exc)
        visual_pages = []

    text_ranked = [
        (
            r.chunk_uid,
            {
                "chunk_uid": r.chunk_uid,
                "chunk_id": r.chunk_id,
                "doc_id": r.metadata.get("doc_id"),
                "page_num": r.metadata.get("page_num"),
            },
        )
        for r in text_results
    ]
    visual_ranked = [
        (f"page:{r.doc_id}:{r.page_num}", {"doc_id": r.doc_id, "page_num": r.page_num}) for r in visual_pages
    ]

    fused = rrf_merge(
        text_ranked=text_ranked,
        visual_ranked=visual_ranked,
        rrf_k=RRF_K,
        w_text=W_TEXT,
        w_visual=W_VISUAL,
        top_k=max(TEXT_TOP_K, VISUAL_TOP_K),
    )

    return HybridResults(text=text_results, visual_pages=visual_pages, fused=fused)
