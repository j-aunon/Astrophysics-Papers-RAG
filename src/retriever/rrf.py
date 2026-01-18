from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedResult:
    key: str
    score: float
    source: str  # "text" | "visual"
    payload: dict


def rrf_merge(
    *,
    text_ranked: list[tuple[str, dict]],
    visual_ranked: list[tuple[str, dict]],
    rrf_k: int,
    w_text: float,
    w_visual: float,
    top_k: int,
) -> list[FusedResult]:
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    payloads: dict[str, dict] = {}

    def add(rank_list: list[tuple[str, dict]], weight: float, source: str) -> None:
        for i, (key, payload) in enumerate(rank_list, start=1):
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + i))
            sources[key] = sources.get(key, source)
            payloads[key] = payloads.get(key, payload)

    add(text_ranked, w_text, "text")
    add(visual_ranked, w_visual, "visual")

    fused = [
        FusedResult(key=k, score=v, source=sources.get(k, "text"), payload=payloads.get(k, {}))
        for k, v in scores.items()
    ]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]

