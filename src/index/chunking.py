from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_uid: str
    chunk_id: str
    chunk_index: int
    section_name: str | None
    text: str
    text_offset_start: int
    text_offset_end: int


_SECTION_RE = re.compile(
    r"^(?:\d+(\.\d+)*\s+)?(abstract|introduction|methods?|results?|discussion|conclusion|references)\b",
    re.IGNORECASE,
)


def _iter_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(True):
        clean = line.strip()
        if clean:
            out.append((offset, clean))
        offset += len(line)
    return out


def infer_section_spans(text: str) -> list[tuple[int, str]]:
    """
    Returns a list of (start_offset, section_name) spans inferred from headings.
    """
    spans: list[tuple[int, str]] = []
    for offset, line in _iter_lines_with_offsets(text):
        m = _SECTION_RE.match(line)
        if m:
            spans.append((offset, m.group(2).strip().title()))
    if not spans:
        return []
    # Ensure sorted and unique offsets.
    spans = sorted({o: s for o, s in spans}.items())
    return [(int(o), str(s)) for o, s in spans]


def chunk_text_section_aware(
    *,
    doc_id: int,
    page_num: int,
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    """
    Section-aware chunking within a single page.

    Notes:
    - PDFs are page-scoped; we infer headings inside the page only.
    - Offsets are character offsets into the page's extracted text.
    """
    text = (text or "").strip()
    if not text:
        return []

    spans = infer_section_spans(text)
    if not spans:
        spans = [(0, None)]

    # Add an end sentinel for segmentation.
    spans_with_end: list[tuple[int, str | None, int]] = []
    for idx, (start, name) in enumerate(spans):
        end = spans[idx + 1][0] if idx + 1 < len(spans) else len(text)
        spans_with_end.append((start, name, end))

    chunks: list[TextChunk] = []
    chunk_index = 0
    for seg_start, section_name, seg_end in spans_with_end:
        seg_text = text[seg_start:seg_end].strip()
        if not seg_text:
            continue

        cursor = seg_start
        while cursor < seg_end:
            window_end = min(cursor + max_chars, seg_end)
            window = text[cursor:window_end].strip()
            if not window:
                break

            chunk_id = f"c{chunk_index}"
            chunk_uid = f"{doc_id}:{page_num}:{chunk_id}"
            chunks.append(
                TextChunk(
                    chunk_uid=chunk_uid,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    section_name=section_name,
                    text=window,
                    text_offset_start=cursor,
                    text_offset_end=window_end,
                )
            )
            chunk_index += 1

            if window_end >= seg_end:
                break
            cursor = max(seg_start, window_end - overlap_chars)

    return chunks
