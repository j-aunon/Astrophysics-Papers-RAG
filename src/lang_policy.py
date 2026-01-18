from __future__ import annotations

import re


class LanguagePolicyError(RuntimeError):
    pass


_NON_ENGLISH_SCRIPT_RE = re.compile(
    r"["  # common non-English scripts; allow Greek for scientific notation
    r"\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\u3040-\u30ff"  # Hiragana/Katakana
    r"\uac00-\ud7af"  # Hangul
    r"\u0400-\u04ff"  # Cyrillic
    r"\u0600-\u06ff"  # Arabic
    r"\u0900-\u097f"  # Devanagari
    r"\u0e00-\u0e7f"  # Thai
    r"\u0590-\u05ff"  # Hebrew
    r"]"
)


def enforce_startup_language_policy(output_language: str) -> None:
    if (output_language or "").strip().lower() != "en":
        raise LanguagePolicyError("Language policy violation: output must be English.")


def enforce_english_output(text: str) -> str:
    """
    Best-effort English-only gate.

    Scientific content may include symbols (e.g., Greek letters) and math.
    This gate only blocks common non-English scripts (CJK).
    """
    if text is None:
        return ""
    if _NON_ENGLISH_SCRIPT_RE.search(text):
        raise LanguagePolicyError("Language policy violation: output must be English.")
    return text
