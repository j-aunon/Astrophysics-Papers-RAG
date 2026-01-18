from __future__ import annotations

import logging
from pathlib import Path

import pytesseract
from PIL import Image

from ..lang_policy import enforce_english_output

logger = logging.getLogger(__name__)


def _tesseract_available() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def ocr_image(image_path: Path) -> str | None:
    if not _tesseract_available():
        logger.warning(
            "Tesseract is not available. Install system package 'tesseract-ocr' to enable OCR."
        )
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng", config="--oem 1 --psm 6")
        text = (text or "").strip()
        if not text:
            return ""
        return enforce_english_output(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed for %s: %s", image_path, exc)
        return None

