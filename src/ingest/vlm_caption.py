from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..lang_policy import enforce_english_output

logger = logging.getLogger(__name__)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class FigureVlmOutput:
    caption: str
    entities: list[str]
    bullets: list[str]


class Qwen3VLCaptioner:
    """
    Figure captioning via a local Qwen3-VL model.

    This implementation is intentionally defensive because model class names and processors can differ
    between Qwen-VL variants. If the model cannot be loaded, ingestion will continue with a stub.
    """

    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Missing dependency for VLM. Install 'transformers' and 'torch'."
            ) from exc

        torch_dtype = None
        if self.dtype in ("float16", "bfloat16", "float32"):
            torch_dtype = getattr(torch, self.dtype)

        model_kwargs: dict[str, Any] = {"device_map": "auto"} if self.device == "auto" else {}
        if self.device in ("cpu", "cuda"):
            model_kwargs["device_map"] = {"": self.device}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, trust_remote_code=True, **model_kwargs
            )
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load Qwen3-VL. Ensure the model exists locally in your Hugging Face cache "
                "and that your installed 'transformers' version supports it. "
                f"Model name: {self.model_name}"
            ) from exc

    def caption(self, image_path: Path) -> FigureVlmOutput | None:
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM captioning disabled: %s", exc)
            return None

        assert self._model is not None and self._processor is not None

        prompt = (
            "You are an expert astrophysics assistant. Analyze the figure and output ONLY valid JSON.\n"
            "JSON schema:\n"
            '{\n'
            '  "caption": "Concise technical caption (1-2 sentences).",\n'
            '  "entities": ["List of detected scientific entities, symbols, variables, instruments."],\n'
            '  "bullets": ["3-5 bullet points describing what the figure shows scientifically."]\n'
            '}\n'
            "Rules:\n"
            "- English only.\n"
            "- Be precise and technical.\n"
            "- Do not include markdown.\n"
        )

        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=img, text=prompt, return_tensors="pt")
            try:
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:  # noqa: BLE001
                pass
            output_ids = self._model.generate(**inputs, max_new_tokens=512)
            out_text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM inference failed for %s: %s", image_path, exc)
            return None

        parsed = _parse_json_object(out_text)
        if parsed is None:
            logger.warning("VLM returned non-JSON output; skipping caption for %s", image_path)
            return None

        caption = enforce_english_output(str(parsed.get("caption", "")).strip())
        entities = [enforce_english_output(str(x).strip()) for x in (parsed.get("entities") or [])]
        bullets = [enforce_english_output(str(x).strip()) for x in (parsed.get("bullets") or [])]

        return FigureVlmOutput(caption=caption, entities=entities, bullets=bullets)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:  # noqa: BLE001
        return None
