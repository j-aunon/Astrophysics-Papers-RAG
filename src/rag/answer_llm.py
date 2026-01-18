from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..lang_policy import enforce_english_output
from .prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    answer_text: str
    model_name: str


class QwenAnswerGenerator:
    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Missing dependency for LLM. Install 'transformers' and 'torch'."
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
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, **model_kwargs
            )
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load Qwen2.5 LLM. Ensure the model exists locally in your Hugging Face cache. "
                f"Model name: {self.model_name}"
            ) from exc

    def generate(self, *, question: str, context_text: str, max_new_tokens: int = 768) -> AnswerResult:
        self._load()
        assert self._model is not None and self._tokenizer is not None

        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            text_evidence=context_text.split("FIGURE EVIDENCE:")[0].replace("TEXT EVIDENCE:", "").strip(),
            figure_evidence=context_text.split("FIGURE EVIDENCE:")[-1].strip(),
        )

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer(prompt, return_tensors="pt")
            else:
                # Fallback: naive concatenation (still English-only prompts).
                prompt = f"{ANSWER_SYSTEM_PROMPT}\n\n{user_prompt}\n"
                inputs = self._tokenizer(prompt, return_tensors="pt")

            try:
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:  # noqa: BLE001
                # device_map/sharded models may not expose a single device; keep CPU tensors.
                pass

            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        # Best-effort strip: if chat template echoed the prompt, keep only the tail.
        # We keep it conservative to avoid dropping content in unknown tokenizer formats.
        answer = text
        answer = enforce_english_output(answer.strip())
        return AnswerResult(answer_text=answer, model_name=self.model_name)
