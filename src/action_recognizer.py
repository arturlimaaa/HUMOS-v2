"""
Action Recognition via Vision-Language Model.

The core engine: sends egocentric video frames to GPT-4o Vision
and asks "what is the human doing?" -- not "what objects exist?"

The human's behavior IS the ground truth.
"""
from __future__ import annotations

import os
import json
import base64
import re
import time
from io import BytesIO
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import config


@dataclass
class FrameAction:
    """Per-frame action recognition result."""
    frame_idx: int
    timestamp_sec: float
    verb: str
    noun: str
    phase: str           # approaching | contact | manipulating | releasing | idle
    confidence: float
    description: str


def _image_to_base64(image: np.ndarray, quality: int = 70) -> str:
    """Convert BGR numpy image to base64 JPEG."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """Extract JSON from VLM response."""
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


class ActionRecognizer:
    """Recognizes human actions in egocentric video frames using a VLM."""

    def __init__(self, backend: str = None, few_shot_bank=None):
        self.backend = backend or config.VLM_BACKEND
        self._client = None
        self._few_shot_bank = few_shot_bank
        self._call_count = 0
        self._total_tokens = 0

        if self.backend == "openai":
            self._init_openai()

    def _init_openai(self):
        try:
            from openai import OpenAI
            api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("  WARNING: No OPENAI_API_KEY. Falling back to heuristic.")
                self.backend = "heuristic"
                return
            self._client = OpenAI(api_key=api_key)
            print(f"  OpenAI client ready (model: {config.OPENAI_MODEL})")
        except ImportError:
            print("  WARNING: openai not installed. Using heuristic.")
            self.backend = "heuristic"

    def recognize_frame(self, image: np.ndarray, frame_idx: int, timestamp_sec: float) -> FrameAction:
        """Recognize the action in a single frame."""
        if self.backend == "openai":
            result = self._recognize_openai(image)
        else:
            result = self._recognize_heuristic()

        return FrameAction(
            frame_idx=frame_idx,
            timestamp_sec=round(timestamp_sec, 3),
            verb=result.get("verb", "idle"),
            noun=result.get("noun", "none"),
            phase=result.get("phase", "idle"),
            confidence=result.get("confidence", 0.0),
            description=result.get("description", ""),
        )

    def _recognize_openai(self, image: np.ndarray) -> dict:
        """Call GPT-4o Vision for action recognition."""
        b64 = _image_to_base64(image)

        messages = []

        # Inject few-shot examples if available
        if self._few_shot_bank:
            examples = self._few_shot_bank.get_examples("action", k=2)
            for ex in examples:
                ex_img = cv2.imread(ex["crop_path"])
                if ex_img is not None:
                    ex_b64 = _image_to_base64(ex_img)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What action is happening?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_b64}", "detail": "low"}},
                        ],
                    })
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "verb": ex.get("state", "idle"),  # 'state' field stores verb in action mode
                            "noun": ex.get("context", {}).get("noun", "unknown"),
                            "phase": "manipulating",
                            "confidence": 0.95,
                            "description": f"Verified: {ex.get('state', 'idle')} {ex.get('context', {}).get('noun', '')}"
                        }),
                    })

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": config.ACTION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                }},
            ],
        })

        try:
            response = self._client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                max_tokens=150,
                temperature=0.1,
            )
            self._call_count += 1
            if response.usage:
                self._total_tokens += response.usage.total_tokens
            text = response.choices[0].message.content
            return _parse_json_response(text)
        except Exception as e:
            print(f"  OpenAI error: {e}")
            return {"verb": "idle", "noun": "none", "phase": "idle", "confidence": 0.0}

    def _recognize_heuristic(self) -> dict:
        """Heuristic fallback -- returns idle. Only useful for testing flow."""
        return {
            "verb": "idle",
            "noun": "none",
            "phase": "idle",
            "confidence": 0.5,
            "description": "Heuristic mode: no VLM available",
        }

    @property
    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "api_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }
