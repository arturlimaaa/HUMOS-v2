"""
Few-Shot Bank: disk-backed store for human-corrected examples.

Each correction (image_crop + label + state) is saved to disk.
When the VLM classifies future objects, the bank provides same-class
examples to inject into the prompt, improving accuracy over time.

This is the mechanism that makes "ground truth compound over time":
every human correction makes the system smarter for the next video.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

import config


class FewShotBank:
    """Disk-backed store for corrected examples, queryable by class name."""

    def __init__(self, bank_dir: str | Path | None = None):
        self.bank_dir = Path(bank_dir or config.DATA_DIR / "few_shot_bank")
        self.crops_dir = self.bank_dir / "crops"
        self.index_path = self.bank_dir / "index.json"

        # Ensure dirs exist
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index: dict = self._load_index()

    def _load_index(self) -> dict:
        """Load the index from disk, or create empty one."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"entries": [], "stats": {}}

    def _save_index(self):
        """Persist index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def add_correction(
        self,
        class_name: str,
        state: str,
        image_crop: np.ndarray,
        context: dict | None = None,
    ) -> str:
        """
        Store a corrected example.

        Args:
            class_name: Object class (e.g. "refrigerator")
            state: Corrected state (e.g. "open")
            image_crop: BGR numpy image of the cropped object
            context: Optional metadata (video_path, frame_idx, track_id, etc.)

        Returns:
            The ID of the stored example.
        """
        entry_id = f"{class_name}_{state}_{int(time.time()*1000)}"
        crop_filename = f"{entry_id}.jpg"
        crop_path = self.crops_dir / crop_filename

        # Save crop
        cv2.imwrite(str(crop_path), image_crop)

        # Add to index
        entry = {
            "id": entry_id,
            "class_name": class_name,
            "state": state,
            "crop_path": crop_filename,
            "timestamp": time.time(),
            "context": context or {},
        }
        self._index["entries"].append(entry)

        # Update stats
        key = class_name
        self._index["stats"][key] = self._index["stats"].get(key, 0) + 1

        self._save_index()
        return entry_id

    def get_examples(self, class_name: str, k: int = 3) -> list[dict]:
        """
        Retrieve up to k corrected examples for a given class.

        Returns list of dicts with:
            - class_name, state, crop_path (absolute), context
        Prioritizes diversity: tries to return different states.
        """
        # Filter entries for this class
        candidates = [
            e for e in self._index["entries"]
            if e["class_name"] == class_name
        ]

        if not candidates:
            return []

        # Prioritize diversity of states
        by_state: dict[str, list] = defaultdict(list)
        for e in candidates:
            by_state[e["state"]].append(e)

        selected = []
        # Round-robin across states
        states = list(by_state.keys())
        idx = 0
        while len(selected) < k and any(by_state.values()):
            state = states[idx % len(states)]
            if by_state[state]:
                entry = by_state[state].pop(-1)  # Most recent first
                selected.append({
                    "class_name": entry["class_name"],
                    "state": entry["state"],
                    "crop_path": str(self.crops_dir / entry["crop_path"]),
                    "context": entry.get("context", {}),
                })
            idx += 1
            # Break if all exhausted
            if all(len(v) == 0 for v in by_state.values()):
                break

        return selected[:k]

    def get_examples_as_images(self, class_name: str, k: int = 3) -> list[tuple[np.ndarray, str]]:
        """
        Retrieve examples as (image, state) tuples for VLM prompt injection.
        """
        examples = self.get_examples(class_name, k)
        results = []
        for ex in examples:
            img = cv2.imread(ex["crop_path"])
            if img is not None:
                results.append((img, ex["state"]))
        return results

    def stats(self) -> dict:
        """Return correction counts per class."""
        return dict(self._index.get("stats", {}))

    def total_corrections(self) -> int:
        return len(self._index.get("entries", []))

    def __repr__(self):
        total = self.total_corrections()
        classes = len(self.stats())
        return f"FewShotBank({total} examples, {classes} classes)"
