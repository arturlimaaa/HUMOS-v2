"""
Action Segmenter: groups per-frame VLM outputs into temporal action segments.

Frame-level outputs like:
  frame 0: idle/none
  frame 5: open/fridge
  frame 10: open/fridge
  frame 15: open/fridge
  frame 20: take/milk
  frame 25: take/milk

Become segments:
  [0.17-0.83s] open fridge (conf=0.85)
  [0.83-1.33s] take milk (conf=0.78)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.action_recognizer import FrameAction


@dataclass
class ActionSegment:
    """A temporal action segment -- the core output unit."""
    segment_id: int
    verb: str
    noun: str
    start_frame: int
    stop_frame: int
    start_sec: float
    stop_sec: float
    confidence: float
    description: str
    frame_actions: list[FrameAction] = field(default_factory=list, repr=False)

    @property
    def duration_sec(self) -> float:
        return self.stop_sec - self.start_sec

    @property
    def action_label(self) -> str:
        return f"{self.verb} {self.noun}"

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "verb": self.verb,
            "noun": self.noun,
            "start_frame": self.start_frame,
            "stop_frame": self.stop_frame,
            "start_sec": round(self.start_sec, 3),
            "stop_sec": round(self.stop_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "confidence": round(self.confidence, 3),
            "description": self.description,
        }


class ActionSegmenter:
    """Groups consecutive same-action frames into temporal segments."""

    def __init__(
        self,
        min_frames: int = None,
        merge_gap: int = None,
        min_confidence: float = None,
    ):
        self.min_frames = min_frames or config.SEGMENT_MIN_FRAMES
        self.merge_gap = merge_gap or config.SEGMENT_MERGE_GAP
        self.min_confidence = min_confidence or config.SEGMENT_MIN_CONFIDENCE

    def segment(self, frame_actions: list[FrameAction]) -> list[ActionSegment]:
        """
        Convert a list of per-frame actions into merged action segments.

        1. Filter out low-confidence and idle frames
        2. Group consecutive frames with same (verb, noun)
        3. Merge nearby segments with same action
        4. Filter out segments shorter than min_frames
        """
        if not frame_actions:
            return []

        # Step 1: Build raw groups
        raw_groups = self._group_consecutive(frame_actions)

        # Step 2: Merge nearby same-action groups
        merged = self._merge_nearby(raw_groups)

        # Step 3: Filter by minimum length
        filtered = [g for g in merged if len(g) >= self.min_frames]

        # Step 4: Convert to ActionSegments
        segments = []
        for idx, group in enumerate(filtered):
            avg_conf = sum(f.confidence for f in group) / len(group)
            # Pick the most common description
            descriptions = [f.description for f in group if f.description]
            desc = descriptions[0] if descriptions else f"{group[0].verb} {group[0].noun}"

            segments.append(ActionSegment(
                segment_id=idx,
                verb=group[0].verb,
                noun=group[0].noun,
                start_frame=group[0].frame_idx,
                stop_frame=group[-1].frame_idx,
                start_sec=group[0].timestamp_sec,
                stop_sec=group[-1].timestamp_sec,
                confidence=round(avg_conf, 3),
                description=desc,
                frame_actions=group,
            ))

        return segments

    def _group_consecutive(self, frame_actions: list[FrameAction]) -> list[list[FrameAction]]:
        """Group consecutive frames with the same (verb, noun), skipping idle/low-conf."""
        groups: list[list[FrameAction]] = []
        current_group: list[FrameAction] = []

        for fa in frame_actions:
            # Skip idle and low-confidence frames
            if fa.verb == "idle" or fa.confidence < self.min_confidence:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                continue

            if current_group:
                prev = current_group[-1]
                if fa.verb == prev.verb and fa.noun == prev.noun:
                    current_group.append(fa)
                else:
                    groups.append(current_group)
                    current_group = [fa]
            else:
                current_group = [fa]

        if current_group:
            groups.append(current_group)

        return groups

    def _merge_nearby(self, groups: list[list[FrameAction]]) -> list[list[FrameAction]]:
        """Merge groups that are close together and have the same action."""
        if len(groups) <= 1:
            return groups

        merged = [groups[0]]
        for group in groups[1:]:
            prev = merged[-1]
            prev_key = (prev[0].verb, prev[0].noun)
            curr_key = (group[0].verb, group[0].noun)

            # Check if same action and gap is small
            gap_frames = group[0].frame_idx - prev[-1].frame_idx
            sample_gap = gap_frames / max(config.FRAME_SAMPLE_RATE, 1)

            if prev_key == curr_key and sample_gap <= self.merge_gap:
                merged[-1] = prev + group
            else:
                merged.append(group)

        return merged
