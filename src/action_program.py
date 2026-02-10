"""
Action Program Builder: converts action segments into structured programs
with preconditions and effects, suitable for PDDL generation.

This is the bridge between "the human opened the fridge" and
"precondition: is_closed(fridge), effect: is_open(fridge)" that
a robot task planner can consume.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import config
from src.action_segmenter import ActionSegment


@dataclass
class ActionStep:
    """A single step in an action program with pre/postconditions."""
    step_id: int
    verb: str
    noun: str
    start_sec: float
    stop_sec: float
    confidence: float
    description: str
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "verb": self.verb,
            "noun": self.noun,
            "start_sec": round(self.start_sec, 3),
            "stop_sec": round(self.stop_sec, 3),
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "preconditions": self.preconditions,
            "effects": self.effects,
        }


# ── Default verb->precondition/effect mapping ─────────────────────────
DEFAULT_MAPPING = {
    "open": {
        "preconditions": ["is_closed({noun})"],
        "effects": ["is_open({noun})", "NOT is_closed({noun})"],
    },
    "close": {
        "preconditions": ["is_open({noun})"],
        "effects": ["is_closed({noun})", "NOT is_open({noun})"],
    },
    "take": {
        "preconditions": ["hand_free(human)"],
        "effects": ["holding(human, {noun})", "NOT hand_free(human)"],
    },
    "put": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["on({noun}, surface)", "hand_free(human)", "NOT holding(human, {noun})"],
    },
    "turn-on": {
        "preconditions": ["is_off({noun})"],
        "effects": ["is_on({noun})", "NOT is_off({noun})"],
    },
    "turn-off": {
        "preconditions": ["is_on({noun})"],
        "effects": ["is_off({noun})", "NOT is_on({noun})"],
    },
    "wash": {
        "preconditions": ["at(human, sink)", "holding(human, {noun})"],
        "effects": ["is_clean({noun})"],
    },
    "cut": {
        "preconditions": ["holding(human, knife)", "on({noun}, cutting_board)"],
        "effects": ["is_cut({noun})"],
    },
    "pour": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["poured({noun})"],
    },
    "mix": {
        "preconditions": ["holding(human, utensil)"],
        "effects": ["is_mixed({noun})"],
    },
    "insert": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["inside({noun}, container)", "NOT holding(human, {noun})", "hand_free(human)"],
    },
    "remove": {
        "preconditions": ["inside({noun}, container)", "hand_free(human)"],
        "effects": ["holding(human, {noun})", "NOT inside({noun}, container)", "NOT hand_free(human)"],
    },
    "move": {
        "preconditions": ["reachable(human, {noun})"],
        "effects": ["moved({noun})"],
    },
    "throw": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["hand_free(human)", "NOT holding(human, {noun})", "disposed({noun})"],
    },
    "peel": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["is_peeled({noun})"],
    },
    "squeeze": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["squeezed({noun})"],
    },
    "shake": {
        "preconditions": ["holding(human, {noun})"],
        "effects": ["shaken({noun})"],
    },
    "adjust": {
        "preconditions": ["reachable(human, {noun})"],
        "effects": ["adjusted({noun})"],
    },
    "dry": {
        "preconditions": ["is_wet({noun})"],
        "effects": ["is_dry({noun})", "NOT is_wet({noun})"],
    },
    "stir": {
        "preconditions": ["holding(human, utensil)"],
        "effects": ["is_stirred({noun})"],
    },
    "scoop": {
        "preconditions": ["holding(human, utensil)", "hand_free_other(human)"],
        "effects": ["scooped({noun})"],
    },
}


def _load_mapping() -> dict:
    """Load verb->precondition/effect mapping from JSON or use defaults."""
    mapping_path = config.SCHEMA_DIR / "epic_kitchens_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            return json.load(f)
    return DEFAULT_MAPPING


def _sanitize(name: str) -> str:
    """Make a PDDL-safe identifier."""
    return name.lower().replace(" ", "_").replace("-", "_").replace(":", "_")


def build_action_program(segments: list[ActionSegment]) -> list[ActionStep]:
    """
    Convert action segments into an action program with preconditions and effects.
    """
    mapping = _load_mapping()
    steps = []

    for seg in segments:
        verb = seg.verb.lower()
        noun = _sanitize(seg.noun)

        verb_map = mapping.get(verb, {})
        preconditions = [p.replace("{noun}", noun) for p in verb_map.get("preconditions", [])]
        effects = [e.replace("{noun}", noun) for e in verb_map.get("effects", [])]

        steps.append(ActionStep(
            step_id=seg.segment_id,
            verb=verb,
            noun=noun,
            start_sec=seg.start_sec,
            stop_sec=seg.stop_sec,
            confidence=seg.confidence,
            description=seg.description,
            preconditions=preconditions,
            effects=effects,
        ))

    return steps


def program_to_text(steps: list[ActionStep]) -> str:
    """Render the action program as human-readable text."""
    lines = ["ACTION PROGRAM", "=" * 50]

    for step in steps:
        lines.append(f"\n  [{step.start_sec:.1f}s - {step.stop_sec:.1f}s]  "
                     f"{step.verb.upper()} {step.noun}")
        lines.append(f"    Confidence: {step.confidence:.0%}")
        lines.append(f"    Description: {step.description}")
        if step.preconditions:
            lines.append(f"    PRECONDITIONS: {', '.join(step.preconditions)}")
        if step.effects:
            lines.append(f"    EFFECTS: {', '.join(step.effects)}")

    return "\n".join(lines)
