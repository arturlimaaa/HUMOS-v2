"""
Symbolic state tracker: replays an action program's effects and flags steps
whose preconditions are contradicted by earlier steps.

A flag needs no ground truth. It is a label-free signal that the VLM's
labeling is internally inconsistent (e.g. "take cup" then "take knife"
with no "put" in between).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.action_program import ActionStep


@dataclass(frozen=True)
class StepCheck:
    step_id: int
    violated: tuple[str, ...]   # preconditions known false when the step ran
    checkable: bool             # False when the step declares no preconditions


def check_program(steps: list[ActionStep]) -> list[StepCheck]:
    # Open world: a fact is True, False, or absent (unknown). Unknown preconditions
    # are abduced as initially true rather than flagged.
    state: dict[str, bool] = {}
    checks = []
    for step in steps:
        violated = tuple(p for p in step.preconditions if state.get(p) is False)
        for p in step.preconditions:
            state.setdefault(p, True)
        for e in step.effects:
            if e.startswith("NOT "):
                state[e[4:]] = False
            else:
                state[e] = True
        checks.append(StepCheck(step.step_id, violated, bool(step.preconditions)))
    return checks
