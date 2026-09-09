"""
Tests for the symbolic state tracker and the consistency-vs-confidence experiment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.action_program import ActionStep


def _step(i, verb, noun, pre, eff):
    return ActionStep(i, verb, noun, float(i), i + 1.0, 0.8, "", pre, eff)


def test_state_tracker_flags_known_false_preconditions():
    from src.state_tracker import check_program

    steps = [
        _step(0, "open", "fridge", ["is_closed(fridge)"], ["is_open(fridge)", "NOT is_closed(fridge)"]),
        _step(1, "take", "cup", ["hand_free(human)"], ["holding(human, cup)", "NOT hand_free(human)"]),
        _step(2, "take", "knife", ["hand_free(human)"], ["holding(human, knife)", "NOT hand_free(human)"]),
        _step(3, "close", "fridge", ["is_open(fridge)"], ["is_closed(fridge)", "NOT is_open(fridge)"]),
        _step(4, "close", "fridge", ["is_open(fridge)"], ["is_closed(fridge)", "NOT is_open(fridge)"]),
        _step(5, "dance", "none", [], []),
    ]

    checks = [(c.step_id, c.violated, c.checkable) for c in check_program(steps)]

    assert checks == [
        (0, (), True),                   # unknown initial state is abduced, not flagged
        (1, (), True),
        (2, ("hand_free(human)",), True),  # hand known busy since step 1
        (3, (), True),
        (4, ("is_open(fridge)",), True),   # fridge known closed since step 3
        (5, (), False),                  # verb outside the domain: nothing to check
    ]
    print("  PASS: state_tracker")


def test_experiment_report_compares_flag_and_confidence_at_equal_budget():
    from experiment import report

    rows = [
        {"error": True,  "confidence": 0.90, "violated": True,  "checkable": True},
        {"error": True,  "confidence": 0.90, "violated": True,  "checkable": True},
        {"error": True,  "confidence": 0.20, "violated": False, "checkable": True},
        {"error": False, "confidence": 0.30, "violated": False, "checkable": True},
        {"error": False, "confidence": 0.95, "violated": False, "checkable": True},
        {"error": False, "confidence": 0.95, "violated": False, "checkable": True},
    ]

    assert report(rows) == {
        "n": 6,
        "error_rate": 0.5,
        "checkable_rate": 1.0,
        "flag_rate": 0.333,
        "consistency": {"precision": 1.0, "recall": 0.667, "auroc": 0.833},
        "confidence": {"precision": 0.5, "recall": 0.333, "auroc": 0.778},
    }
    print("  PASS: experiment_report")


if __name__ == "__main__":
    for fn in (test_state_tracker_flags_known_false_preconditions,
               test_experiment_report_compares_flag_and_confidence_at_equal_budget):
        print(f"--- {fn.__name__} ---")
        fn()
