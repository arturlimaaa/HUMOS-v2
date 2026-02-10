"""
Smoke tests for the action-centric World2Data pipeline.
"""
import sys
import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import config


def create_test_video(path: str, n_frames: int = 60, fps: float = 30.0):
    """Create a simple test video."""
    w, h = 640, 480
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        cv2.putText(frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_video_loader():
    from src.video_loader import VideoLoader
    path = str(config.VIDEO_DIR / "test_synthetic.mp4")
    create_test_video(path)
    loader = VideoLoader(path, sample_rate=5)
    frames = list(loader.frames())
    assert len(frames) == 12, f"Expected 12 frames, got {len(frames)}"
    assert frames[0].frame_idx == 0
    print(f"  {loader}")
    print("  PASS: video_loader")


def test_action_recognizer():
    from src.action_recognizer import ActionRecognizer
    recognizer = ActionRecognizer(backend="heuristic")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = recognizer.recognize_frame(dummy, 0, 0.0)
    assert result.verb == "idle"
    assert result.confidence > 0
    print(f"  Result: {result.verb}/{result.noun} ({result.confidence:.0%})")
    print("  PASS: action_recognizer")


def test_action_segmenter():
    from src.action_recognizer import FrameAction
    from src.action_segmenter import ActionSegmenter

    # Simulate a sequence: idle -> open fridge -> idle -> take cup
    actions = [
        FrameAction(0, 0.0, "idle", "none", "idle", 0.5, ""),
        FrameAction(5, 0.17, "open", "fridge", "approaching", 0.8, "Opening fridge"),
        FrameAction(10, 0.33, "open", "fridge", "manipulating", 0.85, "Opening fridge"),
        FrameAction(15, 0.50, "open", "fridge", "manipulating", 0.9, "Opening fridge"),
        FrameAction(20, 0.67, "idle", "none", "idle", 0.5, ""),
        FrameAction(25, 0.83, "take", "cup", "contact", 0.7, "Taking cup"),
        FrameAction(30, 1.0, "take", "cup", "manipulating", 0.8, "Taking cup"),
        FrameAction(35, 1.17, "take", "cup", "releasing", 0.75, "Taking cup"),
    ]

    segmenter = ActionSegmenter(min_frames=2, merge_gap=2)
    segments = segmenter.segment(actions)

    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
    assert segments[0].verb == "open"
    assert segments[0].noun == "fridge"
    assert segments[1].verb == "take"
    assert segments[1].noun == "cup"
    for s in segments:
        print(f"  [{s.start_sec:.2f}-{s.stop_sec:.2f}s] {s.verb} {s.noun} ({s.confidence:.0%})")
    print("  PASS: action_segmenter")


def test_action_program():
    from src.action_segmenter import ActionSegment
    from src.action_program import build_action_program, program_to_text

    segments = [
        ActionSegment(0, "open", "fridge", 5, 15, 0.17, 0.50, 0.85, "Opening fridge"),
        ActionSegment(1, "take", "cup", 25, 35, 0.83, 1.17, 0.75, "Taking cup"),
    ]

    steps = build_action_program(segments)
    assert len(steps) == 2
    assert "is_closed(fridge)" in steps[0].preconditions
    assert "is_open(fridge)" in steps[0].effects
    assert "hand_free(human)" in steps[1].preconditions
    assert "holding(human, cup)" in steps[1].effects

    text = program_to_text(steps)
    assert "OPEN" in text
    assert "TAKE" in text
    print(f"  Steps: {len(steps)}")
    print(f"  Step 0 pre: {steps[0].preconditions}")
    print(f"  Step 0 eff: {steps[0].effects}")
    print("  PASS: action_program")


def test_pddl_generator():
    from src.action_program import ActionStep
    from src.pddl_generator import generate_pddl

    steps = [
        ActionStep(0, "open", "fridge", 0.17, 0.50, 0.85, "Opening fridge",
                   ["is_closed(fridge)"], ["is_open(fridge)"]),
        ActionStep(1, "take", "cup", 0.83, 1.17, 0.75, "Taking cup",
                   ["hand_free(human)"], ["holding(human, cup)"]),
    ]
    pddl = generate_pddl(steps)
    assert "(domain kitchen-actions)" in pddl
    assert "fridge" in pddl
    assert "cup" in pddl
    assert "open" in pddl
    print(f"  PDDL length: {len(pddl)} chars")
    print("  PASS: pddl_generator")


def test_epic_kitchens_eval():
    from src.action_segmenter import ActionSegment
    from src.epic_kitchens_eval import EpicSegment, evaluate

    predicted = [
        ActionSegment(0, "open", "door", 8, 202, 0.14, 3.37, 0.8, "Opening door"),
        ActionSegment(1, "turn-on", "light", 262, 370, 4.37, 6.17, 0.7, "Turning on light"),
    ]
    gt = [
        EpicSegment("P01_01_0", "P01_01", 8, 202, 0.14, 3.37, "open door", "open", 3, "door", 3),
        EpicSegment("P01_01_1", "P01_01", 262, 370, 4.37, 6.17, "turn on light", "turn-on", 6, "light", 114),
    ]

    report = evaluate(predicted, gt, "P01_01")
    assert report.verb_accuracy == 1.0, f"Verb acc={report.verb_accuracy}"
    assert report.noun_accuracy == 1.0, f"Noun acc={report.noun_accuracy}"
    assert report.mean_temporal_iou > 0.9
    print(f"  Verb acc: {report.verb_accuracy:.0%}")
    print(f"  Noun acc: {report.noun_accuracy:.0%}")
    print(f"  Temporal IoU: {report.mean_temporal_iou:.3f}")
    print("  PASS: epic_kitchens_eval")


def test_few_shot_bank():
    from src.few_shot_bank import FewShotBank
    tmp = tempfile.mkdtemp()
    bank = FewShotBank(tmp)
    dummy = np.zeros((50, 50, 3), dtype=np.uint8)
    bank.add_correction("action", "open", dummy, {"noun": "fridge"})
    bank.add_correction("action", "take", dummy, {"noun": "cup"})
    assert bank.total_corrections() == 2
    exs = bank.get_examples("action", k=2)
    assert len(exs) == 2
    print(f"  Bank: {bank}")
    shutil.rmtree(tmp)
    print("  PASS: few_shot_bank")


def test_visualization():
    from src.visualization import draw_action_label, draw_action_timeline, VideoWriter
    from src.action_segmenter import ActionSegment

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_action_label(img, "open", "fridge", 0.85, "manipulating")
    assert result.shape == img.shape

    segs = [ActionSegment(0, "open", "fridge", 0, 30, 0.0, 1.0, 0.85, "test")]
    result2 = draw_action_timeline(img, segs, 0.5, 3.0)
    assert result2.shape[0] > img.shape[0]  # Timeline adds height
    print("  PASS: visualization")


if __name__ == "__main__":
    print("=" * 55)
    print("World2Data Action Pipeline -- Smoke Tests")
    print("=" * 55)

    tests = [
        ("Video Loader", test_video_loader),
        ("Action Recognizer", test_action_recognizer),
        ("Action Segmenter", test_action_segmenter),
        ("Action Program", test_action_program),
        ("PDDL Generator", test_pddl_generator),
        ("Epic-Kitchens Eval", test_epic_kitchens_eval),
        ("Few-Shot Bank", test_few_shot_bank),
        ("Visualization", test_visualization),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 55)
