"""
World2Data: Action-Centric Ground Truth Generation for Humanoid Navigation

Watches humans in egocentric video and generates structured action programs
(verb + noun + temporal boundaries + preconditions/effects).

Usage:
    python3 main.py <video>                         # Full pipeline
    python3 main.py <video> --vlm heuristic         # No API (testing)
    python3 main.py <video> --epic-csv <csv> --video-id P01_01  # With eval
"""
import sys
import json
import time
import argparse
from pathlib import Path

import config
from src.video_loader import VideoLoader
from src.action_recognizer import ActionRecognizer
from src.action_segmenter import ActionSegmenter
from src.action_program import build_action_program, program_to_text
from src.pddl_generator import generate_pddl
from src.visualization import draw_action_label, draw_action_timeline, VideoWriter
from src.evaluation import generate_report, print_report
from src.metrics_charts import generate_all_charts


def run_pipeline(
    video_path: str,
    output_dir: str | None = None,
    epic_csv: str | None = None,
    epic_video_id: str | None = None,
) -> dict:
    """
    Full action-centric pipeline:
    Video -> Frame extraction -> VLM action recognition per frame
    -> Action segmentation -> Action program -> PDDL -> Evaluation
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # ── Load Video ─────────────────────────────────────────────────
    print(f"Loading video: {video_path}")
    loader = VideoLoader(
        str(video_path),
        sample_rate=config.FRAME_SAMPLE_RATE,
        max_frames=config.MAX_FRAMES,
    )
    print(f"  {loader}")

    # ── Initialize ─────────────────────────────────────────────────
    print("Initializing action recognizer...")
    recognizer = ActionRecognizer()
    segmenter = ActionSegmenter()

    # ── Per-Frame Action Recognition ───────────────────────────────
    frame_actions = []
    annotated_frames = []
    frame_count = 0
    t_start = time.time()
    print("Recognizing actions in frames...")

    for frame in loader.frames():
        fa = recognizer.recognize_frame(frame.image, frame.frame_idx, frame.timestamp_sec)
        frame_actions.append(fa)

        # Draw annotation for output video
        annotated = draw_action_label(
            frame.image, fa.verb, fa.noun, fa.confidence, fa.phase
        )
        annotated_frames.append((annotated, frame.timestamp_sec))

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  Processed {frame_count} frames ({frame_count / max(elapsed, 0.01):.1f} fps) "
                  f"| Last: {fa.verb} {fa.noun} ({fa.confidence:.0%})")

    # ── Segment ────────────────────────────────────────────────────
    print("\nSegmenting actions...")
    segments = segmenter.segment(frame_actions)
    print(f"  {len(segments)} action segments found")
    for seg in segments:
        print(f"    [{seg.start_sec:.1f}-{seg.stop_sec:.1f}s] {seg.verb} {seg.noun} ({seg.confidence:.0%})")

    # ── Action Program ─────────────────────────────────────────────
    print("\nBuilding action program...")
    steps = build_action_program(segments)
    program_text = program_to_text(steps)

    # ── PDDL ───────────────────────────────────────────────────────
    video_info = {
        "path": str(video_path),
        "duration_sec": loader.metadata.duration_sec,
    }
    pddl = generate_pddl(steps, video_info)

    # ── Write Output Video ─────────────────────────────────────────
    print("\nWriting output video...")
    out_video = out_dir / f"{stem}_actions.mp4"
    output_fps = loader.metadata.fps / config.FRAME_SAMPLE_RATE
    sample_h, sample_w = annotated_frames[0][0].shape[:2] if annotated_frames else (720, 1280)
    timeline_h = 40
    out_h = sample_h + timeline_h

    writer = VideoWriter(str(out_video), output_fps, sample_w, out_h)
    for annotated, ts in annotated_frames:
        with_timeline = draw_action_timeline(
            annotated, segments, ts, loader.metadata.duration_sec
        )
        writer.write(with_timeline)
    writer.release()

    # ── Build Output ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    pipeline_output = {
        "video": {
            "path": str(video_path),
            "fps": loader.metadata.fps,
            "total_frames": loader.metadata.total_frames,
            "width": loader.metadata.width,
            "height": loader.metadata.height,
            "duration_sec": loader.metadata.duration_sec,
            "sample_rate": config.FRAME_SAMPLE_RATE,
        },
        "action_segments": [seg.to_dict() for seg in segments],
        "action_program": [step.to_dict() for step in steps],
        "frame_actions": [
            {
                "frame_idx": fa.frame_idx,
                "timestamp_sec": fa.timestamp_sec,
                "verb": fa.verb,
                "noun": fa.noun,
                "phase": fa.phase,
                "confidence": fa.confidence,
                "description": fa.description,
            }
            for fa in frame_actions
        ],
        "stats": {
            "frames_processed": frame_count,
            "total_segments": len(segments),
            "total_steps": len(steps),
            "processing_time_sec": round(elapsed, 2),
        },
        "vlm_stats": recognizer.stats,
    }

    # Save outputs
    out_json = out_dir / f"{stem}_output.json"
    with open(out_json, "w") as f:
        json.dump(pipeline_output, f, indent=2)

    out_pddl = out_dir / f"{stem}_actions.pddl"
    with open(out_pddl, "w") as f:
        f.write(pddl)

    out_program = out_dir / f"{stem}_program.txt"
    with open(out_program, "w") as f:
        f.write(program_text)

    # ── Charts ─────────────────────────────────────────────────────
    chart_dir = out_dir / "charts"

    # ── Epic-Kitchens Evaluation (optional) ────────────────────────
    epic_report_data = None
    if epic_csv and epic_video_id:
        print(f"\nRunning Epic-Kitchens evaluation (video_id={epic_video_id})...")
        from src.epic_kitchens_eval import (
            load_epic_annotations, load_verb_classes, load_noun_classes, evaluate
        )
        gt_segments = load_epic_annotations(epic_csv, epic_video_id)
        print(f"  Loaded {len(gt_segments)} GT segments")

        # Load taxonomy if available
        ek_dir = Path(epic_csv).parent
        verb_classes = None
        noun_classes = None
        verb_csv = ek_dir / "EPIC_100_verb_classes.csv"
        noun_csv = ek_dir / "EPIC_100_noun_classes.csv"
        if verb_csv.exists():
            verb_classes = load_verb_classes(str(verb_csv))
        if noun_csv.exists():
            noun_classes = load_noun_classes(str(noun_csv))

        epic_report = evaluate(segments, gt_segments, epic_video_id, verb_classes, noun_classes)
        print(f"\n{epic_report.summary_text()}")

        epic_report_data = epic_report.to_dict()
        epic_report_data["matched_pairs"] = epic_report.matched_pairs
        out_eval = out_dir / f"{stem}_epic_eval.json"
        with open(out_eval, "w") as f:
            json.dump(epic_report_data, f, indent=2)

    generate_all_charts(pipeline_output, str(chart_dir), epic_report_data)

    # ── Report ─────────────────────────────────────────────────────
    report = generate_report(pipeline_output)
    print_report(report)

    print(f"\n  Output video:   {out_video}")
    print(f"  Pipeline JSON:  {out_json}")
    print(f"  PDDL:           {out_pddl}")
    print(f"  Action program: {out_program}")
    print(f"  Charts:         {chart_dir}/")

    return pipeline_output


def main():
    parser = argparse.ArgumentParser(
        description="World2Data: Action-Centric Ground Truth for Humanoid Navigation"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    parser.add_argument("--vlm", default=None, help="VLM backend (openai, heuristic)")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--epic-csv", default=None, help="Path to EPIC_100_train.csv or validation.csv")
    parser.add_argument("--video-id", default=None, help="Epic-Kitchens video_id for evaluation")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")

    args = parser.parse_args()

    if args.vlm:
        config.VLM_BACKEND = args.vlm
    if args.sample_rate:
        config.FRAME_SAMPLE_RATE = args.sample_rate
    if args.max_frames:
        config.MAX_FRAMES = args.max_frames

    run_pipeline(args.video, args.output, args.epic_csv, args.video_id)


if __name__ == "__main__":
    main()
