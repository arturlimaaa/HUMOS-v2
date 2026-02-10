"""
Action Segment Review UI

Tab 1: Action Segments -- review predicted actions, correct verb/noun
Tab 2: Epic-Kitchens Eval -- compare against GT, view metrics
Tab 3: Action Program -- structured program + PDDL output
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.action_program import build_action_program, program_to_text, ActionStep
from src.action_segmenter import ActionSegment
from src.pddl_generator import generate_pddl
from src.few_shot_bank import FewShotBank

# ── State ──────────────────────────────────────────────────────────────
pipeline_data: dict = {}
video_path: str | None = None
segments_list: list[dict] = []
few_shot_bank = FewShotBank()


def _load_frame(frame_idx: int) -> np.ndarray | None:
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()
    cap.release()
    return img if ret else None


def load_data(json_file):
    global pipeline_data, video_path, segments_list
    if json_file is None:
        return "No file uploaded", ""

    with open(json_file.name) as f:
        pipeline_data = json.load(f)

    video_path = pipeline_data.get("video", {}).get("path")
    segments_list = pipeline_data.get("action_segments", [])

    stats = pipeline_data.get("stats", {})
    status = (
        f"Loaded: {stats.get('frames_processed', '?')} frames, "
        f"{stats.get('total_segments', '?')} action segments\n"
        f"Video: {video_path}"
    )

    # Build segments table
    table = format_segments_table()
    return status, table


def format_segments_table() -> str:
    lines = [f"{'#':<4} {'Time':<14} {'Verb':<12} {'Noun':<16} {'Conf':<6} Description"]
    lines.append("-" * 80)
    for s in segments_list:
        lines.append(
            f"{s['segment_id']:<4} "
            f"{s['start_sec']:>5.1f}-{s['stop_sec']:<5.1f}s  "
            f"{s['verb']:<12} {s['noun']:<16} "
            f"{s['confidence']:.0%}   "
            f"{s.get('description', '')[:40]}"
        )
    return "\n".join(lines)


def get_segment_detail(seg_idx):
    if not segments_list or seg_idx >= len(segments_list):
        return None, "No segment selected"

    seg = segments_list[int(seg_idx)]
    # Load middle frame
    mid_frame = (seg["start_frame"] + seg["stop_frame"]) // 2
    img = _load_frame(mid_frame)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detail = (
        f"Segment #{seg['segment_id']}\n"
        f"Action: {seg['verb']} {seg['noun']}\n"
        f"Time: {seg['start_sec']:.2f}s - {seg['stop_sec']:.2f}s "
        f"({seg['duration_sec']:.2f}s)\n"
        f"Frames: {seg['start_frame']} - {seg['stop_frame']}\n"
        f"Confidence: {seg['confidence']:.0%}\n"
        f"Description: {seg.get('description', '')}"
    )
    return img, detail


def correct_segment(seg_idx, new_verb, new_noun):
    if not segments_list or seg_idx >= len(segments_list):
        return "Invalid segment"

    seg = segments_list[int(seg_idx)]
    old = f"{seg['verb']} {seg['noun']}"
    if new_verb and new_verb.strip():
        seg["verb"] = new_verb.strip().lower()
    if new_noun and new_noun.strip():
        seg["noun"] = new_noun.strip().lower()
    new = f"{seg['verb']} {seg['noun']}"

    # Save to few-shot bank
    mid_frame = (seg["start_frame"] + seg["stop_frame"]) // 2
    img = _load_frame(mid_frame)
    if img is not None:
        few_shot_bank.add_correction(
            class_name="action",
            state=seg["verb"],
            image_crop=img,
            context={"noun": seg["noun"], "frame_idx": mid_frame},
        )

    return f"Corrected: {old} -> {new}\nSaved to few-shot bank ({few_shot_bank.total_corrections()} total)"


def save_corrected():
    if not pipeline_data:
        return "Nothing loaded"
    pipeline_data["action_segments"] = segments_list

    out = config.OUTPUT_DIR / "corrected_output.json"
    with open(out, "w") as f:
        json.dump(pipeline_data, f, indent=2)
    return f"Saved to {out}"


def run_epic_eval(csv_file, video_id_input):
    if not pipeline_data or not csv_file:
        return "Load pipeline output and provide Epic-Kitchens CSV"

    from src.epic_kitchens_eval import load_epic_annotations, load_verb_classes, evaluate
    from src.action_segmenter import ActionSegment as AS

    gt = load_epic_annotations(csv_file.name, video_id_input.strip())
    if not gt:
        return f"No GT found for video_id={video_id_input}"

    # Convert dicts back to ActionSegment objects for evaluation
    pred = [AS(
        segment_id=s["segment_id"], verb=s["verb"], noun=s["noun"],
        start_frame=s["start_frame"], stop_frame=s["stop_frame"],
        start_sec=s["start_sec"], stop_sec=s["stop_sec"],
        confidence=s["confidence"], description=s.get("description", ""),
    ) for s in segments_list]

    ek_dir = Path(csv_file.name).parent
    verb_classes = None
    vc_path = ek_dir / "EPIC_100_verb_classes.csv"
    if vc_path.exists():
        verb_classes = load_verb_classes(str(vc_path))

    report = evaluate(pred, gt, video_id_input, verb_classes)
    return report.summary_text()


def get_program_text():
    if not segments_list:
        return "No segments", ""

    # Rebuild action steps from current (possibly corrected) segments
    segs = [ActionSegment(
        segment_id=s["segment_id"], verb=s["verb"], noun=s["noun"],
        start_frame=s["start_frame"], stop_frame=s["stop_frame"],
        start_sec=s["start_sec"], stop_sec=s["stop_sec"],
        confidence=s["confidence"], description=s.get("description", ""),
    ) for s in segments_list]

    steps = build_action_program(segs)
    prog = program_to_text(steps)
    pddl = generate_pddl(steps)
    return prog, pddl


# ── Build UI ──────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="HUMOS-v2 Action Engine", theme=gr.themes.Soft()) as app:
        gr.Markdown("# HUMOS-v2: Action Ground Truth Engine")
        gr.Markdown("Review and correct action segments. Corrections improve future predictions via the few-shot bank.")

        with gr.Row():
            file_input = gr.File(label="Upload Pipeline Output JSON", file_types=[".json"])
            load_btn = gr.Button("Load", variant="primary", scale=0)
        status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Action Segments"):
            segments_table = gr.Textbox(label="Detected Actions", lines=15, interactive=False,
                                        show_copy_button=True)

            with gr.Row():
                with gr.Column(scale=2):
                    seg_image = gr.Image(label="Frame Preview", type="numpy")
                with gr.Column(scale=1):
                    seg_detail = gr.Textbox(label="Segment Detail", lines=8, interactive=False)

            seg_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Segment #", value=0)

            gr.Markdown("### Correct Action")
            with gr.Row():
                new_verb = gr.Textbox(label="Correct Verb", placeholder="e.g. open, take, wash")
                new_noun = gr.Textbox(label="Correct Noun", placeholder="e.g. fridge, cup, plate")
                correct_btn = gr.Button("Apply Correction")
            correct_status = gr.Textbox(label="Correction Status", interactive=False)

            save_btn = gr.Button("Save Corrected Output", variant="primary")
            save_status = gr.Textbox(label="Save Status", interactive=False)

        with gr.Tab("Epic-Kitchens Eval"):
            with gr.Row():
                ek_csv = gr.File(label="EPIC_100_train.csv", file_types=[".csv"])
                ek_vid_id = gr.Textbox(label="Video ID", placeholder="P01_01")
                ek_btn = gr.Button("Evaluate", variant="primary")
            ek_result = gr.Textbox(label="Evaluation Results", lines=18, interactive=False)

        with gr.Tab("Action Program"):
            prog_btn = gr.Button("Generate Program + PDDL")
            with gr.Row():
                prog_text = gr.Textbox(label="Action Program", lines=20, interactive=False, show_copy_button=True)
                pddl_text = gr.Textbox(label="PDDL Output", lines=20, interactive=False, show_copy_button=True)

        # Wire events
        load_btn.click(load_data, inputs=[file_input], outputs=[status_box, segments_table])
        seg_slider.change(get_segment_detail, inputs=[seg_slider], outputs=[seg_image, seg_detail])
        correct_btn.click(correct_segment, inputs=[seg_slider, new_verb, new_noun], outputs=[correct_status])
        save_btn.click(save_corrected, outputs=[save_status])
        ek_btn.click(run_epic_eval, inputs=[ek_csv, ek_vid_id], outputs=[ek_result])
        prog_btn.click(get_program_text, outputs=[prog_text, pddl_text])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
