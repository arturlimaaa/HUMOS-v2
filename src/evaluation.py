"""
Evaluation module for the action-centric pipeline.
Computes metrics on predicted action segments with optional GT comparison.
"""
from __future__ import annotations

import json
from collections import Counter


def compute_pipeline_stats(pipeline_output: dict) -> dict:
    """Compute basic pipeline statistics."""
    segments = pipeline_output.get("action_segments", [])
    video = pipeline_output.get("video", {})

    verb_counts = Counter(s["verb"] for s in segments)
    noun_counts = Counter(s["noun"] for s in segments)

    total_action_time = sum(s["stop_sec"] - s["start_sec"] for s in segments)
    duration = video.get("duration_sec", 1)
    action_density = total_action_time / max(duration, 0.01)

    return {
        "total_segments": len(segments),
        "unique_verbs": len(verb_counts),
        "unique_nouns": len(noun_counts),
        "top_verbs": verb_counts.most_common(10),
        "top_nouns": noun_counts.most_common(10),
        "total_action_time_sec": round(total_action_time, 1),
        "video_duration_sec": round(duration, 1),
        "action_density": round(action_density, 2),
        "avg_confidence": round(
            sum(s["confidence"] for s in segments) / max(len(segments), 1), 3
        ),
    }


def compute_labeling_speedup(processing_time: float, video_duration: float,
                             manual_factor: float = 10.0) -> dict:
    """Estimate labeling speedup vs manual annotation."""
    manual_time = video_duration * manual_factor
    speedup = manual_time / max(processing_time, 0.01)
    return {
        "pipeline_time_sec": round(processing_time, 1),
        "estimated_manual_time_sec": round(manual_time, 0),
        "speedup_factor": round(speedup, 1),
        "cost_reduction_pct": round(max(0, (1 - processing_time / max(manual_time, 0.01)) * 100), 1),
    }


def generate_report(pipeline_output: dict) -> dict:
    """Generate a full evaluation report."""
    stats = compute_pipeline_stats(pipeline_output)
    proc_time = pipeline_output.get("stats", {}).get("processing_time_sec", 0)
    duration = pipeline_output.get("video", {}).get("duration_sec", 0)

    return {
        "pipeline_stats": stats,
        "labeling_speedup": compute_labeling_speedup(proc_time, duration),
        "vlm_stats": pipeline_output.get("vlm_stats", {}),
    }


def print_report(report: dict):
    """Pretty-print evaluation report."""
    print("\n" + "=" * 55)
    print("  WORLD2DATA ACTION RECOGNITION REPORT")
    print("=" * 55)

    stats = report.get("pipeline_stats", {})
    print(f"\n  Segments detected: {stats.get('total_segments', 0)}")
    print(f"  Unique verbs: {stats.get('unique_verbs', 0)}")
    print(f"  Unique nouns: {stats.get('unique_nouns', 0)}")
    print(f"  Action density: {stats.get('action_density', 0):.0%} of video")
    print(f"  Avg confidence: {stats.get('avg_confidence', 0):.0%}")

    top_v = stats.get("top_verbs", [])
    if top_v:
        print(f"\n  Top verbs: {', '.join(f'{v}({c})' for v, c in top_v[:5])}")
    top_n = stats.get("top_nouns", [])
    if top_n:
        print(f"  Top nouns: {', '.join(f'{n}({c})' for n, c in top_n[:5])}")

    speedup = report.get("labeling_speedup", {})
    print(f"\n  Labeling speedup: {speedup.get('speedup_factor', '?')}x")
    print(f"  Cost reduction: {speedup.get('cost_reduction_pct', '?')}%")

    vlm = report.get("vlm_stats", {})
    if vlm:
        print(f"\n  VLM calls: {vlm.get('api_calls', '?')}")
        print(f"  Total tokens: {vlm.get('total_tokens', '?')}")

    print("\n" + "=" * 55)
