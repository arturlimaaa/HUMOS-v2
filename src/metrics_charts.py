"""
Generate metric charts for the action-centric pipeline.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_action_timeline(segments: list[dict], duration: float, output_path: str):
    """Plot action segments on a timeline with verb-colored bars."""
    if not segments:
        return

    fig, ax = plt.subplots(figsize=(14, max(3, len(segments) * 0.25 + 1)))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")

    for i, seg in enumerate(segments):
        start = seg["start_sec"]
        dur = seg["stop_sec"] - start
        color_bgr = config.VERB_COLORS.get(seg["verb"], config.DEFAULT_VERB_COLOR)
        color = tuple(c / 255 for c in color_bgr)  # Already RGB in config

        ax.barh(i, dur, left=start, height=0.6, color=color, edgecolor="white", linewidth=0.3)
        if dur > duration * 0.03:
            ax.text(start + dur / 2, i, f'{seg["verb"]} {seg["noun"]}',
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels([f'{s["verb"]} {s["noun"]}' for s in segments], fontsize=7, color="white")
    ax.set_xlabel("Time (seconds)", color="white")
    ax.set_title("Action Segments Timeline", color="white", fontsize=12)
    ax.set_xlim(0, duration)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_verb_distribution(segments: list[dict], output_path: str):
    """Bar chart of verb frequency."""
    if not segments:
        return
    verb_counts = Counter(s["verb"] for s in segments)
    verbs = [v for v, _ in verb_counts.most_common(15)]
    counts = [verb_counts[v] for v in verbs]
    colors = [tuple(c / 255 for c in config.VERB_COLORS.get(v, config.DEFAULT_VERB_COLOR)) for v in verbs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(verbs, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Verb Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.2, str(c), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_over_time(segments: list[dict], output_path: str):
    """Plot confidence scores over time."""
    if not segments:
        return
    times = [(s["start_sec"] + s["stop_sec"]) / 2 for s in segments]
    confs = [s["confidence"] for s in segments]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(times, confs, c=confs, cmap="RdYlGn", vmin=0, vmax=1, s=60, edgecolors="black", linewidth=0.5)
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Review threshold")
    ax.axhline(0.85, color="green", linestyle="--", alpha=0.5, label="Auto-accept")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Confidence")
    ax.set_title("Action Recognition Confidence Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_epic_comparison(matched_pairs: list[dict], output_path: str):
    """Plot comparison chart for Epic-Kitchens evaluation."""
    if not matched_pairs:
        return

    labels = ["Verb Match", "Noun Match", "Both Match"]
    values = [
        sum(1 for p in matched_pairs if p["verb_match"]) / max(len(matched_pairs), 1),
        sum(1 for p in matched_pairs if p["noun_match"]) / max(len(matched_pairs), 1),
        sum(1 for p in matched_pairs if p["verb_match"] and p["noun_match"]) / max(len(matched_pairs), 1),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4ECDC4", "#45B7D1", "#96CEB4"],
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Action Recognition Accuracy vs Epic-Kitchens GT")
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_charts(pipeline_output: dict, output_dir: str,
                        epic_report: dict | None = None):
    """Generate all metric charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    segments = pipeline_output.get("action_segments", [])
    duration = pipeline_output.get("video", {}).get("duration_sec", 0)

    print("Generating charts...")
    plot_action_timeline(segments, duration, str(out / "chart_action_timeline.png"))
    plot_verb_distribution(segments, str(out / "chart_verb_distribution.png"))
    plot_confidence_over_time(segments, str(out / "chart_confidence.png"))

    if epic_report and epic_report.get("matched_pairs"):
        plot_epic_comparison(epic_report["matched_pairs"], str(out / "chart_epic_comparison.png"))

    print(f"  Charts saved to {out}/")
