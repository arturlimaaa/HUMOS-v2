"""
Consistency-vs-confidence experiment.

Question: does a symbolic consistency violation predict a labeling error
better than the VLM's own confidence score?

Run the pipeline with Epic-Kitchens evaluation on each video first:

    for v in P01_01 P01_02 P02_01; do
      python3 main.py data/videos/$v.MP4 \
        --epic-csv data/epic_kitchens/EPIC_100_validation.csv --video-id $v
    done

Then:

    python3 experiment.py data/outputs

Both detectors are compared at the same budget: confidence flags exactly as
many segments as consistency did, taking the least confident ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.action_program import ActionStep
from src.state_tracker import check_program


def load_rows(out_dir: str | Path) -> list[dict]:
    rows = []
    for eval_path in sorted(Path(out_dir).glob("*_epic_eval.json")):
        stem = eval_path.name.removesuffix("_epic_eval.json")
        output = json.loads((eval_path.parent / f"{stem}_output.json").read_text())
        steps = [ActionStep(**s) for s in output["action_program"]]
        by_id = {s.step_id: s for s in steps}
        checks = {c.step_id: c for c in check_program(steps)}
        for pair in json.loads(eval_path.read_text())["matched_pairs"]:
            sid = pair["pred_segment_id"]
            rows.append({
                "video": stem,
                "segment_id": sid,
                "error": not (pair["verb_match"] and pair["noun_match"]),
                "confidence": by_id[sid].confidence,
                "violated": bool(checks[sid].violated),
                "checkable": checks[sid].checkable,
            })
    return rows


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Probability a random error scores higher than a random correct segment."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    # ponytail: O(P*N) pairwise; switch to rank sums if segments exceed ~1e4
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _detector(scores: list[float], labels: list[bool], budget: int) -> dict:
    flagged = sorted(range(len(scores)), key=lambda i: -scores[i])[:budget]
    hits = sum(labels[i] for i in flagged)
    return {
        "precision": round(hits / max(budget, 1), 3),
        "recall": round(hits / max(sum(labels), 1), 3),
        "auroc": round(auroc(scores, labels), 3),
    }


def report(rows: list[dict]) -> dict:
    labels = [r["error"] for r in rows]
    flags = [float(r["violated"]) for r in rows]
    budget = int(sum(flags))
    return {
        "n": len(rows),
        "error_rate": round(sum(labels) / max(len(rows), 1), 3),
        "checkable_rate": round(sum(r["checkable"] for r in rows) / max(len(rows), 1), 3),
        "flag_rate": round(budget / max(len(rows), 1), 3),
        "consistency": _detector(flags, labels, budget),
        "confidence": _detector([1 - r["confidence"] for r in rows], labels, budget),
    }


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/outputs"
    rows = load_rows(out_dir)
    if not rows:
        sys.exit(f"No *_epic_eval.json under {out_dir}. Run main.py with --epic-csv first.")
    result = report(rows)
    Path(out_dir, "consistency_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
