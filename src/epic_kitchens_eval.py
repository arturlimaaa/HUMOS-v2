"""
Epic-Kitchens-100 Evaluation Module.

Loads Epic-Kitchens annotations (freely available CSV files), compares
predicted action segments against human ground truth, and computes:
  - Verb accuracy (with synonym mapping)
  - Noun accuracy
  - Temporal IoU
  - Segment precision / recall at IoU thresholds
  - Action sequence edit distance
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from src.action_segmenter import ActionSegment


@dataclass
class EpicSegment:
    """A ground truth action segment from Epic-Kitchens."""
    narration_id: str
    video_id: str
    start_frame: int
    stop_frame: int
    start_sec: float
    stop_sec: float
    narration: str
    verb: str
    verb_class: int
    noun: str
    noun_class: int


@dataclass
class EvalReport:
    """Evaluation results against Epic-Kitchens ground truth."""
    video_id: str
    n_predicted: int
    n_ground_truth: int
    verb_accuracy: float
    noun_accuracy: float
    action_accuracy: float   # Both verb AND noun correct
    mean_temporal_iou: float
    precision_at_03: float   # Segment P at IoU 0.3
    recall_at_03: float
    precision_at_05: float   # Segment P at IoU 0.5
    recall_at_05: float
    edit_distance: int
    matched_pairs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "n_predicted": self.n_predicted,
            "n_ground_truth": self.n_ground_truth,
            "verb_accuracy": round(self.verb_accuracy, 3),
            "noun_accuracy": round(self.noun_accuracy, 3),
            "action_accuracy": round(self.action_accuracy, 3),
            "mean_temporal_iou": round(self.mean_temporal_iou, 3),
            "precision_at_iou_0.3": round(self.precision_at_03, 3),
            "recall_at_iou_0.3": round(self.recall_at_03, 3),
            "precision_at_iou_0.5": round(self.precision_at_05, 3),
            "recall_at_iou_0.5": round(self.recall_at_05, 3),
            "edit_distance": self.edit_distance,
        }

    def summary_text(self) -> str:
        lines = [
            f"Epic-Kitchens Evaluation: {self.video_id}",
            "=" * 50,
            f"Predicted segments: {self.n_predicted}",
            f"Ground truth segments: {self.n_ground_truth}",
            f"",
            f"Verb accuracy:   {self.verb_accuracy:.1%}",
            f"Noun accuracy:   {self.noun_accuracy:.1%}",
            f"Action accuracy: {self.action_accuracy:.1%}  (verb+noun both correct)",
            f"",
            f"Mean temporal IoU: {self.mean_temporal_iou:.3f}",
            f"Precision @IoU=0.3: {self.precision_at_03:.1%}",
            f"Recall @IoU=0.3:    {self.recall_at_03:.1%}",
            f"Precision @IoU=0.5: {self.precision_at_05:.1%}",
            f"Recall @IoU=0.5:    {self.recall_at_05:.1%}",
            f"",
            f"Sequence edit distance: {self.edit_distance}",
        ]
        return "\n".join(lines)


# ── Loading ────────────────────────────────────────────────────────────

def _parse_timestamp(ts: str) -> float:
    """Parse HH:mm:ss.SS to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def load_verb_classes(csv_path: str | Path) -> dict[int, list[str]]:
    """Load Epic-Kitchens verb classes: {class_id: [verb_synonyms]}."""
    classes = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["id"])
            # Parse the instances list
            instances_raw = row["instances"].strip()
            instances = [w.strip(" '\"") for w in instances_raw.strip("[]").split(",")]
            classes[vid] = [row["key"]] + [i for i in instances if i != row["key"]]
    return classes


def load_noun_classes(csv_path: str | Path) -> dict[int, list[str]]:
    """Load Epic-Kitchens noun classes: {class_id: [noun_synonyms]}."""
    classes = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["id"])
            instances_raw = row["instances"].strip()
            instances = [w.strip(" '\"") for w in instances_raw.strip("[]").split(",")]
            classes[nid] = [row["key"]] + [i for i in instances if i != row["key"]]
    return classes


def load_epic_annotations(
    csv_path: str | Path,
    video_id: str | None = None,
) -> list[EpicSegment]:
    """Load action annotations from Epic-Kitchens CSV."""
    segments = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if video_id and row["video_id"] != video_id:
                continue
            segments.append(EpicSegment(
                narration_id=row["narration_id"],
                video_id=row["video_id"],
                start_frame=int(row["start_frame"]),
                stop_frame=int(row["stop_frame"]),
                start_sec=_parse_timestamp(row["start_timestamp"]),
                stop_sec=_parse_timestamp(row["stop_timestamp"]),
                narration=row["narration"],
                verb=row["verb"],
                verb_class=int(row["verb_class"]),
                noun=row["noun"],
                noun_class=int(row["noun_class"]),
            ))
    return segments


# ── Metrics ────────────────────────────────────────────────────────────

def _temporal_iou(pred_start: float, pred_stop: float, gt_start: float, gt_stop: float) -> float:
    """Compute temporal IoU between two time segments."""
    inter_start = max(pred_start, gt_start)
    inter_stop = min(pred_stop, gt_stop)
    inter = max(0, inter_stop - inter_start)
    union = (pred_stop - pred_start) + (gt_stop - gt_start) - inter
    return inter / max(union, 1e-6)


def _verb_matches(pred_verb: str, gt_verb: str, verb_classes: dict | None = None) -> bool:
    """Check if predicted verb matches GT, considering synonyms."""
    pred_v = pred_verb.lower().strip()
    gt_v = gt_verb.lower().strip()
    if pred_v == gt_v:
        return True
    # Check synonym classes
    if verb_classes:
        for _, synonyms in verb_classes.items():
            syns_lower = [s.lower() for s in synonyms]
            if pred_v in syns_lower and gt_v in syns_lower:
                return True
    return False


def _noun_matches(pred_noun: str, gt_noun: str, noun_classes: dict | None = None) -> bool:
    """Check if predicted noun matches GT, considering synonyms."""
    pred_n = pred_noun.lower().strip().replace("_", " ")
    gt_n = gt_noun.lower().strip().replace("_", " ")
    if pred_n == gt_n:
        return True
    # Partial match (pred contains GT or vice versa)
    if pred_n in gt_n or gt_n in pred_n:
        return True
    if noun_classes:
        for _, synonyms in noun_classes.items():
            syns_lower = [s.lower() for s in synonyms]
            if pred_n in syns_lower and gt_n in syns_lower:
                return True
    return False


def _edit_distance(seq1: list[str], seq2: list[str]) -> int:
    """Levenshtein edit distance between two action label sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def _segment_precision_recall(
    pred_segments: list,
    gt_segments: list,
    iou_threshold: float,
) -> tuple[float, float]:
    """Compute segment precision and recall at a given IoU threshold."""
    matched_gt = set()
    tp = 0

    for pred in pred_segments:
        for gi, gt in enumerate(gt_segments):
            if gi in matched_gt:
                continue
            iou = _temporal_iou(pred.start_sec, pred.stop_sec, gt.start_sec, gt.stop_sec)
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(gi)
                break

    precision = tp / max(len(pred_segments), 1)
    recall = tp / max(len(gt_segments), 1)
    return precision, recall


# ── Main Evaluation ────────────────────────────────────────────────────

def evaluate(
    predicted: list[ActionSegment],
    gt_segments: list[EpicSegment],
    video_id: str = "",
    verb_classes: dict | None = None,
    noun_classes: dict | None = None,
) -> EvalReport:
    """
    Compare predicted action segments against Epic-Kitchens ground truth.
    """
    # Match predicted to GT by temporal IoU (greedy)
    matched_pairs = []
    used_gt = set()
    ious = []
    verb_correct = 0
    noun_correct = 0
    action_correct = 0
    total_matched = 0

    for pred in predicted:
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gt_segments):
            if gi in used_gt:
                continue
            iou = _temporal_iou(pred.start_sec, pred.stop_sec, gt.start_sec, gt.stop_sec)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi >= 0 and best_iou > 0.1:
            gt = gt_segments[best_gi]
            used_gt.add(best_gi)
            ious.append(best_iou)
            total_matched += 1

            v_match = _verb_matches(pred.verb, gt.verb, verb_classes)
            n_match = _noun_matches(pred.noun, gt.noun, noun_classes)
            if v_match:
                verb_correct += 1
            if n_match:
                noun_correct += 1
            if v_match and n_match:
                action_correct += 1

            matched_pairs.append({
                "pred_verb": pred.verb, "pred_noun": pred.noun,
                "gt_verb": gt.verb, "gt_noun": gt.noun,
                "gt_narration": gt.narration,
                "temporal_iou": round(best_iou, 3),
                "verb_match": v_match, "noun_match": n_match,
            })

    # Precision / recall at IoU thresholds
    p03, r03 = _segment_precision_recall(predicted, gt_segments, 0.3)
    p05, r05 = _segment_precision_recall(predicted, gt_segments, 0.5)

    # Edit distance on action sequences
    pred_seq = [f"{s.verb}_{s.noun}" for s in predicted]
    gt_seq = [f"{g.verb}_{g.noun}" for g in gt_segments]
    ed = _edit_distance(pred_seq, gt_seq)

    return EvalReport(
        video_id=video_id,
        n_predicted=len(predicted),
        n_ground_truth=len(gt_segments),
        verb_accuracy=verb_correct / max(total_matched, 1),
        noun_accuracy=noun_correct / max(total_matched, 1),
        action_accuracy=action_correct / max(total_matched, 1),
        mean_temporal_iou=float(np.mean(ious)) if ious else 0.0,
        precision_at_03=p03,
        recall_at_03=r03,
        precision_at_05=p05,
        recall_at_05=r05,
        edit_distance=ed,
        matched_pairs=matched_pairs,
    )
