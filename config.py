"""
World2Data Pipeline Configuration
Action-Centric Ground Truth Engine for Humanoid Navigation
"""
from pathlib import Path
import os


# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "outputs"
GT_DIR = DATA_DIR / "ground_truth"
SCHEMA_DIR = PROJECT_ROOT / "schemas"
FEW_SHOT_BANK_DIR = DATA_DIR / "few_shot_bank"
EPIC_KITCHENS_DIR = DATA_DIR / "epic_kitchens"

# ── Video Processing ───────────────────────────────────────────────────
FRAME_SAMPLE_RATE = 6          # Process every Nth frame (~5 fps from 30fps video)
MAX_FRAMES = None              # None = process all frames

# ── VLM (Action Recognition) ──────────────────────────────────────────
VLM_BACKEND = "openai"         # "openai" or "heuristic"
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

ACTION_PROMPT = """You are observing an egocentric (first-person) kitchen video frame.
Describe the SINGLE primary action the camera-wearer's hands are performing RIGHT NOW.

Return ONLY valid JSON, no other text:
{"verb": "<action verb from: take, put, open, close, wash, cut, pour, mix, turn-on, turn-off, move, throw, insert, remove, peel, squeeze, shake, adjust, dry, stir, scoop, or idle>",
 "noun": "<object being acted on, e.g. fridge, cup, drawer, plate, tap, knife, pan>",
 "phase": "<approaching|contact|manipulating|releasing|idle>",
 "confidence": <float 0.0-1.0>,
 "description": "<one sentence: what the human is doing>"}

If the human is not interacting with any object, use verb="idle" and noun="none".
Use simple, lowercase words. Be specific about the noun (e.g. "fridge door" not "appliance")."""

# ── Action Segmentation ───────────────────────────────────────────────
SEGMENT_MIN_FRAMES = 2        # Minimum frames for a valid action segment
SEGMENT_MERGE_GAP = 2         # Max gap (in sampled frames) to merge same-action segments
SEGMENT_MIN_CONFIDENCE = 0.3  # Below this, treat as idle/noise

# ── Visualization ──────────────────────────────────────────────────────
VERB_COLORS = {
    "take":     (46, 204, 113),    # Green
    "put":      (52, 152, 219),    # Blue
    "open":     (241, 196, 15),    # Yellow
    "close":    (231, 76, 60),     # Red
    "wash":     (26, 188, 156),    # Teal
    "cut":      (192, 57, 43),     # Dark red
    "pour":     (155, 89, 182),    # Purple
    "mix":      (243, 156, 18),    # Orange
    "turn-on":  (22, 160, 133),    # Dark teal
    "turn-off": (149, 165, 166),   # Gray
    "move":     (41, 128, 185),    # Dark blue
    "idle":     (100, 100, 100),   # Dark gray
}
DEFAULT_VERB_COLOR = (180, 180, 180)
