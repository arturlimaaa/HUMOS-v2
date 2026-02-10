# World2Data: Action-Centric Ground Truth for Humanoid Navigation

> The human's behavior IS the ground truth. Watch what people do, not what objects exist.

## What It Does

World2Data watches humans in egocentric (first-person) kitchen video and automatically generates **structured action programs** -- the ground truth that humanoid robots need to learn navigation and manipulation.

**Input**: Raw egocentric video of a human cooking, cleaning, navigating
**Output**:
- **Action segments** with verb + noun + temporal boundaries (e.g., "open fridge, 1.2s-2.8s")
- **Action programs** with preconditions and effects (e.g., "PRE: is_closed(fridge), EFFECT: is_open(fridge)")
- **PDDL plans** consumable by robot task planners
- **Accuracy metrics** validated against Epic-Kitchens-100 (90K human-annotated action segments)

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy Pillow matplotlib openai gradio

# Run on your own video (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python3 main.py path/to/video.mp4

# Run without API key (heuristic mode, for testing)
python3 main.py path/to/video.mp4 --vlm heuristic

# Run with Epic-Kitchens evaluation
python3 main.py video.mp4 --epic-csv data/epic_kitchens/EPIC_100_train.csv --video-id P01_01

# Launch review UI
python3 hitl_app.py

# Run tests
python3 test_pipeline.py
```

## Architecture

```
Video -> Frame Extraction (every Nth frame)
  -> GPT-4o Vision: "What action is the human performing?"
    -> Per-frame: {verb, noun, phase, confidence}
  -> Action Segmenter: group into temporal segments
  -> Action Program Builder: add preconditions/effects
  -> PDDL Generator: robot-consumable plan
  -> Epic-Kitchens Evaluator: benchmark accuracy
```

## Epic-Kitchens Integration

Download annotations (free, no license required):
```bash
mkdir -p data/epic_kitchens
cd data/epic_kitchens
curl -O https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv
curl -O https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_validation.csv
curl -O https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_verb_classes.csv
curl -O https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_noun_classes.csv
```

Then evaluate:
```bash
python3 main.py epic_video.mp4 --epic-csv data/epic_kitchens/EPIC_100_train.csv --video-id P01_01
```

## Project Structure

```
NavGraph-v2/
  main.py                   # Pipeline orchestrator
  run_demo.py               # One-command demo runner
  hitl_app.py               # Gradio review UI
  config.py                 # Configuration
  test_pipeline.py          # Smoke tests (8 tests)
  src/
    video_loader.py         # OpenCV frame extraction
    action_recognizer.py    # GPT-4o Vision action recognition
    action_segmenter.py     # Group frames into action segments
    action_program.py       # Build programs with pre/post conditions
    epic_kitchens_eval.py   # Benchmark evaluation against Epic-Kitchens
    pddl_generator.py       # Generate PDDL from action programs
    evaluation.py           # Pipeline metrics
    metrics_charts.py       # Matplotlib charts
    visualization.py        # Video overlay rendering
    few_shot_bank.py        # Store corrections for future improvement
  schemas/
    epic_kitchens_mapping.json  # Verb -> precondition/effect mapping
    pddl_domain.pddl           # PDDL domain definition
  data/
    videos/                 # Input videos
    outputs/                # Pipeline outputs
    epic_kitchens/          # Downloaded annotation CSVs
    few_shot_bank/          # Corrected examples
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Verb Accuracy | % of segments with correct action verb (with synonym mapping) |
| Noun Accuracy | % of segments with correct object noun |
| Action Accuracy | % where both verb AND noun are correct |
| Temporal IoU | Overlap between predicted and GT time ranges |
| Segment Precision/Recall | At IoU thresholds of 0.3 and 0.5 |
| Edit Distance | Operations to convert predicted sequence to GT |
| Labeling Speedup | Pipeline time vs estimated manual annotation time |

## Key Insight

> "Before robots can navigate the human world, they must first understand what the world means."

Instead of labeling objects and guessing states, we watch what humans **do** -- their actions, interactions, and intent -- and translate that into structured programs that robots can execute. Every kitchen video becomes a training environment for humanoid robots.
