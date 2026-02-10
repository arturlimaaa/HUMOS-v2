"""
Video overlay visualization for action recognition.
Draws action labels, timelines, and segment overlays on video frames.
"""
import cv2
import numpy as np

import config


def draw_action_label(image: np.ndarray, verb: str, noun: str,
                      confidence: float, phase: str = "") -> np.ndarray:
    """Draw a large action label banner on the frame."""
    overlay = image.copy()
    h, w = overlay.shape[:2]

    # Banner background
    banner_h = 70
    color = config.VERB_COLORS.get(verb, config.DEFAULT_VERB_COLOR)
    # Semi-transparent background
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = color
    alpha = 0.7
    overlay[:banner_h] = cv2.addWeighted(overlay[:banner_h], 1 - alpha, banner, alpha, 0)

    # Action text
    action_text = f"{verb.upper()} {noun}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, action_text, (15, 45), font, 1.2, (255, 255, 255), 3)

    # Confidence + phase
    info_text = f"{confidence:.0%}"
    if phase and phase != "idle":
        info_text += f" | {phase}"
    cv2.putText(overlay, info_text, (w - 200, 45), font, 0.7, (220, 220, 220), 2)

    return overlay


def draw_action_timeline(
    image: np.ndarray,
    segments: list,
    current_sec: float,
    total_duration: float,
    bar_height: int = 40,
) -> np.ndarray:
    """Draw a horizontal timeline bar showing action segments below the frame."""
    h, w = image.shape[:2]
    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    for seg in segments:
        x_start = int((seg.start_sec / max(total_duration, 0.1)) * w)
        x_end = int((seg.stop_sec / max(total_duration, 0.1)) * w)
        color = config.VERB_COLORS.get(seg.verb, config.DEFAULT_VERB_COLOR)
        cv2.rectangle(bar, (x_start, 2), (x_end, bar_height - 2), color, -1)

        # Label if wide enough
        seg_w = x_end - x_start
        if seg_w > 50:
            label = f"{seg.verb}"
            cv2.putText(bar, label, (x_start + 4, bar_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Current position indicator
    x_now = int((current_sec / max(total_duration, 0.1)) * w)
    cv2.line(bar, (x_now, 0), (x_now, bar_height), (255, 255, 255), 2)

    return np.vstack([image, bar])


class VideoWriter:
    """Write annotated frames to an output video."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = str(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        self.width = width
        self.height = height

    def write(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
