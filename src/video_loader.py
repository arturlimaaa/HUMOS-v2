"""
Video frame extraction using OpenCV.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float


@dataclass
class Frame:
    image: np.ndarray       # BGR image
    frame_idx: int          # Original frame index in the video
    timestamp_sec: float    # Timestamp in seconds


class VideoLoader:
    """Loads video and extracts frames at a configurable sample rate."""

    def __init__(self, video_path: str, sample_rate: int = 5, max_frames: int | None = None):
        self.video_path = str(video_path)
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self._cap = None
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        meta = VideoMetadata(
            path=self.video_path,
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            duration_sec=cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        )
        cap.release()
        return meta

    def frames(self):
        """Generator yielding sampled Frame objects."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        frame_count = 0
        yielded = 0
        fps = self.metadata.fps

        while True:
            ret, image = cap.read()
            if not ret:
                break

            if frame_count % self.sample_rate == 0:
                yield Frame(
                    image=image,
                    frame_idx=frame_count,
                    timestamp_sec=frame_count / max(fps, 1),
                )
                yielded += 1
                if self.max_frames and yielded >= self.max_frames:
                    break

            frame_count += 1

        cap.release()

    def get_frame_at(self, frame_idx: int) -> Frame | None:
        """Get a specific frame by index."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        cap.release()
        if not ret:
            return None
        return Frame(
            image=image,
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / max(self.metadata.fps, 1),
        )

    def __repr__(self):
        m = self.metadata
        return (
            f"VideoLoader({Path(m.path).name}, "
            f"{m.width}x{m.height}, {m.fps:.1f}fps, "
            f"{m.duration_sec:.1f}s, {m.total_frames} frames, "
            f"sample_rate={self.sample_rate})"
        )
