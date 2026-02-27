import cv2
from dataclasses import dataclass
from typing import List, Tuple
import os

@dataclass
class FrameItem:
    timestamp_s: float
    bgr: any  # numpy array

def extract_frames_with_timestamps(
    video_path: str,
    stride_seconds: float = 1.0,
    max_frames: int = 120
) -> List[FrameItem]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    stride_frames = max(1, int(round(stride_seconds * fps)))
    frames: List[FrameItem] = []

    idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        if idx % stride_frames == 0:
            timestamp_s = idx / fps
            frames.append(FrameItem(timestamp_s=timestamp_s, bgr=frame))
        idx += 1

    cap.release()
    return frames
