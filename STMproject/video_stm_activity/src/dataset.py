import os
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class VideoSample:
    video_path: str
    label: str
    video_name: str

def load_dataset(videos_dir: str, labels_csv: str) -> List[VideoSample]:
    df = pd.read_csv(labels_csv)
    samples: List[VideoSample] = []
    for _, row in df.iterrows():
        video = str(row["video"])
        label = str(row["label"]).strip().lower()
        video_path = os.path.join(videos_dir, video)
        samples.append(VideoSample(video_path=video_path, label=label, video_name=video))
    return samples
