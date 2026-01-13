import cv2
import jax
import numpy as np
from pathlib import Path
from jax import numpy as jnp


def load_video_frames(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    # load video file and return frames as (T, H, W, C) uint8
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return np.stack(frames)


def load_frame_folder(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    # load frames from folder of images (sorted by name)
    path = Path(path)
    extensions = {'.png', '.jpg', '.jpeg'}
    files = sorted([f for f in path.iterdir() if f.suffix.lower() in extensions])

    if max_frames:
        files = files[:max_frames]

    frames = []
    for f in files:
        img = cv2.imread(str(f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.stack(frames)


def preprocess_frames(
    frames: np.ndarray,
    target_size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    # frames: (T, H, W, C) uint8
    # returns: (T, C, H, W) float32 normalized to [-1, 1]

    if target_size:
        h, w = target_size
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
        frames = np.stack(resized)

    # (T, H, W, C) -> (T, C, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))

    if normalize:
        frames = frames.astype(np.float32) / 127.5 - 1.0

    return frames


class VideoDataset:
    def __init__(
        self,
        path: str | Path,
        frame_size: tuple[int, int],
        sequence_length: int,
        stride: int = 1,
    ):
        self.path = Path(path)
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.stride = stride

        # load all frames
        if self.path.is_file():
            raw_frames = load_video_frames(self.path)
        else:
            raw_frames = load_frame_folder(self.path)

        self.frames = preprocess_frames(raw_frames, target_size=frame_size)
        self.num_frames = len(self.frames)

        # compute valid start indices for sequences
        self.start_indices = list(range(
            0,
            self.num_frames - (sequence_length - 1) * stride,
            stride
        ))

    def __len__(self) -> int:
        return len(self.start_indices)

    def get_sequence(self, idx: int) -> np.ndarray:
        # returns (sequence_length, C, H, W)
        start = self.start_indices[idx]
        indices = [start + i * self.stride for i in range(self.sequence_length)]
        return self.frames[indices]

    def get_batch(self, indices: list[int]) -> np.ndarray:
        # returns (B, T, C, H, W)
        return np.stack([self.get_sequence(i) for i in indices])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(self), size=batch_size, replace=False)
        return self.get_batch(indices.tolist())


class MultiVideoDataset:
    # combines multiple video sources
    def __init__(
        self,
        paths: list[str | Path],
        frame_size: tuple[int, int],
        sequence_length: int,
        stride: int = 1,
    ):
        self.datasets = [
            VideoDataset(p, frame_size, sequence_length, stride)
            for p in paths
        ]

        # build global index mapping
        self.index_map = []  # (dataset_idx, local_idx)
        for ds_idx, ds in enumerate(self.datasets):
            for local_idx in range(len(ds)):
                self.index_map.append((ds_idx, local_idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def get_sequence(self, idx: int) -> np.ndarray:
        ds_idx, local_idx = self.index_map[idx]
        return self.datasets[ds_idx].get_sequence(local_idx)

    def get_batch(self, indices: list[int]) -> np.ndarray:
        return np.stack([self.get_sequence(i) for i in indices])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(self), size=min(batch_size, len(self)), replace=False)
        return self.get_batch(indices.tolist())
