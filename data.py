import cv2
import json
import jax
import numpy as np
from pathlib import Path
from jax import numpy as jnp
from typing import Iterator


def load_video_frames(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    """load video file, returns (T, H, W, C) uint8"""
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
    """load frames from image folder, sorted by name"""
    path = Path(path)
    extensions = {".png", ".jpg", ".jpeg"}
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
    """(T, H, W, C) uint8 -> (T, C, H, W) float32 normalized to [-1, 1]"""

    if target_size:
        h, w = target_size
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
        frames = np.stack(resized)

    frames = np.transpose(frames, (0, 3, 1, 2))  # THWC -> TCHW

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
        self.start_indices = list(
            range(0, self.num_frames - (sequence_length - 1) * stride, stride)
        )

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
    """combines multiple video sources"""

    def __init__(
        self,
        paths: list[str | Path],
        frame_size: tuple[int, int],
        sequence_length: int,
        stride: int = 1,
    ):
        self.datasets = [
            VideoDataset(p, frame_size, sequence_length, stride) for p in paths
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


class VPTDataset:
    """lazy-loading dataset for VPT minecraft videos with LRU cache"""

    def __init__(
        self,
        data_dir: str | Path,
        frame_size: tuple[int, int],
        sequence_length: int,
        stride: int = 1,
        cache_size: int = 10,
        max_videos: int | None = None,
        shuffle_videos: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.cache_size = cache_size

        # Find all video files
        self.video_paths = sorted(self.data_dir.glob("*.mp4"))

        if not self.video_paths:
            raise ValueError(f"No .mp4 files found in {data_dir}")

        if max_videos:
            self.video_paths = self.video_paths[:max_videos]

        print(f"Found {len(self.video_paths)} videos in {data_dir}")

        # Build index: (video_idx, start_frame) for all valid sequences
        self.index = self._build_index()
        print(f"Total sequences: {len(self.index)}")

        # LRU cache for loaded videos: {video_idx: frames}
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []

        # Shuffled order for iteration
        self._shuffled_indices: list[int] | None = None
        self.shuffle_videos = shuffle_videos

    def _get_video_frame_count(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    def _build_index(self) -> list[tuple[int, int]]:
        """build index of all valid (video_idx, start_frame) pairs"""
        index = []

        for video_idx, video_path in enumerate(self.video_paths):
            frame_count = self._get_video_frame_count(video_path)

            # be conservative - video metadata can be unreliable
            # need: start + (seq_len - 1) * stride < frame_count
            seq_frames = (self.sequence_length - 1) * self.stride + 1
            max_start = frame_count - seq_frames - 5  # 5 frame safety margin

            if max_start > 0:
                for start in range(0, max_start, self.sequence_length):
                    index.append((video_idx, start))

        return index

    def _load_video(self, video_idx: int) -> np.ndarray:
        """load video with LRU cache - helps when sampling multiple sequences from same video"""
        if video_idx in self._cache:
            self._cache_order.remove(video_idx)
            self._cache_order.append(video_idx)
            return self._cache[video_idx]

        video_path = self.video_paths[video_idx]
        raw_frames = load_video_frames(video_path)
        frames = preprocess_frames(raw_frames, target_size=self.frame_size)

        self._cache[video_idx] = frames
        self._cache_order.append(video_idx)

        while len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return frames

    def __len__(self) -> int:
        return len(self.index)

    def get_sequence(self, idx: int) -> np.ndarray:
        """get a single sequence by index, returns (T, C, H, W)"""
        video_idx, start_frame = self.index[idx]
        frames = self._load_video(video_idx)

        indices = [start_frame + i * self.stride for i in range(self.sequence_length)]
        max_idx = max(indices)

        if max_idx >= len(frames):
            raise IndexError(
                f"sequence index {max_idx} out of bounds for video with {len(frames)} frames"
            )

        return frames[indices]

    def get_batch(self, indices: list[int]) -> np.ndarray:
        return np.stack([self.get_sequence(i) for i in indices])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(self), size=min(batch_size, len(self)), replace=False)
        return self.get_batch(indices.tolist())

    def iter_batches(
        self,
        batch_size: int,
        rng: np.random.Generator,
        num_batches: int | None = None,
    ) -> Iterator[np.ndarray]:
        """iterate in order - more cache-friendly than random sampling"""
        indices = list(range(len(self)))

        if self.shuffle_videos:
            rng.shuffle(indices)

        batch_count = 0
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            if len(batch_indices) < batch_size:
                continue  # Skip incomplete final batch

            yield self.get_batch(batch_indices)

            batch_count += 1
            if num_batches and batch_count >= num_batches:
                break

    def get_video_info(self) -> dict:
        return {
            "num_videos": len(self.video_paths),
            "num_sequences": len(self.index),
            "frame_size": self.frame_size,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "cache_size": self.cache_size,
        }


class VPTStreamingDataset:
    """streams videos one at a time - for datasets too large to index"""

    def __init__(
        self,
        data_dir: str | Path,
        frame_size: tuple[int, int],
        sequence_length: int,
        stride: int = 1,
        sequences_per_video: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences_per_video = sequences_per_video

        self.video_paths = sorted(self.data_dir.glob("*.mp4"))

        if not self.video_paths:
            raise ValueError(f"No .mp4 files found in {data_dir}")

        print(f"Streaming dataset: {len(self.video_paths)} videos")

    def __len__(self) -> int:
        return len(self.video_paths) * self.sequences_per_video  # approximate

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        batches = list(self.iter_batches(batch_size, rng, num_batches=1))
        if batches:
            return batches[0]
        raise RuntimeError("could not sample batch")

    def iter_sequences(
        self,
        rng: np.random.Generator,
        shuffle: bool = True,
    ) -> Iterator[np.ndarray]:
        video_order = list(range(len(self.video_paths)))

        if shuffle:
            rng.shuffle(video_order)

        for video_idx in video_order:
            video_path = self.video_paths[video_idx]

            try:
                raw_frames = load_video_frames(video_path)
                frames = preprocess_frames(raw_frames, target_size=self.frame_size)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue

            num_frames = len(frames)
            max_start = num_frames - (self.sequence_length - 1) * self.stride - 1

            if max_start <= 0:
                continue

            # Sample random start positions from this video
            starts = rng.choice(
                max_start, size=min(self.sequences_per_video, max_start), replace=False
            )

            for start in starts:
                indices = [start + i * self.stride for i in range(self.sequence_length)]
                yield frames[indices]

    def iter_batches(
        self,
        batch_size: int,
        rng: np.random.Generator,
        num_batches: int | None = None,
        shuffle: bool = True,
    ) -> Iterator[np.ndarray]:
        batch = []
        batch_count = 0

        for seq in self.iter_sequences(rng, shuffle=shuffle):
            batch.append(seq)

            if len(batch) >= batch_size:
                yield np.stack(batch)
                batch = []
                batch_count += 1

                if num_batches and batch_count >= num_batches:
                    return


def create_dataset(
    data_path: str | Path,
    frame_size: tuple[int, int],
    sequence_length: int,
    stride: int = 1,
    dataset_type: str = "auto",
    **kwargs,
):
    """factory for dataset type - auto-detects from path"""
    path = Path(data_path)

    if dataset_type == "auto":
        if path.is_file():
            dataset_type = "video"
        elif (path / "manifest.json").exists() or list(path.glob("*.mp4")):
            # VPT directory detected
            dataset_type = "vpt"
        else:
            dataset_type = "frames"

    if dataset_type == "video":
        return VideoDataset(path, frame_size, sequence_length, stride)
    elif dataset_type == "frames":
        return VideoDataset(path, frame_size, sequence_length, stride)
    elif dataset_type == "vpt":
        return VPTDataset(path, frame_size, sequence_length, stride, **kwargs)
    elif dataset_type == "vpt_streaming":
        return VPTStreamingDataset(path, frame_size, sequence_length, stride, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
