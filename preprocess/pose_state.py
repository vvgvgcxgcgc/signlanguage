"""Start-pose vs action-pose windows cut from raw (T, 76, 3) clips.

One sample is a causal window of three consecutive frames `t-2, t-1, t` of
body pose xy: pose 0-24 plus the neck at 33. Legs and the 21-point hands are
dropped. Raw pose has no holes, so there is no validity channel.

Labels come from frame position, checked against wrist height:

- start: frames 0..4 and the last two frames, only for clips longer than 30
  frames, and only when the highest wrist is near the hips.
- action: every third frame in [0.3T, 0.7T] whose highest wrist is raised.

Transition frames in between get no label.

    python -m preprocess.pose_state --root vsl400-keypoint --out vsl400-pose-state
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

RAW_JOINTS = 76
# Pose 0-24 (face, arms, pose finger stubs, hips) plus the neck at 33.
POSE_JOINTS = np.array(list(range(25)) + [33], dtype=np.int64)
NUM_JOINTS = int(POSE_JOINTS.shape[0])
NUM_COORDS = 2
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
# Positions inside POSE_JOINTS. They equal the raw indices because 0-24 come first.
MIRROR_PAIRS = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
    (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
)

WINDOW = 3
MIN_FRAMES = 20
# Clips of exactly 30 frames are pre-trimmed and start mid-sign.
REST_MIN_FRAMES = 31
REST_HEAD = 5
REST_TAIL = 2
ACTION_SPAN = (0.3, 0.7)
ACTION_STRIDE = 3
REST_WRIST_MIN = 0.8
ACTION_WRIST_MAX = 0.6
MIN_SHOULDER = 1e-3

LABEL_NAMES = ("start", "action")
START, ACTION = 0, 1
SPLITS = ("train", "test")


def window_indices(t: int, window: int = WINDOW) -> np.ndarray:
    """Frames t-window+1 .. t, clamped at 0 so the first frames repeat."""
    return np.clip(np.arange(t - window + 1, t + 1), 0, None)


def normalize_window(frames: np.ndarray) -> Optional[np.ndarray]:
    """Map raw (W, 76, 3) frames to (W, J, 2) body xy in the last frame's body units.

    Origin is the last frame's shoulder midpoint and scale is its shoulder
    width. The same transform applies to every frame of the window, so the
    differences between frames are motion in body units. Returns None when
    the shoulders are degenerate.
    """
    pose = frames[:, POSE_JOINTS, :NUM_COORDS].astype(np.float32)
    last = frames[-1]
    origin = (last[LEFT_SHOULDER, :2] + last[RIGHT_SHOULDER, :2]) * 0.5
    scale = float(np.linalg.norm(last[LEFT_SHOULDER, :2] - last[RIGHT_SHOULDER, :2]))
    if not np.isfinite(scale) or scale <= MIN_SHOULDER:
        return None
    return ((pose - origin) / np.float32(scale)).astype(np.float32)


def clip_windows(clip: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Causal window for every frame of a raw clip, as a stream would see it.

    Returns `(windows, valid)` with shapes (T, W, J, 2) and (T,). Frames with
    degenerate shoulders are zero and marked invalid.
    """
    frames = int(clip.shape[0])
    windows = np.zeros((frames, window, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    valid = np.zeros(frames, dtype=bool)
    for t in range(frames):
        normalized = normalize_window(clip[window_indices(t, window)])
        if normalized is not None:
            windows[t] = normalized
            valid[t] = True
    return windows, valid


def wrist_height(clip: np.ndarray) -> np.ndarray:
    """Highest wrist per frame: 0 at shoulder level, 1 at hip level. Image y points down."""
    shoulder = clip[:, [LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean(axis=1)
    hip = clip[:, [LEFT_HIP, RIGHT_HIP], 1].mean(axis=1)
    wrist = clip[:, [LEFT_WRIST, RIGHT_WRIST], 1].min(axis=1)
    return (wrist - shoulder) / np.maximum(np.abs(hip - shoulder), 1e-6)


def label_frames(clip: np.ndarray) -> list[tuple[int, int]]:
    """(frame, label) pairs for one clip. Transition frames get no label."""
    frames = int(clip.shape[0])
    if frames < MIN_FRAMES:
        return []
    height = wrist_height(clip)
    picked: list[tuple[int, int]] = []
    if frames >= REST_MIN_FRAMES:
        rest = list(range(REST_HEAD)) + list(range(frames - REST_TAIL, frames))
        picked += [(t, START) for t in rest if height[t] > REST_WRIST_MIN]
    low = int(math.ceil(ACTION_SPAN[0] * frames))
    high = int(ACTION_SPAN[1] * frames)
    picked += [
        (t, ACTION) for t in range(low, high + 1, ACTION_STRIDE) if height[t] < ACTION_WRIST_MAX
    ]
    return picked


def load_raw(path: str | Path) -> Optional[np.ndarray]:
    try:
        clip = np.load(path)
    except (OSError, ValueError):
        return None
    if clip.ndim != 3 or clip.shape[1:] != (RAW_JOINTS, 3) or clip.shape[0] == 0:
        return None
    return clip


def _extract_clip(path: str):
    clip = load_raw(path)
    if clip is None:
        return path, None
    windows, labels, frames = [], [], []
    for t, label in label_frames(clip):
        window = normalize_window(clip[window_indices(t)])
        if window is None:
            continue
        windows.append(window)
        labels.append(label)
        frames.append(t)
    if not windows:
        return path, None
    return path, (
        np.stack(windows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(frames, dtype=np.int32),
        int(clip.shape[0]),
    )


def build_split(root: Path, split: str, workers: Optional[int] = None) -> dict[str, np.ndarray]:
    """Label every clip of one split in a process pool and stack the windows.

    `clip_ids` indexes `paths` and `clip_lengths`, so windows can be grouped
    back to their clip for a split that never puts one clip on both sides.
    """
    paths = sorted(str(path) for path in (Path(root) / split).glob("*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy clips under {Path(root) / split}")
    worker_count = max(1, min(workers or os.cpu_count() or 1, len(paths)))
    chunksize = max(1, len(paths) // (worker_count * 8))
    with Pool(processes=worker_count) as pool:
        results = list(
            tqdm(pool.imap(_extract_clip, paths, chunksize=chunksize), total=len(paths), desc=split, unit="clip")
        )

    windows, labels, clip_ids, frames, lengths, kept_paths = [], [], [], [], [], []
    for path, item in results:
        if item is None:
            continue
        clip_window, clip_label, clip_frame, clip_length = item
        clip_id = len(kept_paths)
        kept_paths.append(path)
        windows.append(clip_window)
        labels.append(clip_label)
        frames.append(clip_frame)
        clip_ids.append(np.full(len(clip_label), clip_id, dtype=np.int32))
        lengths.append(clip_length)
    if not kept_paths:
        raise RuntimeError(f"no clip in {split} produced a labelled window")
    data = {
        "windows": np.concatenate(windows),
        "labels": np.concatenate(labels),
        "clip_ids": np.concatenate(clip_ids),
        "frames": np.concatenate(frames),
        "clip_lengths": np.asarray(lengths, dtype=np.int32),
        "paths": np.asarray(kept_paths),
    }
    counts = np.bincount(data["labels"], minlength=len(LABEL_NAMES))
    logger.info(
        "%s: %d/%d clips kept, %d windows (%s), shape %s",
        split,
        len(kept_paths),
        len(paths),
        len(data["labels"]),
        ", ".join(f"{name}={count}" for name, count in zip(LABEL_NAMES, counts)),
        tuple(data["windows"].shape),
    )
    return data


def save_split(out_dir: Path, split: str, data: dict[str, np.ndarray]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.npz"
    np.savez(path, **data)
    logger.info("Saved %s", path)
    return path


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = Path(data_dir) / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing, run python -m preprocess.pose_state first")
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def holdout_by_clip(data: dict[str, np.ndarray], fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Window indices for (train, holdout), with every clip on exactly one side."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    clip_count = int(data["clip_lengths"].shape[0])
    rng = np.random.default_rng(seed)
    held = rng.permutation(clip_count)[: max(1, int(round(clip_count * fraction)))]
    in_holdout = np.isin(data["clip_ids"], held)
    return np.flatnonzero(~in_holdout), np.flatnonzero(in_holdout)


class PoseStateDataset(Dataset):
    """Windows of shape (W, J, 2) with a start/action label, held in RAM.

    Training draws one mirror, scale, rotation, and shift for the whole window
    so motion between its frames stays coherent, then adds per-joint noise.
    The draw depends on (seed, epoch, index) and changes with `set_epoch`.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        train: bool,
        seed: int = 0,
        flip_prob: float = 0.5,
        scale_range: tuple[float, float] = (0.85, 1.15),
        rotate_deg: float = 12.0,
        shift_std: float = 0.05,
        noise_std: float = 0.01,
    ) -> None:
        if windows.ndim != 4 or windows.shape[2:] != (NUM_JOINTS, NUM_COORDS):
            raise ValueError(f"expected windows of shape (N, W, {NUM_JOINTS}, {NUM_COORDS}), got {windows.shape}")
        if len(windows) != len(labels):
            raise ValueError(f"{len(windows)} windows but {len(labels)} labels")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"scale_range must be positive and ordered, got {scale_range}")
        self.windows = np.ascontiguousarray(windows, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = bool(train)
        self.seed = int(seed)
        self.flip_prob = float(flip_prob)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.rotate_deg = float(rotate_deg)
        self.shift_std = float(shift_std)
        self.noise_std = float(noise_std)
        self.epoch = 0
        mirror = np.arange(NUM_JOINTS)
        for left, right in MIRROR_PAIRS:
            mirror[left], mirror[right] = right, left
        self.mirror = mirror

    @property
    def window(self) -> int:
        return int(self.windows.shape[1])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(LABEL_NAMES))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        window = self.windows[index]
        label = int(self.labels[index])
        if not self.train:
            return torch.from_numpy(window), label
        rng = np.random.default_rng([self.seed, self.epoch, int(index)])
        points = window.copy()
        if rng.random() < self.flip_prob:
            points = points[:, self.mirror]
            points[..., 0] = -points[..., 0]
        scale = rng.uniform(*self.scale_range)
        theta = np.deg2rad(rng.uniform(-self.rotate_deg, self.rotate_deg))
        cosine, sine = np.cos(theta), np.sin(theta)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32) * np.float32(scale)
        shift = rng.normal(0.0, self.shift_std, size=NUM_COORDS)
        noise = rng.normal(0.0, self.noise_std, size=points.shape)
        points = points @ rotation.T + shift + noise
        return torch.from_numpy(points.astype(np.float32)), label


def make_pose_loaders(
    data_dir: Path,
    batch_size: int = 512,
    val_fraction: float = 0.1,
    seed: int = 0,
    **augment,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train, val, and test loaders. Val is a by-clip holdout of the train split.

    Data already sits in RAM and the model is tiny, so workers would cost more
    than they save.
    """
    train_data = load_split(data_dir, "train")
    test_data = load_split(data_dir, "test")
    train_index, val_index = holdout_by_clip(train_data, val_fraction, seed)

    train_set = PoseStateDataset(
        train_data["windows"][train_index], train_data["labels"][train_index], train=True, seed=seed, **augment
    )
    val_set = PoseStateDataset(train_data["windows"][val_index], train_data["labels"][val_index], train=False)
    test_set = PoseStateDataset(test_data["windows"], test_data["labels"], train=False)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    for name, dataset in (("train", train_set), ("val", val_set), ("test", test_set)):
        counts = dataset.class_counts()
        logger.info(
            "%s windows %d (%s)",
            name,
            len(dataset),
            ", ".join(f"{label}={count}" for label, count in zip(LABEL_NAMES, counts)),
        )
    return train_loader, val_loader, test_loader


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="preprocess.pose_state", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("vsl400-keypoint"), help="holds train/ and test/")
    parser.add_argument("--out", type=Path, default=Path("vsl400-pose-state"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for split in SPLITS:
        save_split(args.out, split, build_split(args.root, split, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
