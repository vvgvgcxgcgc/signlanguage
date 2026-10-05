"""Train and test dataset for raw MediaPipe clips of shape (T, 76, 3).

Shoulder width is one scale per clip, so a signer close to the camera and a
signer far away land in the same frame. Each joint is xyz plus a validity
channel: a missing joint stays at the shoulder-midpoint origin, and validity
is what separates that hole from a joint that really sits there. Temporal
branches are resampled with interpolation that never blends a joint across a
hole, so xyz always carries an observed value. Training draws one random
temporal crop of the clip, resamples every branch from that crop, then applies
scale, xy rotation, and noise.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

logger = logging.getLogger(__name__)

N_JOINTS = 76
N_COORDS = 3
N_CHANNELS = 4
BODY_COUNT = 34
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
MIN_VALID_FRAMES = 3
LEG_INDICES = (25, 26, 27, 28, 29, 30, 31, 32)
_KEEP_JOINTS = np.array([index for index in range(N_JOINTS) if index not in LEG_INDICES], dtype=np.int64)


@dataclass(frozen=True)
class TemporalMode:
    """One interpolated temporal branch. `length` is the output frame count."""

    name: str
    length: int


MODE_MAIN = TemporalMode("main", 64)
MODE_SLOW = TemporalMode("slow", 8)
MODE_FAST = TemporalMode("fast", 32)
SINGLE_MODES = (MODE_MAIN,)
SLOWFAST_MODES = (MODE_SLOW, MODE_FAST)


def normalize_clip(array: np.ndarray, min_shoulder: float) -> Optional[np.ndarray]:
    """Map one raw clip into a shoulder-centered frame.

    Frames with a missing shoulder, or with a 2D shoulder width at or below
    `min_shoulder`, are removed. The remaining clip is dropped when fewer than
    three frames survive or the median shoulder width is still too small.
    Body xyz, including the neck, is shifted by that frame's shoulder midpoint
    and divided by the clip median. Hand xy uses the same origin and scale.
    Hand z is already wrist-relative, so it is only divided by the scale.
    The last channel is 1 when the raw joint was not (0, 0, 0) and 0 when it
    was. Missing xyz is written back to the origin after the shift, so a hole
    and a joint that lands on the shoulder midpoint share xyz and differ only
    in that channel. Returned shape is (T, 76, 4).
    """
    clip = np.asarray(array, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1:] != (N_JOINTS, 3) or clip.shape[0] == 0:
        return None
    left = clip[:, LEFT_SHOULDER]
    right = clip[:, RIGHT_SHOULDER]
    shoulders_present = ~(np.all(left == 0, axis=-1) | np.all(right == 0, axis=-1))
    distance = np.linalg.norm(left[:, :2] - right[:, :2], axis=-1)
    valid = shoulders_present & (distance > min_shoulder)
    if int(np.count_nonzero(valid)) < MIN_VALID_FRAMES:
        return None
    scale = float(np.median(distance[valid]))
    if not np.isfinite(scale) or scale < min_shoulder:
        return None

    kept = np.array(clip[valid], dtype=np.float32, copy=True)
    origin = (kept[:, LEFT_SHOULDER] + kept[:, RIGHT_SHOULDER]) * np.float32(0.5)
    scale_value = np.float32(scale)

    body = kept[:, :BODY_COUNT]
    body_missing = np.all(body == 0, axis=-1)
    body_out = (body - origin[:, None, :]) / scale_value
    body_out[body_missing] = 0

    hands = kept[:, BODY_COUNT:]
    hand_missing = np.all(hands == 0, axis=-1)
    hands_out = np.empty_like(hands)
    hands_out[..., 0] = (hands[..., 0] - origin[:, None, 0]) / scale_value
    hands_out[..., 1] = (hands[..., 1] - origin[:, None, 1]) / scale_value
    hands_out[..., 2] = hands[..., 2] / scale_value
    hands_out[hand_missing] = 0
    body_valid = (~body_missing).astype(np.float32)[..., None]
    hand_valid = (~hand_missing).astype(np.float32)[..., None]
    body_out = np.concatenate((body_out, body_valid), axis=-1)
    hands_out = np.concatenate((hands_out, hand_valid), axis=-1)
    return np.ascontiguousarray(np.concatenate((body_out, hands_out), axis=1), dtype=np.float32)


def drop_legs(clip: np.ndarray) -> np.ndarray:
    """Remove knee, ankle, heel, and foot joints. Hips stay."""
    return np.ascontiguousarray(clip[:, _KEEP_JOINTS, :])


def interpolate_clip(clip: np.ndarray, length: int) -> np.ndarray:
    """Sample `length` phases evenly from the first frame to the last.

    Phase i sits at t = i * (N - 1) / (length - 1). When both neighbouring
    frames observed a joint, its xyz is the linear mix of the two. When only
    one of them observed it, xyz is that frame's value, so the joint never
    drifts toward the origin and no motion is invented out of a hole. When
    neither observed it, xyz is the origin. Validity is the linear mix in
    every case: 1 inside an observed span, 0 inside a hole, and in between at
    a boundary, which records how far the xyz was carried. A one-frame clip
    repeats that frame. A length of 1 returns the first frame.
    """
    if length < 1:
        raise ValueError(f"temporal length must be positive, got {length}")
    frames = int(clip.shape[0])
    if frames == 0:
        raise ValueError("cannot interpolate an empty clip")
    if length == 1:
        return np.ascontiguousarray(clip[:1], dtype=np.float32)
    if frames == 1:
        return np.repeat(np.ascontiguousarray(clip[:1], dtype=np.float32), length, axis=0)

    positions = np.arange(length, dtype=np.float64) * (frames - 1) / (length - 1)
    low = np.floor(positions).astype(np.int64)
    high = np.ceil(positions).astype(np.int64)
    weight = (positions - low).astype(np.float32)[:, None, None]

    low_frames = clip[low]
    high_frames = clip[high]
    mixed = (1.0 - weight) * low_frames + weight * high_frames
    low_seen = low_frames[..., N_COORDS:] > 0
    high_seen = high_frames[..., N_COORDS:] > 0

    xyz = mixed[..., :N_COORDS]
    xyz = np.where(low_seen & ~high_seen, low_frames[..., :N_COORDS], xyz)
    xyz = np.where(high_seen & ~low_seen, high_frames[..., :N_COORDS], xyz)
    xyz = np.where(low_seen | high_seen, xyz, np.float32(0.0))
    return np.ascontiguousarray(
        np.concatenate((xyz, mixed[..., N_COORDS:]), axis=-1), dtype=np.float32
    )


def _check_modes(modes: Sequence[TemporalMode]) -> tuple[TemporalMode, ...]:
    checked = tuple(modes)
    if len(checked) not in (1, 2, 3):
        raise ValueError(f"expected 1, 2, or 3 temporal modes, got {len(checked)}")
    for mode in checked:
        if not isinstance(mode, TemporalMode):
            raise TypeError(f"modes must be TemporalMode, got {type(mode).__name__}")
        if not mode.name:
            raise ValueError("temporal mode name must not be empty")
        if mode.length < 1:
            raise ValueError(f"temporal length must be positive, got {mode.length}")
    return checked


def _temporal_crop(
    clip: np.ndarray,
    rng: np.random.Generator,
    crop_range: tuple[float, float],
) -> np.ndarray:
    """Slice one contiguous span covering a random fraction of `clip`.

    The fraction is drawn from `crop_range`. Resampling that span to a fixed
    length is left to `interpolate_clip`. A span that already covers every
    frame returns the clip. A one-frame clip is unchanged.
    """
    frames = int(clip.shape[0])
    if frames <= 1:
        return clip
    fraction = float(rng.uniform(crop_range[0], crop_range[1]))
    span = int(np.clip(np.round(fraction * frames), 1, frames))
    if span >= frames:
        return clip
    start = int(rng.integers(0, frames - span + 1))
    return np.ascontiguousarray(clip[start:start + span])


def _augment_branches(
    branches: Sequence[np.ndarray],
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    rotate_deg: float,
    noise_std: float,
) -> tuple[np.ndarray, ...]:
    """Apply one scale and one xy rotation to xyz on every branch, then add noise.

    Validity is not scaled, rotated, or noised. Validity 0 is the only case
    where xyz is a placeholder, so those joints are written back to the origin
    after the noise. Any validity above 0 means xyz came from a real
    observation and keeps the noise, including a joint that genuinely sits on
    the origin. There is no left-right flip.
    """
    scale = np.float32(rng.uniform(scale_range[0], scale_range[1]))
    theta = np.deg2rad(float(rng.uniform(-rotate_deg, rotate_deg)))
    cosine = np.float32(np.cos(theta))
    sine = np.float32(np.sin(theta))
    augmented: list[np.ndarray] = []
    for branch in branches:
        xyz = branch[..., :N_COORDS]
        valid = branch[..., N_COORDS]
        missing = valid == 0
        scaled_x = xyz[..., 0] * scale
        scaled_y = xyz[..., 1] * scale
        scaled_z = xyz[..., 2] * scale
        rotated_x = cosine * scaled_x - sine * scaled_y
        rotated_y = sine * scaled_x + cosine * scaled_y
        transformed = np.stack((rotated_x, rotated_y, scaled_z), axis=-1)
        noise = rng.normal(0.0, noise_std, size=transformed.shape).astype(np.float32)
        transformed = transformed + noise
        transformed[missing] = 0
        stacked = np.concatenate((transformed, valid[..., None]), axis=-1)
        augmented.append(np.ascontiguousarray(stacked, dtype=np.float32))
    return tuple(augmented)


def _sample_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), int(epoch), int(index)])
    return np.random.default_rng(sequence)


def _generator_seed(seed: int, epoch: int) -> int:
    return (int(seed) + 0x9E3779B97F4A7C15 * int(epoch)) & 0xFFFFFFFFFFFFFFFF


class KeypointDataset(Dataset):
    """Normalized keypoint clips. Each item is `(tuple of branch tensors, label)`.

    A branch tensor has shape `(T, J, 4)`: xyz, then validity.
    """

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        class_names: Sequence[str],
        modes: Sequence[TemporalMode],
        is_leg: bool,
        train: bool,
        seed: int,
        min_shoulder: float,
        scale_range: tuple[float, float] = (0.5, 1.5),
        rotate_deg: float = 15.0,
        noise_std: float = 0.01,
        crop_range: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        if min_shoulder <= 0:
            raise ValueError(f"min_shoulder must be positive, got {min_shoulder}")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"scale_range must be positive and ordered, got {scale_range}")
        if rotate_deg < 0:
            raise ValueError(f"rotate_deg must be non-negative, got {rotate_deg}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if not 0 < crop_range[0] <= crop_range[1] <= 1:
            raise ValueError(f"crop_range must lie in (0, 1] and be ordered, got {crop_range}")
        self.samples = [(Path(path), int(label)) for path, label in samples]
        self.class_names = tuple(class_names)
        self.modes = _check_modes(modes)
        self.is_leg = bool(is_leg)
        self.train = bool(train)
        self.seed = int(seed)
        self.min_shoulder = float(min_shoulder)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.rotate_deg = float(rotate_deg)
        self.noise_std = float(noise_std)
        self.crop_range = (float(crop_range[0]), float(crop_range[1]))
        self.epoch = 0

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    @property
    def num_joints(self) -> int:
        return N_JOINTS if self.is_leg else int(_KEEP_JOINTS.shape[0])

    @property
    def num_channels(self) -> int:
        return N_CHANNELS

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw for this epoch, including the temporal crop."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], int]:
        path, label = self.samples[index]
        normalized = normalize_clip(np.load(path), self.min_shoulder)
        if normalized is None:
            raise RuntimeError(f"clip failed shoulder normalization: {path}")
        if not self.is_leg:
            normalized = drop_legs(normalized)
        if self.train:
            rng = _sample_rng(self.seed, self.epoch, index)
            normalized = _temporal_crop(normalized, rng, self.crop_range)
        branches = tuple(interpolate_clip(normalized, mode.length) for mode in self.modes)
        if self.train:
            branches = _augment_branches(branches, rng, self.scale_range, self.rotate_deg, self.noise_std)
        tensors = tuple(torch.from_numpy(branch) for branch in branches)
        return tensors, label


class BalancedBatchSampler(Sampler[int]):
    """Yield indices so each batch contains labels as evenly as possible.

    When the batch is smaller than the number of classes, each picked class
    contributes one clip. Otherwise every class contributes `batch // C`
    clips and the remainder is spread one class at a time. The draw changes
    with `set_epoch` and repeats if the epoch is unchanged.
    """

    def __init__(self, labels: Sequence[int], batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(labels) == 0:
            raise ValueError("cannot balance an empty dataset")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self._by_class[int(label)].append(index)
        self._classes = sorted(self._by_class)
        self._num_batches = max(1, len(labels) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches * self.batch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(_generator_seed(self.seed, self.epoch))
        indices: list[int] = []
        for _ in range(self._num_batches):
            indices.extend(self._batch_indices(generator))
        return iter(indices)

    def _batch_indices(self, generator: torch.Generator) -> list[int]:
        class_count = len(self._classes)
        if self.batch_size <= class_count:
            order = torch.randperm(class_count, generator=generator)[: self.batch_size]
            quotas = {self._classes[int(choice)]: 1 for choice in order}
        else:
            base = self.batch_size // class_count
            extra = self.batch_size % class_count
            order = torch.randperm(class_count, generator=generator)
            bonus = {self._classes[int(order[offset])] for offset in range(extra)}
            quotas = {label: base + (1 if label in bonus else 0) for label in self._classes}
        picked: list[int] = []
        for label in self._classes:
            quota = quotas.get(label, 0)
            if quota == 0:
                continue
            picked.extend(self._draw(self._by_class[label], quota, generator))
        shuffle = torch.randperm(len(picked), generator=generator)
        return [picked[int(position)] for position in shuffle]

    @staticmethod
    def _draw(pool: Sequence[int], quota: int, generator: torch.Generator) -> list[int]:
        if quota <= len(pool):
            order = torch.randperm(len(pool), generator=generator)[:quota]
            return [pool[int(position)] for position in order]
        draws = torch.randint(0, len(pool), (quota,), generator=generator)
        return [pool[int(draw)] for draw in draws]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(
    dataset: KeypointDataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Train uses a balanced sampler. Test walks every clip once, in order.

    Call `dataset.set_epoch` and, for training, `loader.sampler.set_epoch`
    at the start of each epoch.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if dataset.train:
        sampler = BalancedBatchSampler(dataset.labels, batch_size, seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=_seed_worker,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def _validate_clip(job: tuple[str, float]) -> bool:
    path, min_shoulder = job
    try:
        array = np.load(path)
    except (OSError, ValueError):
        return False
    return normalize_clip(array, min_shoulder) is not None


def _valid_paths(paths: Sequence[Path], min_shoulder: float, workers: int, desc: str) -> list[Path]:
    if not paths:
        return []
    jobs = [(str(path), min_shoulder) for path in paths]
    worker_count = max(1, min(workers, len(jobs)))
    chunksize = max(1, len(jobs) // (worker_count * 4))
    logger.info("Checking shoulder scale for %d clips with %d workers", len(jobs), worker_count)
    with Pool(processes=worker_count) as pool:
        flags = list(
            tqdm(
                pool.imap(_validate_clip, jobs, chunksize=chunksize),
                total=len(jobs),
                desc=desc,
                unit="clip",
            )
        )
    kept = [path for path, ok in zip(paths, flags) if ok]
    rejected = len(paths) - len(kept)
    if rejected:
        logger.info("Rejected %d clips with a bad shoulder scale", rejected)
    return kept


def _npy_files(gloss_dir: Path) -> list[Path]:
    files = [path for path in gloss_dir.iterdir() if path.is_file() and path.suffix.lower() == ".npy"]
    files.sort(key=lambda path: path.name)
    return files


def _gloss_files(split_dir: Path) -> dict[str, list[Path]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split directory: {split_dir}")
    grouped: dict[str, list[Path]] = {}
    for entry in split_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        grouped[entry.name] = _npy_files(entry)
    return grouped


def _write_labels(path: Path, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labels": list(labels)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d labels to %s", len(labels), path)


def _samples_for(valid: dict[str, list[Path]], class_names: Sequence[str]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label, name in enumerate(class_names):
        for path in valid.get(name, []):
            samples.append((path, label))
    return samples


def build_datasets(
    root: Path | str,
    min_train_samples: int = 1,
    modes: Optional[Sequence[TemporalMode]] = None,
    is_leg: bool = False,
    seed: int = 0,
    min_shoulder: float = 1e-3,
    metadata_path: Optional[Path | str] = None,
    workers: Optional[int] = None,
    scale_range: tuple[float, float] = (0.5, 1.5),
    rotate_deg: float = 15.0,
    noise_std: float = 0.01,
    crop_range: tuple[float, float] = (0.5, 1.0),
) -> tuple[KeypointDataset, KeypointDataset, list[str]]:
    """Filter glosses on the train split, write `labels.json`, and build both datasets.

    A gloss stays only when its number of shoulder-valid train clips is at least
    `min_train_samples`. The JSON list is that filtered order, and index i in
    the file is label i for both splits. The test split does not write the file again.
    """
    if min_train_samples < 1:
        raise ValueError(f"min_train_samples must be positive, got {min_train_samples}")
    if min_shoulder <= 0:
        raise ValueError(f"min_shoulder must be positive, got {min_shoulder}")
    selected_modes = SINGLE_MODES if modes is None else _check_modes(modes)
    worker_count = os.cpu_count() or 1 if workers is None else max(1, int(workers))
    root_path = Path(root)
    train_groups = _gloss_files(root_path / "train")
    train_paths = [path for name in sorted(train_groups) for path in train_groups[name]]
    valid_train_paths = set(_valid_paths(train_paths, min_shoulder, worker_count, "train"))
    valid_train = {
        name: [path for path in paths if path in valid_train_paths]
        for name, paths in train_groups.items()
    }
    class_names = sorted(name for name, paths in valid_train.items() if len(paths) >= min_train_samples)
    if not class_names:
        raise RuntimeError(f"no gloss has at least {min_train_samples} valid train clips in {root_path}")
    dropped = len(train_groups) - len(class_names)
    logger.info(
        "Kept %d glosses and dropped %d below min_train_samples=%d",
        len(class_names),
        dropped,
        min_train_samples,
    )
    label_path = root_path / "labels.json" if metadata_path is None else Path(metadata_path)
    _write_labels(label_path, class_names)

    test_groups = _gloss_files(root_path / "test")
    test_paths = [path for name in class_names for path in test_groups.get(name, [])]
    valid_test_paths = set(_valid_paths(test_paths, min_shoulder, worker_count, "test"))
    valid_test = {
        name: [path for path in test_groups.get(name, []) if path in valid_test_paths]
        for name in class_names
    }
    train_samples = _samples_for(valid_train, class_names)
    test_samples = _samples_for(valid_test, class_names)
    logger.info("Train clips %d, test clips %d", len(train_samples), len(test_samples))
    shared = {
        "modes": selected_modes,
        "is_leg": is_leg,
        "seed": seed,
        "min_shoulder": min_shoulder,
        "scale_range": scale_range,
        "rotate_deg": rotate_deg,
        "noise_std": noise_std,
        "crop_range": crop_range,
    }
    train_dataset = KeypointDataset(train_samples, class_names, train=True, **shared)
    test_dataset = KeypointDataset(test_samples, class_names, train=False, **shared)
    return train_dataset, test_dataset, class_names
