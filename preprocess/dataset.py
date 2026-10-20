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
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)
_SPAWN = get_context("spawn")

N_JOINTS = 76
N_COORDS = 3
N_CHANNELS = 4
BODY_COUNT = 34
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
MIN_VALID_FRAMES = 3
# One hand is 21 joints. A one-handed sign may hide a whole hand. Losing more
# than that means both hands failed, so the frame is not a sign frame.
MAX_MISSING_HAND_POINTS = 21
# Inference refuses a clip when at least this share of its frames was rejected.
MAX_DROPPED_FRACTION = 0.5
# A live queue is mostly the gesture, about 30-40 frames. Training clips sit
# nearer 60 and keep the rest pose at both ends, so inference repeats the
# opening rest and the closing rest this many times on each side.
PREPARE_PAD = 5
# Training keeps one contiguous span of 85-95% of the clip, so about a tenth
# is cut. Wider cuts reach into the signing span of short signs.
CROP_RANGE = (0.85, 0.95)
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
MULTIRATE_MODES = (
    TemporalMode("t32", 32),
    TemporalMode("t64", 64),
    TemporalMode("t96", 96),
)


def shoulder_frame_mask(clip: np.ndarray, min_shoulder: float) -> np.ndarray:
    """True where both shoulders exist and the 2D shoulder width clears `min_shoulder`.

    The two shoulders are the base set. A failed pose zeros the whole body at
    once, and every model scales the clip by this width, so a frame that lost
    them has lost the only reference the checkpoint was trained on.
    """
    left = clip[:, LEFT_SHOULDER]
    right = clip[:, RIGHT_SHOULDER]
    present = ~(np.all(left == 0, axis=-1) | np.all(right == 0, axis=-1))
    distance = np.linalg.norm(left[:, :2] - right[:, :2], axis=-1)
    return present & (distance > min_shoulder)


def missing_hand_points(clip: np.ndarray) -> np.ndarray:
    """Per-frame count of hand joints sitting at (0, 0, 0). Shape `(T,)`."""
    hands = clip[:, BODY_COUNT:]
    return np.all(hands == 0, axis=-1).sum(axis=-1)


def accept_raw_frame(row: np.ndarray, min_shoulder: float, check_hands: bool = True) -> bool:
    """Keep one raw `(76, 3)` frame, or refuse it.

    The base set is the shoulder pair. A frame that lost it is refused.
    `check_hands` is false only for a rest frame whose hands have not been
    filled yet. Once hands have been run, more than one hand of missing
    points refuses the frame too.
    """
    frame = np.asarray(row, dtype=np.float32)
    if frame.shape != (N_JOINTS, N_COORDS):
        raise ValueError(f"expected ({N_JOINTS}, {N_COORDS}), got {frame.shape}")
    if min_shoulder <= 0:
        raise ValueError(f"min_shoulder must be positive, got {min_shoulder}")
    if not bool(shoulder_frame_mask(frame[None], min_shoulder)[0]):
        return False
    if not check_hands:
        return True
    return int(missing_hand_points(frame[None])[0]) <= MAX_MISSING_HAND_POINTS


def select_sign_frames(array: np.ndarray, min_shoulder: float = 1e-3) -> tuple[np.ndarray, int]:
    """Drop raw frames that lost the shoulder base or more than one hand.

    Training keeps a missing hand as a validity-0 hole. Inference does not:
    those frames never reach `resample_clip`. Returns the kept `(N, 76, 3)`
    clip and how many frames were removed. An empty input stays empty.
    """
    clip = np.asarray(array, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1:] != (N_JOINTS, N_COORDS):
        raise ValueError(f"expected (T, {N_JOINTS}, {N_COORDS}), got {clip.shape}")
    if min_shoulder <= 0:
        raise ValueError(f"min_shoulder must be positive, got {min_shoulder}")
    if clip.shape[0] == 0:
        return np.zeros((0, N_JOINTS, N_COORDS), dtype=np.float32), 0
    keep = shoulder_frame_mask(clip, min_shoulder) & (missing_hand_points(clip) <= MAX_MISSING_HAND_POINTS)
    dropped = int(clip.shape[0] - int(np.count_nonzero(keep)))
    if not np.any(keep):
        return np.zeros((0, N_JOINTS, N_COORDS), dtype=np.float32), dropped
    return np.ascontiguousarray(clip[keep]), dropped


def clip_needs_redo(kept: int, dropped: int, max_fraction: float = MAX_DROPPED_FRACTION) -> bool:
    """True when the surviving frames are too few, or the reject rate is too high.

    Fewer than three kept frames cannot be normalized. A reject rate at or
    above `max_fraction` means the clip is mostly holes, so the caller alerts
    and does not score it.
    """
    if kept < 0 or dropped < 0:
        raise ValueError(f"frame counts must be non-negative, got kept={kept} dropped={dropped}")
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")
    if kept < MIN_VALID_FRAMES:
        return True
    return dropped / (kept + dropped) >= max_fraction


def pad_prepare(clip: np.ndarray, count: int = PREPARE_PAD) -> np.ndarray:
    """Repeat the first frame in front and the last frame behind, `count` times each.

    Those two frames are the rest poses that opened and closed the live queue.
    Five extra copies on each side bring a 30-40 frame gesture closer to the
    rest-sign-rest shape of a training clip. An empty clip is unchanged.
    """
    if count < 0:
        raise ValueError(f"prepare pad must be non-negative, got {count}")
    array = np.asarray(clip, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (N_JOINTS, N_COORDS):
        raise ValueError(f"expected (T, {N_JOINTS}, {N_COORDS}), got {array.shape}")
    if array.shape[0] == 0 or count == 0:
        return np.ascontiguousarray(array)
    head = np.repeat(array[:1], count, axis=0)
    tail = np.repeat(array[-1:], count, axis=0)
    return np.ascontiguousarray(np.concatenate((head, array, tail), axis=0))


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
    valid = shoulder_frame_mask(clip, min_shoulder)
    if int(np.count_nonzero(valid)) < MIN_VALID_FRAMES:
        return None
    left = clip[:, LEFT_SHOULDER]
    right = clip[:, RIGHT_SHOULDER]
    distance = np.linalg.norm(left[:, :2] - right[:, :2], axis=-1)
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


def resample_clip(
    array: np.ndarray,
    length: int,
    is_leg: bool,
    min_shoulder: float = 1e-3,
) -> Optional[np.ndarray]:
    """Raw `(N, 76, 3)` of any length to `(length, J, 4)` for one model.

    `length` is that model's temporal size, the `max_frames` stored in its
    checkpoint. The steps are the training ones: shoulder-normalize, drop the
    legs unless `is_leg`, then interpolate. Returns None when the shoulders
    are too weak to normalize.
    """
    if length < 1:
        raise ValueError(f"temporal length must be positive, got {length}")
    if min_shoulder <= 0:
        raise ValueError(f"min_shoulder must be positive, got {min_shoulder}")
    normalized = normalize_clip(array, min_shoulder)
    if normalized is None:
        return None
    if not is_leg:
        normalized = drop_legs(normalized)
    return interpolate_clip(normalized, length)


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


def _aligned_temporal_crop(
    clip: np.ndarray,
    fraction: float,
    start_fraction: float,
) -> np.ndarray:
    """Slice the same relative span from one branch of a multi-rate clip.

    `fraction` and `start_fraction` are drawn once per sample and reused, so
    T=32, 64, and 96 stay on the same signing interval. Single-branch training
    still uses `_temporal_crop` and keeps its original RNG stream.
    """
    frames = int(clip.shape[0])
    if frames <= 1:
        return clip
    span = int(np.clip(np.round(fraction * frames), 1, frames))
    if span >= frames:
        return clip
    start = int(np.clip(np.round(start_fraction * (frames - span)), 0, frames - span))
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
    """Paths of raw clips. `__getitem__` reads one file and resamples it.

    Nothing is stacked into a cache. A sample is `(T, 76, 3)` on disk and
    comes back as one `(T_mode, J, 4)` tensor per branch: xyz, then validity.
    Training crops and augments that resampled clip. `set_epoch` writes a
    shared counter so loader workers keep the same draw as the main process.
    """

    def __init__(
        self,
        paths: Sequence[Path | str],
        labels: Sequence[int],
        class_names: Sequence[str],
        modes: Sequence[TemporalMode],
        is_leg: bool,
        train: bool,
        seed: int,
        min_shoulder: float,
        scale_range: tuple[float, float] = (0.5, 1.5),
        rotate_deg: float = 15.0,
        noise_std: float = 0.01,
        crop_range: tuple[float, float] = CROP_RANGE,
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
        self.modes = _check_modes(modes)
        self.paths = tuple(str(path) for path in paths)
        self._labels = [int(label) for label in labels]
        if len(self._labels) != len(self.paths):
            raise ValueError(f"expected {len(self.paths)} labels, got {len(self._labels)}")
        self.class_names = tuple(class_names)
        self.is_leg = bool(is_leg)
        self.train = bool(train)
        self.seed = int(seed)
        self.min_shoulder = float(min_shoulder)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.rotate_deg = float(rotate_deg)
        self.noise_std = float(noise_std)
        self.crop_range = (float(crop_range[0]), float(crop_range[1]))
        self._epoch = _SPAWN.Value("q", 0)

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def num_joints(self) -> int:
        return N_JOINTS if self.is_leg else int(_KEEP_JOINTS.shape[0])

    @property
    def num_channels(self) -> int:
        return N_CHANNELS

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw for this epoch, including the temporal crop."""
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self._labels)

    def _resample(self, path: str) -> tuple[np.ndarray, ...]:
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"failed to read clip {path}") from exc
        normalized = normalize_clip(array, self.min_shoulder)
        if normalized is None:
            raise RuntimeError(f"clip failed shoulder normalization: {path}")
        if not self.is_leg:
            normalized = drop_legs(normalized)
        return tuple(interpolate_clip(normalized, mode.length) for mode in self.modes)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], int]:
        label = self._labels[index]
        branches = self._resample(self.paths[index])
        if not self.train:
            return tuple(torch.from_numpy(branch) for branch in branches), label
        rng = _sample_rng(self.seed, int(self._epoch.value), index)
        cropped: list[np.ndarray] = []
        if len(branches) == 1:
            clip = branches[0]
            span = _temporal_crop(clip, rng, self.crop_range)
            if span.shape[0] != clip.shape[0]:
                span = interpolate_clip(span, int(clip.shape[0]))
            cropped.append(span)
        else:
            fraction = float(rng.uniform(self.crop_range[0], self.crop_range[1]))
            start_fraction = float(rng.random())
            for clip in branches:
                span = _aligned_temporal_crop(clip, fraction, start_fraction)
                if span.shape[0] != clip.shape[0]:
                    span = interpolate_clip(span, int(clip.shape[0]))
                cropped.append(span)
        augmented = _augment_branches(cropped, rng, self.scale_range, self.rotate_deg, self.noise_std)
        return tuple(torch.from_numpy(branch) for branch in augmented), label


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


def loader_kwargs(num_workers: int) -> dict:
    """Spawn workers so they do not inherit a CUDA context created later.

    Zero workers load on the calling process. A positive count keeps the
    workers alive across epochs; `KeypointDataset.set_epoch` is a shared
    counter, so the augmentation draw still advances.
    """
    count = int(num_workers)
    if count < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if count == 0:
        return {"num_workers": 0}
    return {
        "num_workers": count,
        "multiprocessing_context": _SPAWN,
        "persistent_workers": True,
        "worker_init_fn": _seed_worker,
    }


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
    runtime = loader_kwargs(num_workers)
    if dataset.train:
        sampler = BalancedBatchSampler(dataset.labels, batch_size, seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            generator=generator,
            **runtime,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        **runtime,
    )


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


def _read_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"labels file must contain a non-empty 'labels' list: {path}")
    names: list[str] = []
    seen: set[str] = set()
    for item in labels:
        if not isinstance(item, str) or not item or item in {".", ".."} or "/" in item or "\\" in item:
            raise ValueError(f"invalid label {item!r} in {path}")
        if item in seen:
            raise ValueError(f"duplicate label {item!r} in {path}")
        seen.add(item)
        names.append(item)
    return names


def _labels_from_counts(train_dir: Path, min_train_samples: int) -> list[str]:
    grouped = _gloss_files(train_dir)
    names = sorted(name for name, paths in grouped.items() if len(paths) >= min_train_samples)
    if not names:
        raise RuntimeError(
            f"no gloss has at least {min_train_samples} train clips in {train_dir}"
        )
    logger.info(
        "Kept %d glosses and dropped %d below min_train_samples=%d",
        len(names),
        len(grouped) - len(names),
        min_train_samples,
    )
    return names


def _samples_for_labels(
    split_dir: Path,
    class_names: Sequence[str],
    require_clips: bool,
) -> list[tuple[Path, int]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split directory: {split_dir}")
    samples: list[tuple[Path, int]] = []
    empty: list[str] = []
    for label, name in enumerate(class_names):
        gloss_dir = split_dir / name
        files = _npy_files(gloss_dir) if gloss_dir.is_dir() else []
        if not files:
            if require_clips:
                empty.append(name)
            continue
        for path in files:
            samples.append((path, label))
    if empty:
        preview = ", ".join(empty[:8])
        extra = "" if len(empty) <= 8 else f" (+{len(empty) - 8} more)"
        raise FileNotFoundError(
            f"{len(empty)} labels have no clips under {split_dir}: {preview}{extra}"
        )
    return samples


def _label_path(name: str, metadata_path: Optional[Path | str]) -> Path:
    if metadata_path is not None:
        return Path(metadata_path)
    trial_name = Path(name).name
    if not trial_name or trial_name in {".", ".."}:
        raise ValueError(f"label file name must not be empty, got {name!r}")
    return Path.cwd() / "labels" / f"{trial_name}.json"


def build_datasets(
    root: Path | str,
    min_train_samples: int = 1,
    modes: Optional[Sequence[TemporalMode]] = None,
    is_leg: bool = False,
    seed: int = 0,
    min_shoulder: float = 1e-3,
    name: str = "labels",
    metadata_path: Optional[Path | str] = None,
    workers: Optional[int] = None,
    scale_range: tuple[float, float] = (0.5, 1.5),
    rotate_deg: float = 15.0,
    noise_std: float = 0.01,
    crop_range: tuple[float, float] = CROP_RANGE,
) -> tuple[KeypointDataset, KeypointDataset, list[str]]:
    """Index clips for the label list and leave the arrays on disk.

    An existing labels file is the class order: index i in that file is label
    i for both splits, and the clips are not opened to rebuild it. The file is
    `labels/{name}.json` in the current working directory, not under the
    dataset root. `metadata_path` overrides that location.

    When the file is missing, glosses are chosen by train-file count against
    `min_train_samples` and the list is written. `workers` is unused here; the
    training loader spends that count on DataLoader workers.
    """
    del workers
    if min_train_samples < 1:
        raise ValueError(f"min_train_samples must be positive, got {min_train_samples}")
    if min_shoulder <= 0:
        raise ValueError(f"min_shoulder must be positive, got {min_shoulder}")
    selected_modes = SINGLE_MODES if modes is None else _check_modes(modes)
    root_path = Path(root)
    train_dir = root_path / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"missing split directory: {train_dir}")
    label_path = _label_path(name, metadata_path)
    if label_path.is_file():
        class_names = _read_labels(label_path)
        logger.info("Using %d labels from %s", len(class_names), label_path)
    else:
        class_names = _labels_from_counts(train_dir, min_train_samples)
        _write_labels(label_path, class_names)
    train_samples = _samples_for_labels(train_dir, class_names, require_clips=True)
    test_samples = _samples_for_labels(root_path / "test", class_names, require_clips=False)
    logger.info(
        "Indexed %d train clips and %d test clips",
        len(train_samples),
        len(test_samples),
    )
    shared = {
        "class_names": class_names,
        "modes": selected_modes,
        "is_leg": is_leg,
        "seed": seed,
        "min_shoulder": min_shoulder,
        "scale_range": scale_range,
        "rotate_deg": rotate_deg,
        "noise_std": noise_std,
        "crop_range": crop_range,
    }
    train_paths, train_labels = _unzip_samples(train_samples)
    test_paths, test_labels = _unzip_samples(test_samples)
    train_dataset = KeypointDataset(train_paths, train_labels, train=True, **shared)
    test_dataset = KeypointDataset(test_paths, test_labels, train=False, **shared)
    return train_dataset, test_dataset, class_names


def _unzip_samples(samples: Sequence[tuple[Path, int]]) -> tuple[list[Path], list[int]]:
    paths = [path for path, _label in samples]
    labels = [label for _path, label in samples]
    return paths, labels
