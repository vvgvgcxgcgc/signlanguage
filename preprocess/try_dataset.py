"""Build the datasets and log one train batch shape and one test batch shape.

Point ROOT at a folder with train/ and test/ gloss directories of raw
(T, 76, 3) clips. Leave it None to use a small synthetic tree. From the
signlanguage project root:

    python -m preprocess.try_dataset
"""

from __future__ import annotations

import logging
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from preprocess.dataset import (
    N_JOINTS,
    SLOWFAST_MODES,
    SINGLE_MODES,
    TemporalMode,
    build_datasets,
    loader_kwargs,
    make_loader,
)

logger = logging.getLogger("try_dataset")

# Edit these, then run the module again.
ROOT: Optional[str] = None
MIN_TRAIN_SAMPLES = 1
MODE = "slowfast"  # "main" is one 60-frame branch, "slowfast" is 8 and 32
IS_LEG = False
SEED = 0
MIN_SHOULDER = 1e-3
SCALE_RANGE = (0.5, 1.5)
ROTATE_DEG = 15.0
NOISE_STD = 0.01
CROP_RANGE = (0.85, 0.95)
BATCH_SIZE = 4
WORKERS = 2


def _selected_modes() -> Sequence[TemporalMode]:
    if MODE == "main":
        return SINGLE_MODES
    if MODE == "slowfast":
        return SLOWFAST_MODES
    raise ValueError(f"MODE must be 'main' or 'slowfast', got {MODE}")


def _save_npy(job: tuple[str, np.ndarray]) -> str:
    path, array = job
    np.save(path, np.asarray(array, dtype=np.float32))
    return path


def _raw_clip(frames: int = 12) -> np.ndarray:
    clip = np.zeros((frames, N_JOINTS, 3), dtype=np.float32)
    clip[:, 11, 0] = -0.5
    clip[:, 12, 0] = 0.5
    return clip


def _write_synthetic(root: Path) -> None:
    plan = {
        ("train", "alpha"): 8,
        ("train", "beta"): 4,
        ("test", "alpha"): 2,
        ("test", "beta"): 2,
    }
    jobs: list[tuple[Path, np.ndarray]] = []
    clip = _raw_clip()
    for (split, gloss), count in plan.items():
        for index in range(count):
            jobs.append((root / split / gloss / f"{index:02d}.npy", clip))
    for path, _array in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [(str(path), array) for path, array in jobs]
    workers = max(1, min(len(payloads), os.cpu_count() or 1))
    chunksize = max(1, len(payloads) // (workers * 4))
    with Pool(processes=workers) as pool:
        pool.map(_save_npy, payloads, chunksize=chunksize)


def _log_shape(name: str, dataset) -> None:
    if len(dataset) == 0:
        logger.info("%s loader is empty", name)
        return
    dataset.set_epoch(0)
    loader = make_loader(dataset, BATCH_SIZE, SEED, num_workers=WORKERS)
    iterator = iter(loader)
    try:
        branches, labels = next(iterator)
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        del iterator
        del loader
    shapes = ", ".join(f"{mode.name}={tuple(branch.shape)}" for mode, branch in zip(dataset.modes, branches))
    logger.info("%s %s labels=%s", name, shapes, tuple(labels.shape))


def _run(root: Path, metadata_path: Path) -> None:
    try:
        train_set, test_set, _names = build_datasets(
            root,
            min_train_samples=MIN_TRAIN_SAMPLES,
            modes=_selected_modes(),
            is_leg=IS_LEG,
            seed=SEED,
            min_shoulder=MIN_SHOULDER,
            metadata_path=metadata_path,
            workers=WORKERS,
            scale_range=SCALE_RANGE,
            rotate_deg=ROTATE_DEG,
            noise_std=NOISE_STD,
            crop_range=CROP_RANGE,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("build_datasets stopped: %s", exc)
        return
    _log_shape("train", train_set)
    _log_shape("test", test_set)
    _check_epoch(train_set)


def _check_epoch(dataset) -> None:
    """Workers must see set_epoch, or every epoch repeats the first crop."""
    dataset.set_epoch(1)
    held = dataset[0][0][0].clone()
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        **loader_kwargs(WORKERS),
    )
    iterator = iter(loader)
    try:
        dataset.set_epoch(2)
        branched = next(iterator)[0][0][0]
        fresh = dataset[0][0][0]
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        del iterator
        del loader
    if torch.equal(held, fresh):
        logger.error("epoch change left the train sample unchanged")
        return
    if not torch.equal(branched, fresh):
        logger.error("loader workers kept a stale epoch")
        return
    logger.info("workers follow set_epoch")


def _check_labels_stick(root: Path, metadata_path: Path) -> None:
    """A second build must keep the existing label list and stay off the files."""
    original = metadata_path.read_text(encoding="utf-8")
    decoy = root / "train" / "gamma"
    decoy.mkdir(parents=True, exist_ok=True)
    np.save(decoy / "00.npy", _raw_clip())
    _train, _test, names = build_datasets(
        root,
        min_train_samples=MIN_TRAIN_SAMPLES,
        modes=_selected_modes(),
        is_leg=IS_LEG,
        seed=SEED,
        min_shoulder=MIN_SHOULDER,
        metadata_path=metadata_path,
        scale_range=SCALE_RANGE,
        rotate_deg=ROTATE_DEG,
        noise_std=NOISE_STD,
        crop_range=CROP_RANGE,
    )
    if names != ["alpha", "beta"] or metadata_path.read_text(encoding="utf-8") != original:
        logger.error("existing labels file was rebuilt")
        return
    if any("gamma" in path for path in _train.paths):
        logger.error("labels file did not limit the indexed glosses")
        return
    logger.info("existing labels file left unchanged")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if ROOT is None:
        with tempfile.TemporaryDirectory(prefix="try-dataset-") as folder:
            root = Path(folder)
            _write_synthetic(root)
            metadata_path = root / "labels.json"
            _run(root, metadata_path)
            _check_labels_stick(root, metadata_path)
        return
    root = Path(ROOT)
    with tempfile.TemporaryDirectory(prefix="try-labels-") as folder:
        _run(root, Path(folder) / "labels.json")


if __name__ == "__main__":
    main()
