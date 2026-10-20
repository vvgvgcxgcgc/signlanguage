"""Loaders for a run. The test split is used as validation, by request.

That means the score selecting a checkpoint is measured on the same clips that
get reported, so the final number is optimistic. It is the right call only
while there is no third split; treat the headline accuracy as an upper bound.

preprocess.dataset.make_loader is not used here because it has no notion of
rank. Under DDP each rank needs its own draw from BalancedBatchSampler, which
it gets through a rank-offset seed, and the validation pass needs to be sharded
so no clip is scored twice.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from preprocess.dataset import (
    BalancedBatchSampler,
    KeypointDataset,
    MULTIRATE_MODES,
    TemporalMode,
    build_datasets,
)
from training.config import TrainConfig
from training.distributed import Topology

logger = logging.getLogger(__name__)

_POLL_SECONDS = 2.0
_LOG_EVERY_SECONDS = 60.0


@dataclass
class Splits:
    """Loaders plus the label order every checkpoint has to agree on."""

    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: KeypointDataset
    val_dataset: KeypointDataset
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_joints(self) -> int:
        return self.train_dataset.num_joints


def _ready_path(config: TrainConfig, rank: int) -> Path:
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    return config.checkpoint_dir / f".dataset-ready-{run_id}-rank{rank}"


def _publish_ready(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("ready\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait_ready(path: Path, rank: int) -> None:
    """Poll a sentinel. This must not be an NCCL collective.

    ``dist.barrier`` is an allreduce, and the NCCL watchdog aborts it after
    600s. Loading clips takes longer than that on the full set.
    """
    logger.info("Waiting for rank %d to finish loading clips", rank)
    started = time.monotonic()
    next_log = started
    while not path.is_file():
        now = time.monotonic()
        if now - next_log >= _LOG_EVERY_SECONDS:
            logger.info(
                "Still waiting for rank %d, %.0f min elapsed",
                rank,
                (now - started) / 60.0,
            )
            next_log = now
        time.sleep(_POLL_SECONDS)


def _wait_for_earlier_ranks(config: TrainConfig, topology: Topology) -> None:
    if not topology.distributed:
        return
    for earlier in range(topology.rank):
        _wait_ready(_ready_path(config, earlier), earlier)


def _publish_and_wait_for_later_ranks(config: TrainConfig, topology: Topology) -> None:
    if not topology.distributed:
        return
    _publish_ready(_ready_path(config, topology.rank))
    for later in range(topology.rank + 1, topology.world_size):
        _wait_ready(_ready_path(config, later), later)


def build_splits(config: TrainConfig, topology: Topology) -> Splits:
    """Load the label list and the cached clips once per rank.

    Ranks load one at a time and wait on sentinel files. The NCCL watchdog
    stays idle until every rank has its clips in RAM and training actually
    starts.
    """
    modes = (
        MULTIRATE_MODES
        if config.model == "MultiRate-Attn-STGCN"
        else (TemporalMode("main", config.frames),)
    )
    _wait_for_earlier_ranks(config, topology)
    train_dataset, val_dataset, class_names = build_datasets(
        config.root,
        min_train_samples=config.min_train_samples,
        modes=modes,
        is_leg=config.is_leg,
        seed=config.seed,
        min_shoulder=config.min_shoulder,
        workers=config.workers if config.workers > 0 else None,
        scale_range=config.scale_range,
        rotate_deg=config.rotate_deg,
        noise_std=config.noise_std,
        crop_range=config.crop_range,
        name=f"{config.tag}_{config.model}",
        metadata_path=config.labels,
    )
    _publish_and_wait_for_later_ranks(config, topology)

    if len(train_dataset) == 0:
        raise RuntimeError(f"no usable train clips under {config.root}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"no usable test clips under {config.root}")

    # Each rank draws its own balanced batch, so the effective batch is
    # micro_batch * world_size per optimizer micro-step.
    sampler = BalancedBatchSampler(
        train_dataset.labels,
        config.micro_batch,
        config.seed + 1000 * topology.rank,
    )
    # Clips already sit in RAM. Extra workers would fork that tensor and the
    # CUDA context, which is what was filling host memory.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        if topology.distributed
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    if topology.is_main:
        logger.info(
            "Classes %d, joints %d, train clips %d, val clips %d (val is the test split)",
            len(class_names),
            train_dataset.num_joints,
            len(train_dataset),
            len(val_dataset),
        )
    return Splits(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_names=list(class_names),
    )


def set_epoch(splits: Splits, epoch: int) -> None:
    """Advance the augmentation draw and the balanced sampler for one epoch."""
    splits.train_dataset.set_epoch(epoch)
    sampler = splits.train_loader.sampler
    if isinstance(sampler, BalancedBatchSampler):
        sampler.set_epoch(epoch)
    val_sampler = splits.val_loader.sampler
    if isinstance(val_sampler, DistributedSampler):
        val_sampler.set_epoch(epoch)
