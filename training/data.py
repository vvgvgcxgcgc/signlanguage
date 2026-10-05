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
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from preprocess.dataset import (
    BalancedBatchSampler,
    KeypointDataset,
    TemporalMode,
    build_datasets,
)
from training.config import TrainConfig
from training.distributed import Topology, barrier

logger = logging.getLogger(__name__)


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


def _seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_splits(config: TrainConfig, topology: Topology) -> Splits:
    """Scan the dataset once per rank and wire up both loaders."""
    modes = (TemporalMode("main", config.frames),)
    # Rank 0 goes first so only one process writes labels.json.
    if not topology.is_main:
        barrier(topology)
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
    )
    if topology.is_main:
        barrier(topology)

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        drop_last=False,
        # Workers must be rebuilt each epoch, otherwise set_epoch never reaches
        # the copies holding the augmentation seed.
        persistent_workers=False,
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
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        drop_last=False,
        persistent_workers=False,
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
