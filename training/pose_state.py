"""Train PoseStateMLP on the windows written by preprocess.pose_state.

    python -m preprocess.pose_state --root vsl400-keypoint --out vsl400-pose-state
    python -m training.pose_state --data vsl400-pose-state

The checkpoint is selected on a by-clip holdout of the train split, so the
test number is not optimistic. After training, the best weights run over
whole test clips, including the unlabelled transition frames, and the mean
action probability is logged per tenth of the clip.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models import PoseStateMLP, count_parameters, pick_device
from preprocess.pose_state import (
    LABEL_NAMES,
    NUM_COORDS,
    NUM_JOINTS,
    POSE_JOINTS,
    WINDOW,
    clip_windows,
    load_raw,
    load_split,
    make_pose_loaders,
)
from training.engine import parameter_groups

logger = logging.getLogger("training.pose_state")


@dataclass(frozen=True)
class StateScores:
    loss: float
    accuracy: float
    macro_f1: float
    recall: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]

    def __str__(self) -> str:
        recalls = " ".join(f"{name}={value:.4f}" for name, value in zip(LABEL_NAMES, self.recall))
        return f"loss={self.loss:.4f} acc={self.accuracy:.4f} macroF1={self.macro_f1:.4f} recall[{recalls}]"


def scores_from(confusion: np.ndarray, loss_sum: float) -> StateScores:
    """Rows are targets, columns predictions. A class with no support scores 0."""
    total = int(confusion.sum())
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return StateScores(
        loss=loss_sum / max(total, 1),
        accuracy=float(true_positive.sum() / max(total, 1)),
        macro_f1=float(f1.mean()),
        recall=tuple(float(value) for value in recall),
        confusion=tuple(tuple(int(value) for value in row) for row in confusion),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> StateScores:
    model.eval()
    classes = len(LABEL_NAMES)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    loss_sum = 0.0
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(windows)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        index = (labels * classes + logits.argmax(dim=1)).cpu().numpy()
        confusion += np.bincount(index, minlength=classes * classes).reshape(classes, classes)
    return scores_from(confusion, loss_sum)


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def class_weights(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse frequency, scaled so the weights average to 1."""
    counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    seen = 0
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss = criterion(model(windows), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach()) * labels.shape[0]
        seen += labels.shape[0]
    return loss_sum / max(seen, 1)


def save_checkpoint(path: Path, model: PoseStateMLP, epoch: int, val: StateScores, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model.config,
            "label_names": list(LABEL_NAMES),
            "pose_joints": POSE_JOINTS.tolist(),
            "epoch": epoch,
            "val": val.__dict__,
            "args": {key: str(value) for key, value in vars(args).items()},
        },
        path,
    )


def _clip_job(path: str):
    clip = load_raw(path)
    if clip is None:
        return None
    return clip_windows(clip)


@torch.no_grad()
def transition_profile(
    model: nn.Module,
    data_dir: Path,
    device: torch.device,
    clips: int,
    bins: int,
    seed: int,
    workers: Optional[int],
) -> np.ndarray:
    """Mean P(action) per position bin over whole test clips, transition frames included.

    A healthy model rises from near 0 at the start, stays high through the
    middle, and falls back at the end.
    """
    paths = load_split(data_dir, "test")["paths"]
    rng = np.random.default_rng(seed)
    chosen = [str(path) for path in rng.permutation(paths)[: min(clips, len(paths))]]
    worker_count = max(1, min(workers or os.cpu_count() or 1, len(chosen)))
    with Pool(processes=worker_count) as pool:
        loaded = pool.map(_clip_job, chosen, chunksize=max(1, len(chosen) // (worker_count * 4)))

    model.eval()
    sums = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    for item in loaded:
        if item is None:
            continue
        windows, valid = item
        if not valid.any():
            continue
        tensor = torch.from_numpy(windows[valid]).to(device)
        probability = torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
        position = np.flatnonzero(valid) / max(len(valid) - 1, 1)
        index = np.minimum((position * bins).astype(np.int64), bins - 1)
        np.add.at(sums, index, probability)
        np.add.at(counts, index, 1)
    profile = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)
    for index, value in enumerate(profile):
        logger.info(
            "  clip %3d%%-%3d%%  P(action)=%.3f  frames=%d",
            100 * index // bins,
            100 * (index + 1) // bins,
            value,
            counts[index],
        )
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training.pose_state", description="Train the start/action pose classifier.")
    parser.add_argument("--data", type=Path, default=Path("vsl400-pose-state"), help="holds train.npz and test.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pose_state_mlp_best.pth"))
    parser.add_argument("--profile-clips", type=int, default=1000, help="0 skips the transition profile")
    parser.add_argument("--workers", type=int, default=None)
    return parser


def fit(args: argparse.Namespace) -> StateScores:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    train_loader, val_loader, test_loader = make_pose_loaders(
        args.data, batch_size=args.batch_size, val_fraction=args.val_fraction, seed=args.seed
    )
    train_set = train_loader.dataset

    model = PoseStateMLP(
        num_joints=NUM_JOINTS,
        window=WINDOW,
        num_coords=NUM_COORDS,
        num_classes=len(LABEL_NAMES),
        hidden=tuple(args.hidden),
        dropout=args.dropout,
    ).to(device)
    logger.info("PoseStateMLP on %s, %s parameters", device, f"{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_set.class_counts(), device), label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch)

    best_f1 = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.grad_clip)
        val = evaluate(model, val_loader, device)
        improved = val.macro_f1 > best_f1
        if improved:
            best_f1, best_epoch = val.macro_f1, epoch
            save_checkpoint(args.checkpoint, model, epoch, val, args)
        logger.info(
            "epoch %3d/%d  lr=%.2e  train_loss=%.4f  val %s%s",
            epoch + 1,
            args.epochs,
            scheduler.get_last_lr()[0],
            train_loss,
            val,
            "  *" if improved else "",
        )

    model, _payload = PoseStateMLP.from_checkpoint(args.checkpoint, device)
    test = evaluate(model, test_loader, device)
    logger.info("Best epoch %d, val macroF1=%.4f, saved to %s", best_epoch + 1, best_f1, args.checkpoint)
    logger.info("Test %s", test)
    logger.info("Test confusion (rows target %s, cols predicted): %s", LABEL_NAMES, test.confusion)
    if args.profile_clips > 0:
        logger.info("Transition profile over %d test clips", args.profile_clips)
        transition_profile(model, args.data, device, args.profile_clips, 10, args.seed, args.workers)
    return test


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    if not (args.data / "train.npz").is_file():
        logger.error("%s/train.npz is missing, run python -m preprocess.pose_state first", args.data)
        return 2
    fit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
