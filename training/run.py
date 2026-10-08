"""Command line entry point.

On a Kaggle notebook with two T4s, `dp` is the one that works from a plain
notebook cell and `ddp` is the faster one but needs a shell cell:

    python -m training.run --root DATA --model CTR-GCN --parallel dp
    torchrun --standalone --nproc_per_node=2 -m training.run --root DATA --parallel ddp

Both split `--micro-batch` across the two cards, so the effective batch is
`micro_batch * accumulate` under `dp` and twice that under `ddp`, where each
rank draws its own balanced micro-batch. A T4 has no bfloat16, so mixed
precision runs in float16 with a loss scaler.

Single device, no parallelism:

    python -m training.run --root DATA --model CTR-GCN --micro-batch 32 --accumulate 2

Train every architecture back to back with `--model all`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from models import MODEL_NAMES
from training.config import DEVICES, OPTIMIZERS, PARALLEL_MODES, SELECTION_METRICS, TrainConfig
from training.engine import fit

logger = logging.getLogger("training")


def configure_logging(verbose: bool) -> None:
    """INFO on rank 0, warnings elsewhere, so ranks do not interleave logs."""
    is_main = int(os.environ.get("RANK", "0")) == 0
    logging.basicConfig(
        level=(logging.DEBUG if verbose else logging.INFO) if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training.run",
        description="Train a skeleton gloss classifier. The test split is used as validation.",
    )
    parser.add_argument("--root", type=Path, required=True, help="dataset root holding train/ and test/")
    parser.add_argument("--model", default="CTR-GCN", choices=(*MODEL_NAMES, "all"))
    parser.add_argument("--frames", type=int, default=64, help="temporal length T")
    parser.add_argument("--legs", action="store_true", help="keep leg joints, J=76 instead of 68")

    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--micro-batch", type=int, default=32, help="samples per forward pass")
    parser.add_argument("--accumulate", type=int, default=2, help="micro-batches per optimizer step")

    parser.add_argument("--optimizer", default="sgd", choices=OPTIMIZERS)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables the averaged copy")

    parser.add_argument("--parallel", default="none", choices=PARALLEL_MODES)
    parser.add_argument(
        "--device",
        default="auto",
        choices=DEVICES,
        help="auto prefers CUDA, then MPS, then CPU",
    )
    parser.add_argument("--no-amp", action="store_true", help="keep float32 on CUDA")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--min-train-samples", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to continue from, usually {tag}_{model}_last.pth",
    )
    parser.add_argument("--tag", default="vsl", help="checkpoint filename prefix")
    parser.add_argument("--select-by", default="macro_f1", choices=SELECTION_METRICS)
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="leak test: zero xyz and keep validity only",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace, model: str) -> TrainConfig:
    return TrainConfig(
        root=args.root,
        model=model,
        frames=args.frames,
        is_leg=args.legs,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        accumulate=args.accumulate,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        amp=not args.no_amp,
        parallel=args.parallel,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        min_train_samples=args.min_train_samples,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        tag=args.tag,
        select_by=args.select_by,
        mask_only=args.mask_only,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if not args.root.is_dir():
        logger.error("Dataset root %s does not exist", args.root)
        return 2

    names = MODEL_NAMES if args.model == "all" else (args.model,)
    results = []
    for name in names:
        results.append(fit(config_from_args(args, name)))

    for result in results:
        logger.info(
            "%-12s best %s at epoch %d, %d params, %s",
            result.model,
            f"{result.best_score:.4f}",
            result.best_epoch,
            result.parameters,
            result.best_scores,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
