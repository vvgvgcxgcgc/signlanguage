"""Training package for the skeleton gloss classifiers in `models`.

The test split doubles as validation, every optimizer step can span several
micro-batches through gradient accumulation, and a run scales out with either
DataParallel (`parallel="dp"`) or one process per GPU (`parallel="ddp"`).
"""

from training.config import (
    OPTIMIZERS,
    PARALLEL_MODES,
    SELECTION_METRICS,
    TrainConfig,
)
from training.data import Splits, build_splits, set_epoch
from training.distributed import Topology, resolve_device, setup
from training.engine import (
    ModelEma,
    RunResult,
    evaluate,
    fit,
    make_amp,
    make_optimizer,
    make_scheduler,
    parameter_groups,
    train_one_epoch,
    tune_backend,
    unwrap,
    wrap_parallel,
)
from training.metrics import MetricAccumulator, Scores

__all__ = [
    "MetricAccumulator",
    "ModelEma",
    "OPTIMIZERS",
    "PARALLEL_MODES",
    "RunResult",
    "SELECTION_METRICS",
    "Scores",
    "Splits",
    "Topology",
    "TrainConfig",
    "build_splits",
    "evaluate",
    "fit",
    "make_amp",
    "make_optimizer",
    "make_scheduler",
    "parameter_groups",
    "resolve_device",
    "set_epoch",
    "setup",
    "train_one_epoch",
    "tune_backend",
    "unwrap",
    "wrap_parallel",
]
