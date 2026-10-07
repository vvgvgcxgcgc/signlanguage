"""Optimizer, schedule, EMA, and the accumulate-aware train and eval loops."""

from __future__ import annotations

import copy
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from models import build_model, count_parameters, save_checkpoint
from training.config import TrainConfig
from training.data import Splits, build_splits, set_epoch
from training.distributed import Topology, barrier, cleanup, resolve_device, setup
from training.metrics import MetricAccumulator, Scores

logger = logging.getLogger(__name__)

# Weight decay on these would fight what they are for. edge_importance starts
# at ones and topology starts at the normalized adjacency, so pulling them to
# zero erases the declared skeleton. The rest are tokens and gates, not
# weights, and decaying them only biases the model toward the origin.
NO_DECAY_PARAMETERS = frozenset(
    {"edge_importance", "topology", "offset", "alpha", "missing", "class_token", "position"}
)


def unwrap(model: nn.Module) -> nn.Module:
    """Reach the real module through DataParallel or DistributedDataParallel."""
    return model.module if isinstance(model, (nn.DataParallel, DistributedDataParallel)) else model


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into a decayed group and an undecayed one.

    Anything with at most one dimension is a bias or a norm scale, and anything
    named in NO_DECAY_PARAMETERS is a graph or token parameter. Both are
    excluded.
    """
    decayed: list[nn.Parameter] = []
    plain: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if parameter.ndim <= 1 or leaf in NO_DECAY_PARAMETERS:
            plain.append(parameter)
        else:
            decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    groups = parameter_groups(unwrap(model), config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            groups, lr=config.lr, momentum=config.momentum, nesterov=config.momentum > 0
        )
    return torch.optim.AdamW(groups, lr=config.lr)


def make_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, steps_per_epoch: int) -> LambdaLR:
    """Linear warmup then cosine decay, stepped once per optimizer update.

    The schedule counts optimizer steps, not micro-batches, so changing
    `accumulate` stretches the same curve instead of reshaping it.
    """
    warmup = config.warmup_epochs * steps_per_epoch
    total = max(config.epochs * steps_per_epoch, 1)

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return LambdaLR(optimizer, scale)


def make_amp(device: torch.device, enabled: bool) -> tuple[torch.dtype | None, torch.amp.GradScaler]:
    """Pick an autocast dtype and a matching scaler. Only CUDA gets autocast.

    bfloat16 needs no loss scaling. A T4 is Turing and has no bfloat16, so it
    takes the float16 path and does need the scaler.
    """
    if not enabled or device.type != "cuda":
        return None, torch.amp.GradScaler(device.type, enabled=False)
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    return torch.float16, torch.amp.GradScaler("cuda", enabled=True)


def tune_backend(device: torch.device) -> None:
    """Let cuDNN autotune, which pays off because every clip has the same shape."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


class ModelEma:
    """Shadow copy of the weights, averaged over steps.

    Evaluated instead of the live weights because the averaged copy is both
    steadier and usually a point or two better late in training.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.module = copy.deepcopy(unwrap(model)).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        shadow = self.module.state_dict()
        for key, value in unwrap(model).state_dict().items():
            stored = shadow[key]
            if stored.dtype.is_floating_point:
                stored.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                # Integer buffers such as num_batches_tracked cannot be mixed.
                stored.copy_(value)


def _prepare(batch, device: torch.device, mask_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one batch to the device. `mask_only` blanks xyz for the leak test."""
    branches, labels = batch
    clip = branches[0].to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if mask_only:
        clip = clip.clone()
        clip[..., :3] = 0.0
    return clip, labels


def train_one_epoch(
    model: nn.Module,
    splits: Splits,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    topology: Topology,
    ema: ModelEma | None,
    amp_dtype: torch.dtype | None,
    epoch: int,
) -> Scores:
    """One pass over the train split, stepping every `accumulate` micro-batches."""
    model.train()
    accumulator = MetricAccumulator(splits.num_classes, device)
    optimizer.zero_grad(set_to_none=True)

    loader = splits.train_loader
    total = len(loader)
    progress = tqdm(
        loader,
        total=total,
        desc=f"epoch {epoch}/{config.epochs}",
        disable=not topology.is_main,
        leave=False,
    )
    for step, batch in enumerate(progress):
        clip, labels = _prepare(batch, device, config.mask_only)
        # The last group of an epoch can be short, so divide by its real size
        # rather than by `accumulate`, or those batches count for less.
        group_start = (step // config.accumulate) * config.accumulate
        group_size = min(config.accumulate, total - group_start)
        is_boundary = (step + 1 - group_start) == group_size

        # Skip the all-reduce until the step that actually consumes the grads.
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not is_boundary
            else nullcontext()
        )
        with sync:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(clip)
                loss = criterion(logits, labels)
            scaler.scale(loss / group_size).backward()

        accumulator.update(logits.detach().float(), labels, loss.detach())

        if is_boundary:
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)
            if topology.is_main:
                progress.set_postfix(loss=f"{float(loss.detach()):.3f}", lr=f"{scheduler.get_last_lr()[0]:.4f}")
    progress.close()

    accumulator.reduce(topology)
    return accumulator.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    config: TrainConfig,
    topology: Topology,
    amp_dtype: torch.dtype | None,
) -> Scores:
    """Score every clip in `loader` once. Under DDP each rank gets a shard."""
    model.eval()
    accumulator = MetricAccumulator(num_classes, device)
    for batch in loader:
        clip, labels = _prepare(batch, device, config.mask_only)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(clip)
            loss = criterion(logits, labels)
        accumulator.update(logits.float(), labels, loss)
    accumulator.reduce(topology)
    return accumulator.compute()


def wrap_parallel(model: nn.Module, config: TrainConfig, topology: Topology, device: torch.device) -> nn.Module:
    """Apply the requested parallel mode, or return the model untouched."""
    if config.parallel == "ddp" and topology.distributed:
        device_ids = [topology.local_rank] if device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=False)
    if config.parallel == "dp":
        if torch.cuda.device_count() > 1:
            return nn.DataParallel(model)
        logger.warning("parallel='dp' needs more than one CUDA device, running on one")
    return model


@dataclass
class RunResult:
    model: str
    best_epoch: int
    best_score: float
    best_scores: Scores | None
    checkpoint: str | None
    parameters: int
    history: list[dict]


def fit(config: TrainConfig) -> RunResult:
    """Train one model end to end and keep the best checkpoint by `select_by`."""
    topology = setup(config.parallel)
    try:
        torch.manual_seed(config.seed + topology.rank)
        device = resolve_device(topology, config.parallel, config.device)
        tune_backend(device)
        splits = build_splits(config, topology)

        model = build_model(
            config.model,
            splits.num_classes,
            is_leg=config.is_leg,
            max_frames=config.frames,
            **config.model_overrides,
        ).to(device)
        parameters = count_parameters(model)
        wrapped = wrap_parallel(model, config, topology, device)

        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        optimizer = make_optimizer(model, config)
        steps_per_epoch = max(math.ceil(len(splits.train_loader) / config.accumulate), 1)
        scheduler = make_scheduler(optimizer, config, steps_per_epoch)
        amp_dtype, scaler = make_amp(device, config.amp)
        ema = ModelEma(model, config.ema_decay) if config.ema_decay > 0 else None

        if topology.is_main:
            logger.info(
                "%s on %s | %d params | micro %d x accum %d x %d rank(s) = batch %d | %d steps/epoch",
                config.model,
                device,
                parameters,
                config.micro_batch,
                config.accumulate,
                topology.world_size,
                config.effective_batch(topology.world_size),
                steps_per_epoch,
            )
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_score = -1.0
        best_epoch = 0
        best_scores: Scores | None = None
        checkpoint: str | None = None
        history: list[dict] = []

        for epoch in range(1, config.epochs + 1):
            set_epoch(splits, epoch)
            train_scores = train_one_epoch(
                wrapped, splits, optimizer, scheduler, scaler, criterion,
                device, config, topology, ema, amp_dtype, epoch,
            )
            scored = ema.module if ema is not None else wrapped
            val_scores = evaluate(
                scored, splits.val_loader, criterion, device,
                splits.num_classes, config, topology, amp_dtype,
            )
            score = getattr(val_scores, config.select_by)
            history.append(
                {"epoch": epoch, "train": train_scores.__dict__, "val": val_scores.__dict__}
            )

            if topology.is_main:
                logger.info(
                    "epoch %3d/%d | train %s | val %s", epoch, config.epochs, train_scores, val_scores
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_scores = val_scores
                    checkpoint = str(
                        save_checkpoint(
                            config.checkpoint_path(),
                            ema.module if ema is not None else model,
                            config.model,
                            splits.num_classes,
                            config.is_leg,
                            config.frames,
                            epoch=epoch,
                            scores=val_scores.__dict__,
                            class_names=splits.class_names,
                        )
                    )
                    logger.info("new best %s=%.4f at epoch %d", config.select_by, best_score, epoch)
            barrier(topology)

        if topology.is_main:
            logger.info(
                "Done. best %s=%.4f at epoch %d -> %s",
                config.select_by, best_score, best_epoch, checkpoint,
            )
        return RunResult(
            model=config.model,
            best_epoch=best_epoch,
            best_score=best_score,
            best_scores=best_scores,
            checkpoint=checkpoint,
            parameters=parameters,
            history=history,
        )
    finally:
        cleanup(topology)
