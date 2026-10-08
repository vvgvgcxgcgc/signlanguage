"""One config object for a training run.

The effective batch is `micro_batch * accumulate * world_size`. Raising
`accumulate` buys a larger effective batch without more memory, but batch norm
still only ever sees `micro_batch` samples, so its running statistics are
noisier than the effective batch would suggest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

OPTIMIZERS = ("sgd", "adamw")
PARALLEL_MODES = ("none", "dp", "ddp")
SELECTION_METRICS = ("accuracy", "macro_f1", "top5")
DEVICES = ("auto", "cpu", "cuda", "mps")


@dataclass
class TrainConfig:
    """Everything a run needs. `root` holds the train/ and test/ gloss folders."""

    root: Path
    model: str = "CTR-GCN"
    frames: int = 64
    is_leg: bool = False

    epochs: int = 70
    micro_batch: int = 32
    accumulate: int = 2

    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 4e-4
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.0
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    amp: bool = True
    parallel: str = "none"
    device: str = "auto"
    workers: int = 4
    seed: int = 0

    min_train_samples: int = 1
    min_shoulder: float = 1e-3
    scale_range: tuple[float, float] = (0.5, 1.5)
    rotate_deg: float = 15.0
    noise_std: float = 0.01
    crop_range: tuple[float, float] = (0.5, 1.0)

    checkpoint_dir: Path = Path("checkpoints")
    # Full training state to continue from. None trains from scratch.
    resume: Path | None = None
    tag: str = "vsl"
    # Macro F1 is the default selector because accuracy on hundreds of skewed
    # classes can climb while most of the tail stays unlearned.
    select_by: str = "macro_f1"
    # Leak diagnostic: zero xyz and keep validity. On 422 classes chance is
    # 0.24%, so anything near 5% means the model can read the holes instead of
    # the handshape.
    mask_only: bool = False
    model_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.resume is not None:
            self.resume = Path(self.resume)
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.parallel not in PARALLEL_MODES:
            raise ValueError(f"parallel must be one of {PARALLEL_MODES}, got {self.parallel!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.select_by not in SELECTION_METRICS:
            raise ValueError(f"select_by must be one of {SELECTION_METRICS}, got {self.select_by!r}")
        if self.frames < 1:
            raise ValueError(f"frames must be positive, got {self.frames}")
        if self.micro_batch < 1:
            raise ValueError(f"micro_batch must be positive, got {self.micro_batch}")
        if self.accumulate < 1:
            raise ValueError(f"accumulate must be positive, got {self.accumulate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError(f"warmup_epochs must be in [0, {self.epochs}), got {self.warmup_epochs}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be non-negative, got {self.grad_clip}")

    def effective_batch(self, world_size: int = 1) -> int:
        return self.micro_batch * self.accumulate * world_size

    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.tag}_{self.model}_best.pth"

    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.tag}_{self.model}_last.pth"
