"""One entry point for building any of the five models.

`is_leg` must match the flag the KeypointDataset was built with: False keeps
the 68 joints that preprocess.dataset.drop_legs leaves, True keeps all 76.
Every model takes (B, T, J, 4) and returns logits, and only the transformer
cares what T is.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn

from models.aagcn import AAGCN
from models.ctrgcn import CTRGCN
from models.graph import SkeletonGraph, build_graph
from models.mstcn import MSTCN
from models.stgcn import STGCN
from models.transformer import SignTransformer

logger = logging.getLogger(__name__)

MODEL_NAMES = ("MS-TCN", "ST-GCN", "CTR-GCN", "AAGCN", "Transformer")

_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "MS-TCN": MSTCN,
    "ST-GCN": STGCN,
    "CTR-GCN": CTRGCN,
    "AAGCN": AAGCN,
    "Transformer": SignTransformer,
}

# Only the transformer has a frame-dependent parameter, the positional table.
_NEEDS_MAX_FRAMES = frozenset({"Transformer"})


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(
    name: str,
    num_classes: int,
    is_leg: bool = False,
    max_frames: int = 64,
    graph: SkeletonGraph | None = None,
    **overrides: Any,
) -> nn.Module:
    """Build one model by name. Extra keywords go to that model's constructor."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown model {name!r}, expected one of {', '.join(MODEL_NAMES)}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    skeleton = build_graph(is_leg) if graph is None else graph
    if graph is not None and graph.is_leg != is_leg:
        raise ValueError(f"graph was built with is_leg={graph.is_leg} but is_leg={is_leg} was asked for")

    kwargs: dict[str, Any] = dict(overrides)
    if name in _NEEDS_MAX_FRAMES:
        kwargs.setdefault("max_frames", max_frames)
    model = _BUILDERS[name](skeleton, num_classes, **kwargs)
    logger.info(
        "Built %s with %d joints, %d classes, %s parameters",
        name,
        skeleton.num_joints,
        num_classes,
        f"{count_parameters(model):,}",
    )
    return model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
