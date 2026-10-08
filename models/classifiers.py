"""Checkpoint saving, loading, and prediction for the models in this package.

Nothing here defines an architecture. models.factory does that, so the graph
and the joint layout used at inference can no longer drift away from the ones
used in training, which is exactly how the previous hand-copied adjacency went
wrong. A checkpoint therefore has to record which joint set and clip length it
was trained on, and the loader rebuilds from that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models.factory import MODEL_NAMES, build_model, count_parameters, pick_device

logger = logging.getLogger(__name__)

__all__ = [
    "LoadedModel",
    "find_checkpoints",
    "load_models",
    "pick_device",
    "predict_gloss",
    "save_checkpoint",
]

CHECKPOINT_SUFFIX = "_best.pth"


@dataclass(frozen=True)
class LoadedModel:
    """A model in eval mode together with the input shape it expects."""

    name: str
    module: nn.Module
    is_leg: bool
    max_frames: int
    num_joints: int


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    name: str,
    num_classes: int,
    is_leg: bool,
    max_frames: int,
    **extra: Any,
) -> Path:
    """Write a state dict plus the metadata the loader needs to rebuild it.

    Any extra keyword, such as the epoch or the validation score, is stored
    alongside. A DataParallel wrapper is unwrapped so the keys stay clean.
    """
    if name not in MODEL_NAMES:
        raise KeyError(f"unknown model {name!r}, expected one of {', '.join(MODEL_NAMES)}")
    module = model.module if hasattr(model, "module") else model
    state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    payload = {
        "model_name": name,
        "num_classes": int(num_classes),
        "is_leg": bool(is_leg),
        "max_frames": int(max_frames),
        "state_dict": state,
        **extra,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    logger.info("Wrote %s checkpoint to %s", name, target)
    return target


def _checkpoint_model_name(path: Path) -> Optional[str]:
    stem = path.stem
    for name in MODEL_NAMES:
        if stem.endswith(f"_{name}_best") or stem == name:
            return name
    return None


def find_checkpoints(directory: Path | str) -> dict[str, Path]:
    """Map model name to checkpoint for every recognised file in `directory`."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {root}")
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*.pth")):
        # `_last.pth` is the resume snapshot, not a model to serve.
        if path.name.endswith("_last.pth"):
            continue
        name = _checkpoint_model_name(path)
        if name is None:
            logger.warning("Skipping checkpoint with no known model name: %s", path.name)
            continue
        found[name] = path
    if not found:
        raise FileNotFoundError(f"no recognised checkpoint in {root}")
    return found


def _read_payload(path: Path, device: torch.device) -> dict:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"unrecognized checkpoint format: {path}")
    if "state_dict" not in payload:
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return payload


def load_models(
    checkpoint_dir: Path | str,
    num_classes: int,
    device: torch.device,
    names: Optional[Sequence[str]] = None,
    is_leg: bool = False,
    max_frames: int = 64,
) -> dict[str, LoadedModel]:
    """Rebuild and load every checkpoint found, or only the ones named.

    `is_leg` and `max_frames` are fallbacks for a checkpoint that does not
    record them. A checkpoint that does record them wins, and a mismatch with
    its own class count is an error rather than a silent reshape.
    """
    checkpoints = find_checkpoints(checkpoint_dir)
    if names is not None:
        unknown = set(names) - set(checkpoints)
        if unknown:
            raise KeyError(f"no checkpoint for {', '.join(sorted(unknown))} in {checkpoint_dir}")
        checkpoints = {name: checkpoints[name] for name in names}

    loaded: dict[str, LoadedModel] = {}
    for name, path in checkpoints.items():
        payload = _read_payload(path, device)
        saved_classes = int(payload.get("num_classes", num_classes))
        if saved_classes != num_classes:
            raise ValueError(
                f"{path.name} was trained on {saved_classes} classes but {num_classes} were asked for"
            )
        model_is_leg = bool(payload.get("is_leg", is_leg))
        model_frames = int(payload.get("max_frames", max_frames))
        module = build_model(name, num_classes, is_leg=model_is_leg, max_frames=model_frames)
        state = {key.replace("module.", ""): value for key, value in payload["state_dict"].items()}
        module.load_state_dict(state)
        module.to(device).eval()
        num_joints = module.embed.num_joints
        loaded[name] = LoadedModel(
            name=name,
            module=module,
            is_leg=model_is_leg,
            max_frames=model_frames,
            num_joints=num_joints,
        )
        logger.info(
            "Ready %s from %s: %d joints, %d frames, %s parameters, epoch=%s val=%s",
            name,
            path.name,
            num_joints,
            model_frames,
            f"{count_parameters(module):,}",
            payload.get("epoch"),
            payload.get("val_acc"),
        )
    return loaded


def predict_gloss(
    clip: np.ndarray,
    models: dict[str, LoadedModel],
    glosses: Sequence[str],
    device: torch.device,
) -> dict[str, object]:
    """Run every loaded model on one (T, J, 4) clip and average the probabilities.

    Averaging beats picking the most confident model, which only ever rewards
    whichever head is worst calibrated. The per-model top-1 is still reported
    so a disagreement stays visible.
    """
    if not models:
        raise ValueError("no models to predict with")
    if clip.ndim != 3 or clip.shape[-1] != 4:
        raise ValueError(f"clip must be (T, J, 4), got {clip.shape}")
    for loaded in models.values():
        if clip.shape[1] != loaded.num_joints:
            raise ValueError(
                f"{loaded.name} expects {loaded.num_joints} joints, clip has {clip.shape[1]}"
            )
        if clip.shape[0] > loaded.max_frames:
            raise ValueError(
                f"{loaded.name} expects at most {loaded.max_frames} frames, clip has {clip.shape[0]}"
            )

    batch = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).unsqueeze(0).to(device)
    per_model: list[dict[str, object]] = []
    total: Optional[torch.Tensor] = None
    with torch.no_grad():
        for name, loaded in models.items():
            probabilities = torch.softmax(loaded.module(batch), dim=1)[0]
            total = probabilities.clone() if total is None else total + probabilities
            index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[index].item())
            per_model.append(
                {
                    "model": name,
                    "gloss": glosses[index],
                    "confidence": confidence,
                    "index": index,
                }
            )
            logger.info("%s -> %s (%.4f)", name, glosses[index], confidence)

    averaged = total / len(models)
    index = int(torch.argmax(averaged).item())
    return {
        "gloss": glosses[index],
        "confidence": float(averaged[index].item()),
        "index": index,
        "models": sorted(models),
        "predictions": per_model,
    }
