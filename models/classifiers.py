"""ST-GCN and TCN from the VSL400 training notebook, plus checkpoint loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess.keypoints import FEATURE_DIM, N_JOINTS

logger = logging.getLogger(__name__)


def build_mediapipe_adjacency(num_joints: int = N_JOINTS) -> np.ndarray:
    """Same graph the ST-GCN checkpoint was trained with."""
    adjacency = np.zeros((num_joints, num_joints), dtype=np.float32)
    pose_edges = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ]
    for left, right in pose_edges:
        if left < 33 and right < 33:
            adjacency[left, right] = adjacency[right, left] = 1
    hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
    ]
    for left, right in hand_edges:
        if 33 + left < num_joints and 33 + right < num_joints:
            adjacency[33 + left, 33 + right] = adjacency[33 + right, 33 + left] = 1
        if 54 + left < num_joints and 54 + right < num_joints:
            adjacency[54 + left, 54 + right] = adjacency[54 + right, 54 + left] = 1
    if num_joints >= 55:
        adjacency[15, 33] = adjacency[33, 15] = 1
        adjacency[16, 54] = adjacency[54, 16] = 1
    adjacency += np.eye(num_joints)
    degree = np.diag(1.0 / np.sqrt(np.sum(adjacency, axis=1) + 1e-8))
    return (degree @ adjacency @ degree).astype(np.float32)


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("adj", torch.tensor(adjacency, dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.einsum("bctv,co->botv", features, self.weight)
        return torch.einsum("botv,vw->botw", features, self.adj)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray, stride: int = 1) -> None:
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.residual(features)
        features = F.relu(self.bn1(self.gcn(features)))
        features = self.bn2(self.tcn(features))
        return F.relu(features + residual)


class STGCN(nn.Module):
    def __init__(self, num_joints: int, num_classes: int, adjacency: np.ndarray) -> None:
        super().__init__()
        self.data_bn = nn.BatchNorm1d(3 * num_joints)
        self.blocks = nn.ModuleList([
            STGCNBlock(3, 64, adjacency, stride=1),
            STGCNBlock(64, 64, adjacency, stride=1),
            STGCNBlock(64, 128, adjacency, stride=2),
            STGCNBlock(128, 256, adjacency, stride=2),
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _time, flat_dim = features.shape
        joints = flat_dim // 3
        features = features.view(batch, _time, joints, 3).permute(0, 3, 1, 2).contiguous()
        features = features.view(batch, 3 * joints, _time)
        features = self.data_bn(features)
        features = features.view(batch, 3, joints, _time).permute(0, 1, 3, 2)
        for block in self.blocks:
            features = block(features)
        return self.fc(features.mean(dim=(2, 3)))


class TCN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=8, dilation=8)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.permute(0, 2, 1)
        features = self.drop(F.relu(self.bn1(self.conv1(features))))
        features = self.drop(F.relu(self.bn2(self.conv2(features))))
        features = self.drop(F.relu(self.bn3(self.conv3(features))))
        return self.fc(features.mean(dim=2))


def _checkpoint_model_name(path: Path) -> Optional[str]:
    stem = path.stem
    if "_ST-GCN_best" in stem:
        return "ST-GCN"
    if "_TCN_best" in stem:
        return "TCN"
    return None


def find_checkpoints(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(Path(directory).glob("*.pth")):
        name = _checkpoint_model_name(path)
        if name is None:
            logger.warning("Skipping checkpoint with no known model name: %s", path.name)
            continue
        found[name] = path
    missing = {"ST-GCN", "TCN"} - set(found)
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {directory}: {', '.join(sorted(missing))}")
    return found


def _load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        logger.info(
            "Loaded %s epoch=%s val_acc=%s",
            path.name,
            payload.get("epoch"),
            payload.get("val_acc"),
        )
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "ST-GCN":
        adjacency = build_mediapipe_adjacency(N_JOINTS)
        return STGCN(N_JOINTS, num_classes, adjacency)
    if name == "TCN":
        return TCN(FEATURE_DIM, num_classes)
    raise KeyError(f"Unknown model: {name}")


def load_models(
    checkpoint_dir: Path,
    num_classes: int,
    device: torch.device,
    names: Optional[Sequence[str]] = None,
) -> dict[str, nn.Module]:
    loaded: dict[str, nn.Module] = {}
    checkpoints = find_checkpoints(checkpoint_dir)
    selected = checkpoints if names is None else {name: checkpoints[name] for name in names}
    for name, path in selected.items():
        model = build_model(name, num_classes).to(device)
        model.load_state_dict(_load_state_dict(path, device))
        model.eval()
        loaded[name] = model
        logger.info("Ready %s (%s parameters)", name, sum(p.numel() for p in model.parameters()))
    return loaded


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_gloss(
    features: np.ndarray,
    models: dict[str, nn.Module],
    glosses: Sequence[str],
    device: torch.device,
) -> dict[str, object]:
    """Run every loaded model on one (80, 228) window. The gloss is the higher-confidence top-1."""
    if features.shape != (80, FEATURE_DIM):
        raise ValueError(f"features must be (80, {FEATURE_DIM}), got {features.shape}")
    batch = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).unsqueeze(0).to(device)
    per_model = []
    with torch.no_grad():
        for name, model in models.items():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[index].item())
            gloss = glosses[index]
            per_model.append({"model": name, "gloss": gloss, "confidence": confidence, "index": index})
            logger.info("%s -> %s (%.4f)", name, gloss, confidence)

    winner = max(per_model, key=lambda item: float(item["confidence"]))
    return {"gloss": winner["gloss"], "model": winner["model"], "predictions": per_model}
