"""Start-pose vs action-pose classifier for the (W, J, 2) windows of preprocess.pose_state.

The window is short, so there is no temporal structure worth a graph or a
temporal conv. The model reads the last frame's pose plus every frame-to-frame
difference, flattened, which is what a kernel-W temporal conv would see.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseStateMLP(nn.Module):
    """Input (B, W, J, C), output logits (B, num_classes).

    The input BatchNorm puts position and velocity on one scale, since body
    motion between two frames is an order of magnitude smaller than position.
    """

    def __init__(
        self,
        num_joints: int,
        window: int = 3,
        num_coords: int = 2,
        num_classes: int = 2,
        hidden: tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        self.num_joints = int(num_joints)
        self.window = int(window)
        self.num_coords = int(num_coords)
        in_features = self.window * self.num_joints * self.num_coords
        layers: list[nn.Module] = [nn.BatchNorm1d(in_features)]
        width = in_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.GELU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)
        self.config = {
            "num_joints": self.num_joints,
            "window": self.window,
            "num_coords": self.num_coords,
            "num_classes": int(num_classes),
            "hidden": tuple(int(size) for size in hidden),
            "dropout": float(dropout),
        }

    @classmethod
    def from_checkpoint(cls, path, device: torch.device | str = "cpu") -> tuple["PoseStateMLP", dict]:
        """Rebuild the model from a checkpoint written by training.pose_state. Returns (model, payload)."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval(), payload

    def features(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4 or windows.shape[1:] != (self.window, self.num_joints, self.num_coords):
            raise ValueError(
                f"expected (B, {self.window}, {self.num_joints}, {self.num_coords}), got {tuple(windows.shape)}"
            )
        position = windows[:, -1].flatten(1)
        velocity = (windows[:, 1:] - windows[:, :-1]).flatten(1)
        return torch.cat((position, velocity), dim=1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(windows))
