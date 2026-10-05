"""Shared input adaptor for the 4-channel keypoint clips, plus mask plumbing.

Channel 3 is validity. After preprocess.dataset.interpolate_clip it is 0 only
where no frame observed the joint, and that is the only case where xyz is a
placeholder sitting on the shoulder-midpoint origin. Any validity above 0 means
xyz is a real observation, possibly carried over from the nearest observed
frame. So the gate that swaps in the missing token is hard, and the fractional
value rides along as an ordinary input channel for the network to discount on
its own.

Never derive the mask from xyz. The neck joint is defined as the shoulder
midpoint, which is exactly the normalization origin, so it sits at (0, 0, 0)
with validity 1 in every frame of every clip.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# xyz, frame-to-frame motion, vector to the parent joint, and validity.
GEOMETRY_CHANNELS = 10


class SkeletonFeatures(nn.Module):
    """(B, T, J, 4) -> geometry (B, 10, T, J) and hard mask (B, 1, T, J).

    Motion and bone are gated by the mask of both endpoints, otherwise a hand
    that reappears produces a velocity spike out of the origin. The gate is
    hard rather than the fractional validity, because a fractional value still
    marks a real observation and scaling by it would shrink genuine motion
    between two observed frames.
    """

    def __init__(self, parents: torch.Tensor) -> None:
        super().__init__()
        if parents.ndim != 1:
            raise ValueError(f"parents must be one index per joint, got shape {tuple(parents.shape)}")
        self.register_buffer("parents", parents.to(torch.long))

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clip.ndim != 4 or clip.shape[-1] != 4:
            raise ValueError(f"expected a (B, T, J, 4) clip, got {tuple(clip.shape)}")
        if clip.shape[2] != self.parents.shape[0]:
            raise ValueError(
                f"clip has {clip.shape[2]} joints but the graph has {self.parents.shape[0]}"
            )

        xyz = clip[..., :3].permute(0, 3, 1, 2)
        validity = clip[..., 3:].permute(0, 3, 1, 2)
        mask = (validity > 0).to(xyz.dtype)

        motion = torch.zeros_like(xyz)
        motion[:, :, 1:] = (xyz[:, :, 1:] - xyz[:, :, :-1]) * mask[:, :, 1:] * mask[:, :, :-1]

        parent_xyz = xyz[..., self.parents]
        parent_mask = mask[..., self.parents]
        bone = (xyz - parent_xyz) * mask * parent_mask

        geometry = torch.cat((xyz * mask, motion, bone, validity), dim=1)
        return geometry, mask


class MaskedSkeletonEmbed(nn.Module):
    """Project the geometry channels and replace every hole with a learned token.

    The token is per joint, because a missing right wrist and a missing nose
    are not the same event. It only applies where validity is exactly 0.
    """

    def __init__(self, num_joints: int, out_channels: int, parents: torch.Tensor) -> None:
        super().__init__()
        self.features = SkeletonFeatures(parents)
        self.project = nn.Conv2d(GEOMETRY_CHANNELS, out_channels, kernel_size=1)
        self.missing = nn.Parameter(torch.zeros(1, out_channels, 1, num_joints))
        self.norm = nn.BatchNorm2d(out_channels)
        self.out_channels = int(out_channels)
        self.num_joints = int(num_joints)
        nn.init.trunc_normal_(self.missing, std=0.02)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geometry, mask = self.features(clip)
        embedded = self.project(geometry)
        embedded = embedded * mask + self.missing * (1.0 - mask)
        return self.norm(embedded), mask


def downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    """Shrink a (B, 1, T, J) mask along time to follow a strided block.

    Max pooling keeps a position valid when any frame it absorbed was valid,
    which matches what the strided convolution above it can actually see.
    """
    if stride == 1:
        return mask
    return F.max_pool2d(mask, kernel_size=(stride, 1), stride=(stride, 1))


def masked_global_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average (B, C, T, J) over time and joints, counting observed positions only.

    A plain mean divides by T * J no matter how much of the clip was a hole, so
    a clip where one hand was never detected would hand the classifier a
    feature scaled down by the fraction of joints that are tokens. Dividing by
    the real count keeps the scale comparable across clips.
    """
    total = (features * mask).sum(dim=(2, 3))
    count = mask.sum(dim=(2, 3)).clamp(min=1.0)
    return total / count


def masked_temporal_pool(sequence: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Average (B, C, T) over time, counting frames that observed any joint."""
    total = (sequence * frame_mask).sum(dim=2)
    count = frame_mask.sum(dim=2).clamp(min=1.0)
    return total / count


def init_weights(module: nn.Module) -> None:
    """Kaiming for convolutions, unit scale for norms, truncated normal for linears."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
