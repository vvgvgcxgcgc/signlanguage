"""AAGCN: adaptive topology plus spatial, temporal, and channel attention.

Every partition combines three matrices: the static graph from models.graph, a
freely learned offset, and one inferred from the clip itself. A wrong edge in
the declared skeleton can therefore be unlearned during training, which is the
reason to keep this model around after the hardcoded-index graph of the first
attempt turned out to be wrong. The price is roughly twice the parameters of
CTR-GCN for the same block layout.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import MaskedSkeletonEmbed, downsample_mask, init_weights, masked_global_pool
from models.graph import SkeletonGraph
from models.stgcn import BLOCKS


class AdaptiveGraphConv(nn.Module):
    """Graph convolution whose topology is static plus learned plus inferred.

    `alpha` starts at zero so the inferred part contributes nothing at
    initialization, and the learned offset starts near zero, which leaves the
    declared skeleton in charge until the gradients say otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        embed_divisor: int = 4,
    ) -> None:
        super().__init__()
        self.partitions = int(adjacency.shape[0])
        inner = max(out_channels // embed_divisor, 1)
        self.register_buffer("static", adjacency.clone())
        self.offset = nn.Parameter(torch.full_like(adjacency, 1e-6))
        self.query = nn.ModuleList(
            [nn.Conv2d(in_channels, inner, kernel_size=1) for _ in range(self.partitions)]
        )
        self.key = nn.ModuleList(
            [nn.Conv2d(in_channels, inner, kernel_size=1) for _ in range(self.partitions)]
        )
        self.project = nn.Conv2d(in_channels, out_channels * self.partitions, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, frames, joints = features.shape
        value = self.project(features).view(batch, self.partitions, -1, frames, joints)
        out = None
        for index in range(self.partitions):
            query = self.query[index](features)
            key = self.key[index](features)
            scores = torch.einsum("bctu,bctv->buv", query, key) / (query.shape[1] * frames)
            # The summed index is the source, so normalize over it to match the
            # column-normalized static adjacency.
            inferred = torch.softmax(scores, dim=-2)
            topology = inferred * self.alpha + self.static[index] + self.offset[index]
            partial = torch.einsum("bctu,buv->bctv", value[:, index], topology)
            out = partial if out is None else out + partial
        return F.relu(self.norm(out) + self.down(features))


class SpatialTemporalChannelAttention(nn.Module):
    """Three residual sigmoid gates: over joints, over frames, over channels."""

    def __init__(self, channels: int, kernel: int = 9, reduction: int = 2) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd, got {kernel}")
        inner = max(channels // reduction, 1)
        self.joint = nn.Conv1d(channels, 1, kernel, padding=kernel // 2)
        self.frame = nn.Conv1d(channels, 1, kernel, padding=kernel // 2)
        self.squeeze = nn.Linear(channels, inner)
        self.excite = nn.Linear(inner, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.joint(features.mean(dim=2)))
        features = features * gate.unsqueeze(2) + features
        gate = torch.sigmoid(self.frame(features.mean(dim=3)))
        features = features * gate.unsqueeze(3) + features
        pooled = features.mean(dim=(2, 3))
        gate = torch.sigmoid(self.excite(F.relu(self.squeeze(pooled))))
        return features * gate.unsqueeze(-1).unsqueeze(-1) + features


class AAGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int,
        dropout: float,
        temporal_kernel: int = 9,
        embed_divisor: int = 4,
        attention: bool = True,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError(f"temporal_kernel must be odd, got {temporal_kernel}")
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, adjacency, embed_divisor)
        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(temporal_kernel // 2, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.attention = (
            SpatialTemporalChannelAttention(out_channels) if attention else nn.Identity()
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.residual(features)
        out = self.drop(self.norm(self.tcn(self.gcn(features))))
        return self.attention(F.relu(out + residual))


class AAGCN(nn.Module):
    """Input is (B, T, J, 4), output is class logits."""

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        embed_channels: int = 64,
        blocks: Sequence[tuple[int, int]] = BLOCKS,
        dropout: float = 0.2,
        attention: bool = True,
    ) -> None:
        super().__init__()
        adjacency = torch.from_numpy(graph.adjacency.copy())
        self.embed = MaskedSkeletonEmbed(
            graph.num_joints, embed_channels, torch.from_numpy(graph.parents)
        )
        channels = embed_channels
        layers = []
        strides = []
        for out_channels, stride in blocks:
            layers.append(
                AAGCNBlock(
                    channels,
                    out_channels,
                    adjacency,
                    stride,
                    dropout,
                    attention=attention,
                )
            )
            strides.append(int(stride))
            channels = out_channels
        self.blocks = nn.ModuleList(layers)
        self.strides = tuple(strides)
        self.head = nn.Linear(channels, num_classes)
        self.apply(init_weights)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        features, mask = self.embed(clip)
        for block, stride in zip(self.blocks, self.strides):
            features = block(features)
            mask = downsample_mask(mask, stride)
        return self.head(masked_global_pool(features, mask))
