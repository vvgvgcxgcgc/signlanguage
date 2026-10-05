"""Multi-scale temporal CNN over flattened joint embeddings.

Joints are embedded per frame, flattened into one channel axis, and pushed
through residual dilated convolutions. With dilations 1, 2, 4, 8, 16 and two
kernel-3 convolutions per block the receptive field is 125 frames, so a
64-frame clip is covered end to end instead of the 29 frames the first
three-layer attempt could reach. No graph is involved, which is the point:
this is the cheap number every graph model has to beat.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import MaskedSkeletonEmbed, init_weights, masked_temporal_pool
from models.graph import SkeletonGraph

DILATIONS = (1, 2, 4, 8, 16)


class ResidualDilatedBlock(nn.Module):
    """Two dilated convolutions at the same rate, with a residual path."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        out = self.drop(F.gelu(self.norm1(self.conv1(sequence))))
        out = self.norm2(self.conv2(out))
        return F.gelu(out + sequence)


class MSTCN(nn.Module):
    """Temporal-only baseline. Input is (B, T, J, 4), output is class logits."""

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        embed_channels: int = 16,
        channels: int = 128,
        dilations: Sequence[int] = DILATIONS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = MaskedSkeletonEmbed(
            graph.num_joints, embed_channels, torch.from_numpy(graph.parents)
        )
        self.project = nn.Conv1d(embed_channels * graph.num_joints, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.blocks = nn.ModuleList(
            [ResidualDilatedBlock(channels, int(dilation), dropout) for dilation in dilations]
        )
        self.head = nn.Linear(channels, num_classes)
        self.apply(init_weights)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(2 * 2 * block.conv1.dilation[0] for block in self.blocks)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        features, mask = self.embed(clip)
        batch, _, frames, _ = features.shape
        flat = features.permute(0, 1, 3, 2).reshape(batch, -1, frames)
        flat = F.gelu(self.norm(self.project(flat)))
        for block in self.blocks:
            flat = block(flat)
        return self.head(masked_temporal_pool(flat, mask.amax(dim=3)))
