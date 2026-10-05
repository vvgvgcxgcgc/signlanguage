"""ST-GCN with the three-partition spatial graph and learnable edge importance.

Each block is a graph convolution over the partitions followed by a 9x1
temporal convolution, which is the original design. Two things that were wrong
in the first attempt are fixed here: the graph comes from models.graph, where
edges are declared by landmark name instead of by hardcoded index, and the
single shared adjacency is replaced by the three-partition spatial
configuration scaled by a per-block learnable importance.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import MaskedSkeletonEmbed, downsample_mask, init_weights, masked_global_pool
from models.graph import SkeletonGraph

# (out_channels, temporal_stride) for each block. Time goes 64 -> 32 -> 16.
BLOCKS = (
    (64, 1), (64, 1), (64, 1),
    (128, 2), (128, 1), (128, 1),
    (256, 2), (256, 1), (256, 1),
)


class GraphConv(nn.Module):
    """Project to one feature set per partition, then aggregate over neighbours."""

    def __init__(self, in_channels: int, out_channels: int, partitions: int) -> None:
        super().__init__()
        self.partitions = int(partitions)
        self.project = nn.Conv2d(in_channels, out_channels * self.partitions, kernel_size=1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, _, frames, joints = features.shape
        value = self.project(features).view(batch, self.partitions, -1, frames, joints)
        # adjacency is [source, target] and column-normalized, so the summed
        # index is the source. See models.graph.spatial_adjacency.
        return torch.einsum("bkctu,kuv->bctv", value, adjacency)


class STGCNBlock(nn.Module):
    """Graph convolution, temporal convolution, residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        partitions: int,
        num_joints: int,
        stride: int,
        dropout: float,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError(f"temporal_kernel must be odd, got {temporal_kernel}")
        self.gcn = GraphConv(in_channels, out_channels, partitions)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(temporal_kernel // 2, 0),
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.edge_importance = nn.Parameter(torch.ones(partitions, num_joints, num_joints))
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(features)
        out = F.relu(self.norm1(self.gcn(features, adjacency * self.edge_importance)))
        out = self.drop(self.norm2(self.tcn(out)))
        return F.relu(out + residual)


class STGCN(nn.Module):
    """Input is (B, T, J, 4), output is class logits."""

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        embed_channels: int = 64,
        blocks: Sequence[tuple[int, int]] = BLOCKS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.from_numpy(graph.adjacency.copy()))
        self.embed = MaskedSkeletonEmbed(
            graph.num_joints, embed_channels, torch.from_numpy(graph.parents)
        )
        channels = embed_channels
        layers = []
        strides = []
        for out_channels, stride in blocks:
            layers.append(
                STGCNBlock(
                    channels,
                    out_channels,
                    graph.num_partitions,
                    graph.num_joints,
                    stride,
                    dropout,
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
            features = block(features, self.adjacency)
            mask = downsample_mask(mask, stride)
        return self.head(masked_global_pool(features, mask))
