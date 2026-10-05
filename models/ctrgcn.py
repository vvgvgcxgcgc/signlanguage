"""CTR-GCN: channel-wise topology refinement with a multi-scale temporal module.

One shared adjacency forces every feature channel to use the same joint
relations. That is wrong for sign language, where the link between index and
middle finger matters to the channels describing handshape and means nothing
to the channels describing where the arm is in space. Each partition here
refines the shared topology per output channel from the pairwise difference
between joint embeddings.

The temporal half is four parallel views instead of one 9x1 convolution, which
is where most of the parameter saving against ST-GCN comes from.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import MaskedSkeletonEmbed, downsample_mask, init_weights, masked_global_pool
from models.graph import SkeletonGraph
from models.stgcn import BLOCKS

TEMPORAL_DILATIONS = (1, 2)


class ChannelTopologyRefine(nn.Module):
    """One CTR-GC branch: shared topology plus a per-channel refinement.

    Query and key are pooled over time, so the refinement is one topology per
    clip rather than per frame. `alpha` starts at zero, which makes the branch
    a plain graph convolution on the shared topology at initialization.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.query = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.key = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.expand = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        query = self.query(features).mean(dim=2)
        key = self.key(features).mean(dim=2)
        difference = torch.tanh(query.unsqueeze(-1) - key.unsqueeze(-2))
        topology = self.expand(difference) * self.alpha + adjacency
        return torch.einsum("bctu,bcuv->bctv", self.value(features), topology)


class CTRGraphConv(nn.Module):
    """One refinement branch per partition, summed, with a projection residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        partitions = int(adjacency.shape[0])
        self.branches = nn.ModuleList(
            [ChannelTopologyRefine(in_channels, out_channels, reduction) for _ in range(partitions)]
        )
        self.topology = nn.Parameter(adjacency.clone())
        self.norm = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = None
        for index, branch in enumerate(self.branches):
            partial = branch(features, self.topology[index])
            out = partial if out is None else out + partial
        return F.relu(self.norm(out) + self.down(features))


class MultiScaleTemporalConv(nn.Module):
    """Four temporal views: two dilated 5x1 convolutions, a max pool, and a 1x1.

    Every branch produces the same frame count for a given stride, so they can
    be concatenated back to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilations: Sequence[int] = TEMPORAL_DILATIONS,
    ) -> None:
        super().__init__()
        branch_count = len(dilations) + 2
        if out_channels % branch_count != 0:
            raise ValueError(f"out_channels must be divisible by {branch_count}, got {out_channels}")
        width = out_channels // branch_count

        branches: list[nn.Module] = []
        for dilation in dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, width, kernel_size=1),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                    nn.Conv2d(
                        width,
                        width,
                        kernel_size=(5, 1),
                        stride=(stride, 1),
                        padding=(2 * int(dilation), 0),
                        dilation=(int(dilation), 1),
                    ),
                    nn.BatchNorm2d(width),
                )
            )
        branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=1),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(width),
            )
        )
        branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(width),
            )
        )
        self.branches = nn.ModuleList(branches)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(features) for branch in self.branches], dim=1)


class CTRGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.gcn = CTRGraphConv(in_channels, out_channels, adjacency, reduction)
        self.tcn = MultiScaleTemporalConv(out_channels, out_channels, stride)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(self.tcn(self.gcn(features)) + self.residual(features))


class CTRGCN(nn.Module):
    """Input is (B, T, J, 4), output is class logits."""

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        embed_channels: int = 64,
        blocks: Sequence[tuple[int, int]] = BLOCKS,
        reduction: int = 8,
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
            layers.append(CTRGCNBlock(channels, out_channels, adjacency, stride, reduction))
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
