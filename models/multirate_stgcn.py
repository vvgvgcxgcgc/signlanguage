"""Three-rate ST-GCN with attention pooling across temporal resolutions.

The 32-frame pathway keeps the full ST-GCN width. The 64-frame and 96-frame
pathways use one half and one quarter of that width, respectively, so the
high-rate branches preserve motion detail without tripling the compute.
Each pathway produces one masked pooled vector. A learned fusion query attends
to the three projected vectors and produces the representation classified by
the final linear layer.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.embed import MaskedSkeletonEmbed, downsample_mask, init_weights, masked_global_pool
from models.graph import SkeletonGraph
from models.stgcn import BLOCKS, STGCNBlock

TEMPORAL_LENGTHS = (32, 64, 96)
WIDTH_DIVISORS = (1, 2, 4)


class STGCNPathway(nn.Module):
    """One independently parameterized ST-GCN temporal pathway."""

    def __init__(
        self,
        graph: SkeletonGraph,
        embed_channels: int,
        blocks: Sequence[tuple[int, int]],
        width_divisor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if width_divisor < 1:
            raise ValueError(f"width_divisor must be positive, got {width_divisor}")
        if embed_channels % width_divisor != 0:
            raise ValueError(
                f"embed_channels must be divisible by {width_divisor}, got {embed_channels}"
            )

        parents = torch.from_numpy(graph.parents)
        channels = embed_channels // width_divisor
        self.embed = MaskedSkeletonEmbed(graph.num_joints, channels, parents)
        layers: list[nn.Module] = []
        strides: list[int] = []
        for full_width, stride in blocks:
            if full_width % width_divisor != 0:
                raise ValueError(
                    f"block width {full_width} is not divisible by {width_divisor}"
                )
            out_channels = full_width // width_divisor
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
            channels = out_channels
            strides.append(int(stride))
        self.blocks = nn.ModuleList(layers)
        self.strides = tuple(strides)
        self.out_channels = channels

    def forward(self, clip: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        features, mask = self.embed(clip)
        for block, stride in zip(self.blocks, self.strides):
            features = block(features, adjacency)
            mask = downsample_mask(mask, stride)
        return masked_global_pool(features, mask)


class MultiRateAttentionSTGCN(nn.Module):
    """ST-GCN pathways at T=32, 64, and 96 followed by attention fusion."""

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        embed_channels: int = 64,
        blocks: Sequence[tuple[int, int]] = BLOCKS,
        fusion_dim: int = 256,
        attention_heads: int = 4,
        fusion_hidden: int = 512,
        dropout: float = 0.2,
        fusion_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_dim % attention_heads != 0:
            raise ValueError(
                f"fusion_dim must be divisible by attention_heads, got "
                f"{fusion_dim} and {attention_heads}"
            )
        checked_blocks = tuple((int(width), int(stride)) for width, stride in blocks)
        if not checked_blocks:
            raise ValueError("blocks must not be empty")

        self.pathways = nn.ModuleList(
            [
                STGCNPathway(
                    graph,
                    embed_channels,
                    checked_blocks,
                    width_divisor,
                    dropout,
                )
                for width_divisor in WIDTH_DIVISORS
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pathway.out_channels, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                )
                for pathway in self.pathways
            ]
        )
        self.rate_embedding = nn.Parameter(
            torch.empty(1, len(TEMPORAL_LENGTHS), fusion_dim)
        )
        self.fusion_token = nn.Parameter(torch.empty(1, 1, fusion_dim))
        self.token_norm = nn.LayerNorm(fusion_dim)
        self.attention = nn.MultiheadAttention(
            fusion_dim,
            attention_heads,
            dropout=fusion_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(fusion_dropout)
        self.attention_norm = nn.LayerNorm(fusion_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_dim),
            nn.Dropout(fusion_dropout),
        )
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Linear(fusion_dim, num_classes)
        self.register_buffer("adjacency", torch.from_numpy(graph.adjacency.copy()))

        self.apply(init_weights)
        nn.init.trunc_normal_(self.rate_embedding, std=0.02)
        nn.init.trunc_normal_(self.fusion_token, std=0.02)

    @staticmethod
    def _interpolate_clip(clip: torch.Tensor, length: int) -> torch.Tensor:
        """Validity-aware interpolation used for single-tensor inference."""
        frames = int(clip.shape[1])
        if frames < 1:
            raise ValueError("clip must contain at least one frame")
        if frames == length:
            return clip
        if length == 1:
            return clip[:, :1]
        if frames == 1:
            return clip.expand(-1, length, -1, -1)

        positions = torch.linspace(0, frames - 1, length, device=clip.device)
        low = positions.floor().to(torch.long)
        high = positions.ceil().to(torch.long)
        weight = (positions - low).to(clip.dtype).view(1, length, 1, 1)
        low_frames = clip.index_select(1, low)
        high_frames = clip.index_select(1, high)
        mixed = (1.0 - weight) * low_frames + weight * high_frames

        low_seen = low_frames[..., 3:] > 0
        high_seen = high_frames[..., 3:] > 0
        xyz = mixed[..., :3]
        xyz = torch.where(low_seen & ~high_seen, low_frames[..., :3], xyz)
        xyz = torch.where(high_seen & ~low_seen, high_frames[..., :3], xyz)
        xyz = torch.where(low_seen | high_seen, xyz, torch.zeros_like(xyz))
        return torch.cat((xyz, mixed[..., 3:]), dim=-1)

    def _branches(
        self, clips: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(clips):
            if clips.ndim != 4 or clips.shape[-1] != 4:
                raise ValueError(f"expected a (B, T, J, 4) clip, got {tuple(clips.shape)}")
            return tuple(
                self._interpolate_clip(clips, length) for length in TEMPORAL_LENGTHS
            )

        branches = tuple(clips)
        if len(branches) != len(TEMPORAL_LENGTHS):
            raise ValueError(
                f"expected {len(TEMPORAL_LENGTHS)} temporal branches, got {len(branches)}"
            )
        for branch, length in zip(branches, TEMPORAL_LENGTHS):
            if branch.ndim != 4 or branch.shape[-1] != 4:
                raise ValueError(
                    f"expected branch T={length} to have shape (B, T, J, 4), "
                    f"got {tuple(branch.shape)}"
                )
            if int(branch.shape[1]) != length:
                raise ValueError(
                    f"expected branch T={length}, got T={int(branch.shape[1])}"
                )
        return branches

    def forward(self, clips: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        branches = self._branches(clips)
        vectors = [
            pathway(branch, self.adjacency)
            for pathway, branch in zip(self.pathways, branches)
        ]
        tokens = torch.stack(
            [
                projection(vector)
                for projection, vector in zip(self.projections, vectors)
            ],
            dim=1,
        )
        tokens = self.token_norm(tokens + self.rate_embedding)

        query = self.fusion_token.expand(tokens.shape[0], -1, -1)
        attended, _weights = self.attention(
            query,
            tokens,
            tokens,
            need_weights=False,
        )
        fused = self.attention_norm(query + self.attention_dropout(attended))
        fused = self.output_norm(fused + self.feed_forward(fused))
        return self.head(fused[:, 0])
