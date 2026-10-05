"""Encoder-only transformer over flattened joint embeddings, in the SPOTER mould.

Three things differ from the first attempt. The positional table is learned and
sized to the fixed clip length instead of a 5000-row sinusoidal buffer that
made the checkpoint larger than the model. Classification reads a class token
rather than a mean over frames, so a long still pose cannot wash out the few
frames that carry the sign. And frames with nothing observed are excluded from
attention through the key padding mask.

Kept as the one model outside the graph family: its mistakes differ from the
graph models', which is what makes an ensemble worth anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.embed import MaskedSkeletonEmbed, init_weights
from models.graph import SkeletonGraph


class SignTransformer(nn.Module):
    """Input is (B, T, J, 4) with T no longer than `max_frames`."""

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        max_frames: int = 64,
        embed_channels: int = 16,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_frames = int(max_frames)
        self.embed = MaskedSkeletonEmbed(
            graph.num_joints, embed_channels, torch.from_numpy(graph.parents)
        )
        self.project = nn.Linear(embed_channels * graph.num_joints, d_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, self.max_frames + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # Only the pieces outside the encoder are re-initialized, so the
        # encoder keeps the defaults its own reset_parameters chose.
        self.embed.apply(init_weights)
        init_weights(self.project)
        init_weights(self.head)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        features, mask = self.embed(clip)
        batch, _, frames, _ = features.shape
        if frames > self.max_frames:
            raise ValueError(f"clip has {frames} frames but max_frames is {self.max_frames}")

        tokens = self.project(features.permute(0, 2, 1, 3).reshape(batch, frames, -1))
        tokens = torch.cat((self.class_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = tokens + self.position[:, : frames + 1]

        # A frame where every joint is a hole carries only missing tokens. The
        # class token is always kept, so no row can be fully masked.
        observed = mask.amax(dim=3).squeeze(1) > 0
        keep = torch.cat((observed.new_ones((batch, 1)), observed), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~keep)
        return self.head(self.norm(encoded[:, 0]))
