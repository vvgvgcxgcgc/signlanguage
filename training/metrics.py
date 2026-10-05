"""Confusion-matrix metrics in torch, so there is no scikit-learn dependency.

Accuracy alone on 422 skewed classes says very little, so macro F1 is reported
next to it. Macro F1 averages over every class, counting a class with no
predictions and no support as 0, which is what scikit-learn does with
`average="macro", zero_division=0`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.distributed import Topology, reduce_sum


@dataclass(frozen=True)
class Scores:
    loss: float
    accuracy: float
    top5: float
    macro_f1: float

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} acc={self.accuracy:.4f} "
            f"top5={self.top5:.4f} macroF1={self.macro_f1:.4f}"
        )


class MetricAccumulator:
    """Running confusion matrix plus loss and top-5 tallies.

    Everything is kept on the compute device as integer counts, so a DDP run
    can sum the whole state in three all-reduces and get the same numbers a
    single process would.
    """

    def __init__(self, num_classes: int, device: torch.device) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.device = device
        # MPS has no float64, so the running loss keeps the widest dtype the
        # device supports. Integer counts are exact everywhere.
        loss_dtype = torch.float32 if device.type == "mps" else torch.float64
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
        self.totals = torch.zeros(2, dtype=torch.long, device=device)
        self.loss_sum = torch.zeros(1, dtype=loss_dtype, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1)
        index = targets * self.num_classes + predictions
        self.confusion.view(-1).index_add_(
            0, index, torch.ones_like(index, dtype=torch.long)
        )
        top_k = min(5, self.num_classes)
        hits = logits.topk(top_k, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1)
        self.totals[0] += targets.numel()
        self.totals[1] += int(hits.sum())
        self.loss_sum += loss.detach().to(self.loss_sum.dtype) * targets.numel()

    def reduce(self, topology: Topology) -> None:
        reduce_sum(self.confusion, topology)
        reduce_sum(self.totals, topology)
        reduce_sum(self.loss_sum, topology)

    def compute(self) -> Scores:
        count = int(self.totals[0])
        if count == 0:
            return Scores(loss=0.0, accuracy=0.0, top5=0.0, macro_f1=0.0)
        # A few hundred squared counts, reduced once per epoch, so the host does
        # it in float64 and the numbers come out the same on every device.
        confusion = self.confusion.cpu().double()
        true_positive = confusion.diagonal()
        predicted = confusion.sum(dim=0)
        actual = confusion.sum(dim=1)
        precision = true_positive / predicted.clamp(min=1.0)
        recall = true_positive / actual.clamp(min=1.0)
        denominator = (precision + recall).clamp(min=1e-12)
        f1 = torch.where(
            (precision + recall) > 0,
            2.0 * precision * recall / denominator,
            torch.zeros_like(denominator),
        )
        return Scores(
            loss=float(self.loss_sum.cpu().item() / count),
            accuracy=float(true_positive.sum().item() / count),
            top5=float(int(self.totals[1]) / count),
            macro_f1=float(f1.mean().item()),
        )
