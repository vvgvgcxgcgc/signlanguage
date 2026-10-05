"""Process group setup for `parallel="ddp"`, and no-ops for everything else.

Launch with torchrun, which sets RANK, LOCAL_RANK, and WORLD_SIZE:

    torchrun --nproc_per_node=2 -m training.run --parallel ddp --root DATA

Every rank builds its own dataset, including the shoulder-scale scan, so that
cost is paid once per process. Rank 0 goes first so only one process writes
labels.json.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Topology:
    """Who this process is. `world_size` is 1 outside DDP."""

    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def launched_with_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup(parallel: str) -> Topology:
    """Join the process group when asked for DDP, otherwise report a single rank."""
    if parallel != "ddp":
        return Topology(rank=0, local_rank=0, world_size=1, distributed=False)
    if not launched_with_torchrun():
        raise RuntimeError(
            "parallel='ddp' needs torchrun, which sets RANK and WORLD_SIZE. "
            "Use parallel='dp' for a single-process multi-GPU run."
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    logger.info("Joined %s group as rank %d of %d", backend, rank, world_size)
    return Topology(rank=rank, local_rank=local_rank, world_size=world_size, distributed=True)


def cleanup(topology: Topology) -> None:
    if topology.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(topology: Topology) -> None:
    if topology.distributed and dist.is_initialized():
        dist.barrier()


def reduce_sum(tensor: torch.Tensor, topology: Topology) -> torch.Tensor:
    """Sum a tensor across ranks in place and return it."""
    if topology.distributed and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def resolve_device(topology: Topology, parallel: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{topology.local_rank}" if topology.distributed else "cuda")
    if parallel != "none":
        logger.warning("No CUDA device, so parallel=%r falls back to a single device", parallel)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
