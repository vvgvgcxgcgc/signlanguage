"""Process group setup for `parallel="ddp"`, and no-ops for everything else.

Launch with torchrun, which sets RANK, LOCAL_RANK, and WORLD_SIZE:

    torchrun --nproc_per_node=2 -m training.run --parallel ddp --root DATA

Every rank builds its own dataset, including the shoulder-scale scan, so that
cost is paid once per process. Rank 0 goes first so only one process writes
labels/<tag>_<model>.json in the working directory.
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


def _configure_nccl() -> None:
    """Keep NCCL off the paths that SIGSEGV on a pair of T4s.

    Those cards have no NVLink. The process group can still start, then the
    first DDP broadcast dies with signal 11 if P2P or the cuMem allocator is on.
    """
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")


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
        _configure_nccl()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))
    else:
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


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and torch.backends.mps.is_available()


def resolve_device(topology: Topology, parallel: str, device: str = "auto") -> torch.device:
    """Pick the compute device for this process.

    `auto` prefers CUDA, then Apple MPS, then CPU. An explicit `cpu` skips
    both accelerators. `dp` and `ddp` stay CUDA-only: a T4 session with the
    accelerator off, or a CPU torch wheel, has to fail here instead of
    training for hours on the host.
    """
    if parallel in {"dp", "ddp"} and device in {"cpu", "mps"}:
        raise RuntimeError(f"parallel={parallel!r} needs CUDA, got device={device!r}.")

    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    if parallel in {"dp", "ddp"} and not available:
        raise RuntimeError(
            f"parallel={parallel!r} needs CUDA, but torch.cuda.is_available() is False. "
            "On Kaggle set the accelerator to GPU T4 x2. Reinstalling torch from "
            "requirements.txt can replace that image's CUDA build with a CPU wheel."
        )
    if device == "cuda" and not available:
        raise RuntimeError("device='cuda' was requested, but torch.cuda.is_available() is False.")
    if device == "mps" and not _mps_available():
        raise RuntimeError("device='mps' was requested, but torch.backends.mps.is_available() is False.")

    if parallel == "dp" and count < 2:
        raise RuntimeError(f"parallel='dp' needs at least 2 CUDA devices, found {count}.")
    if parallel == "ddp" and topology.world_size < 2:
        raise RuntimeError(
            "parallel='ddp' needs torchrun --standalone --nproc_per_node=2 so each T4 gets a process."
        )

    use_cuda = device == "cuda" or (device == "auto" and available)
    if use_cuda:
        index = topology.local_rank if topology.distributed else 0
        if index >= count:
            raise RuntimeError(f"local_rank {index} is outside the {count} visible CUDA device(s).")
        names = ", ".join(f"{i}:{torch.cuda.get_device_name(i)}" for i in range(count))
        logger.info("CUDA device cuda:%d | %d visible (%s) | parallel=%s", index, count, names, parallel)
        return torch.device(f"cuda:{index}")
    if device == "mps" or (device == "auto" and _mps_available()):
        logger.info("Apple MPS device | parallel=%s", parallel)
        return torch.device("mps")
    logger.info("CPU device | parallel=%s", parallel)
    return torch.device("cpu")
