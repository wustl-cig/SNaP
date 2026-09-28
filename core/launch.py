"""Process launch + DDP setup.

Only the train and test phases exist here.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist

from core.experiment import prepare_experiment
from stage import SNaPStage


def setup_ddp_env(rank, world_size, config):
    if not bool(config["distributed"].get("use_ddp", False)):
        return
    os.environ["MASTER_ADDR"] = config["distributed"].get("master_addr", "localhost")
    os.environ["MASTER_PORT"] = str(config["distributed"].get("master_port", 29500))
    backend = config["distributed"].get("backend", "nccl")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main_worker(rank, world_size, config, gpu_ids):
    physical_gpu = gpu_ids[rank] if gpu_ids else None
    setup_ddp_env(rank, world_size, config)

    # Bound PyTorch's intra-op thread pool per process (mirrors the OMP/MKL env caps set
    # in run.py) so `world_size` ranks share the cores instead of each grabbing all of
    # them. Covers the single-GPU path too, where run.py's env caps land after torch is
    # already imported and so may not re-apply to the OpenMP backend.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, world_size)))

    # CPU fallback: the model trains on CUDA in practice (the JVP step is fp32 and
    # memory-hungry), but everything runs on CPU too, which is what makes the smoke
    # tests and a laptop-sized demo possible.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    is_main = (rank == 0)

    try:
        prepare_experiment(config=config, is_main=is_main, rank=rank)
        phase = config["experiment"]["phase"]

        stage = SNaPStage(config=config, device=device, rank=rank,
                                     world_size=world_size, is_main=is_main,
                                     physical_gpu=physical_gpu)
        stage.build()

        if phase == "train":
            stage.run_train()
        elif phase == "test":
            stage.run_test()
        else:
            raise ValueError(f"Unknown phase: {phase!r} (expected 'train' or 'test')")
    finally:
        cleanup()
