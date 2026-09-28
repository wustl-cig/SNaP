"""Launcher for the SNaP inverse-problem trainer.

Single- or multi-GPU. Multi-GPU uses manual data-parallelism (see stage.py), so DDP is
enabled only to bootstrap the process group / weight broadcast.

Usage:
    python run.py --config configs/celeba.yaml
    python run.py --config configs/afhq.yaml --set experiment.stage=denoising
    python run.py --config configs/celeba.yaml --set experiment.phase=test \
        experiment.stage=sr model.resume_path=/path/ckpt.pt
"""

import argparse
import os
import sys
from datetime import datetime

import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.config import load_config, apply_overrides          # noqa: E402
from core.launch import main_worker, free_port                # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "configs", "celeba.yaml"),
                        help="Path to a YAML config (default: configs/celeba.yaml).")
    parser.add_argument("--set", metavar="key=value", nargs="+", default=[],
                        help="Override config values, e.g. --set experiment.stage=denoising")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.set:
        config = apply_overrides(config, args.set)

    import torch
    gpu_ids = list(config["distributed"].get("gpus") or [])
    if not torch.cuda.is_available():
        # No GPU: single CPU process. Fine for the smoke tests and small demos; real
        # training needs a GPU.
        print("[snap] No CUDA device found -- running on CPU (single process).")
        gpu_ids = []
        config.setdefault("distributed", {})["use_ddp"] = False
    world_size = max(1, len(gpu_ids))

    # Cap CPU thread pools BEFORE spawning workers. Without this each of the
    # `world_size` processes (plus every DataLoader worker) defaults its BLAS/OpenMP
    # pool to one thread PER CORE, so N procs oversubscribe the CPU by N x.
    # setdefault so an explicit env override still wins. Children (spawn) inherit these.
    n_threads = max(1, (os.cpu_count() or 1) // max(1, world_size))
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, str(n_threads))

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    config.setdefault("distributed", {})["master_port"] = free_port()
    if world_size > 1:
        config["distributed"]["use_ddp"] = True

    config.setdefault("experiment", {})
    if not config["experiment"].get("run_id"):
        config["experiment"]["run_id"] = datetime.now().strftime("%d-%b-%Y/%H-%M-%S")

    print(f"[snap] GPUs {gpu_ids} (world_size={world_size}) | "
          f"stage={config['experiment'].get('stage')} phase={config['experiment'].get('phase')} | "
          f"run_id={config['experiment']['run_id']}")

    if world_size == 1:
        main_worker(0, 1, config, gpu_ids)
    else:
        mp.spawn(main_worker, args=(world_size, config, gpu_ids), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
