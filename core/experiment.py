"""Experiment setup: seeding, output paths, log/config/code snapshots.

The code snapshot captures THIS package, so every run directory carries the exact code
that produced it.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import torch
import yaml


class Tee:
    """Duplicate stdout to a log file (minimal, dependency-free)."""

    def __init__(self, path: str, mode: str = "w"):
        self.file = open(path, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


@dataclass
class ExperimentContext:
    save_root: str
    run_id: str
    is_main: bool


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# package root = the parent of this core/ dir
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_experiment(config, is_main: bool, rank: int = 0) -> ExperimentContext:
    exp = config["experiment"]
    stage = exp["stage"]
    phase = exp["phase"]
    run_id = exp["run_id"]

    base_root = config["paths"]["output_root"]
    save_root = os.path.join(base_root, stage, phase, run_id)

    seed = int(exp.get("seed", 42)) + int(rank)
    seed_all(seed)

    if is_main:
        os.makedirs(save_root, exist_ok=True)

        log_path = os.path.join(save_root, f"{phase}.log")
        sys.stdout = Tee(log_path, mode="w")

        print(f"[EXP] save_root: {save_root}")
        print(f"[EXP] stage={stage} phase={phase} run_id={run_id} seed={seed}")

        cfg_out = os.path.join(save_root, "config_used.yaml")
        with open(cfg_out, "w") as f:
            yaml.safe_dump(config, f)
        print(f"[EXP] wrote config snapshot: {cfg_out}")

        code_dst = os.path.join(save_root, "code")
        if not os.path.exists(code_dst):
            shutil.copytree(
                _PACKAGE_ROOT, code_dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "*.pyo", "*.log", ".git", "logs", "runs"
                ),
            )
            print(f"[EXP] saved code snapshot: {code_dst}")

    config.setdefault("_runtime", {})
    config["_runtime"]["save_root"] = save_root

    return ExperimentContext(save_root=save_root, run_id=run_id, is_main=is_main)
