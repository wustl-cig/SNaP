from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import torch.utils.data as tud
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import mydataset


def _build_dataset(cfg_dict: Dict[str, Any], split: str):
    """Pick the dataset by data.loader ('image' default, 'mri_lmdb' for InverseBench MRI)."""
    loader = str(cfg_dict["data"].get("loader", "image")).lower()
    if loader in ("mri_lmdb", "mri"):
        from .mri_lmdb import MRILMDBDataset
        return MRILMDBDataset(cfg_dict, split=split)
    if loader in ("fastmri_brain", "brain"):
        from .fastmri_brain import FastMRIBrainDataset
        return FastMRIBrainDataset(cfg_dict, split=split)
    return mydataset(cfg_dict, split=split)


def build_loader(
    cfg_dict: Dict[str, Any],
    split: str,
    rank: int = 0,
    world_size: int = 1,
    steps_per_epoch: Optional[int] = None,
) -> Tuple[DataLoader, Optional[tud.Sampler]]:

    dataset = _build_dataset(cfg_dict, split=split)

    batch_size = int(cfg_dict["dataloader"].get("batch_size", 32))
    # knee LMDB: real ESPIRiT maps (val/test) have a per-slice coil count, so they can't
    # be stacked -> force batch_size=1 there. (Train uses fixed-coil synthetic maps.)
    # fastmri_brain does NOT need this: it zero-pads every slice to max_coils, so all
    # splits stack normally.
    if str(cfg_dict["data"].get("loader", "")).lower() in ("mri_lmdb", "mri") and split != "train":
        batch_size = 1
    drop_last = (split == "train")

    sampler: Optional[tud.Sampler] = None
    shuffle = False

    if split == "train" and steps_per_epoch is not None:
        # Each rank draws its own share of steps independently with replacement.
        num_samples = (steps_per_epoch * batch_size + world_size - 1) // world_size
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    elif world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
    else:
        shuffle = bool(cfg_dict["dataloader"].get("shuffle_train", False))

    num_workers = int(cfg_dict["dataloader"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=cfg_dict["dataloader"]["pin_memory"],
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler
