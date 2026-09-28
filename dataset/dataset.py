from __future__ import annotations
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class mydataset(Dataset):
    def __init__(self, cfg, split="train"):
        super().__init__()
        self.dcfg = cfg["data"]
        self.root = self.dcfg["root"]
        self.split_dir = Path(self.root) / split
        self.max_len = self.dcfg.get("max_len", None)
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        # data.recursive (optional, default false) globs nested subfolders too.
        pattern = "**/*" if self.dcfg.get("recursive", False) else "*"
        files = [p for p in self.split_dir.glob(pattern) if p.suffix.lower() in IMG_EXTS]
        if len(files) == 0:
            raise RuntimeError(f"No images found in {self.split_dir}")
        self.files: List[Path] = sorted(files)

        if self.max_len is not None:
            self.files = self.files[: self.max_len]

        tfms = []
        size = self.dcfg.get("size", 128)

        if self.dcfg.get("center_crop", True):
            crop_size = self.dcfg.get("crop_size", 160)
            tfms.append(T.CenterCrop(crop_size))
        tfms.append(
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR,antialias=True,))
        tfms.append(T.ToTensor())  # [0,1], CxHxW
        if self.dcfg.get("normalize", True):
            tfms.append(T.Lambda(self.to_minus1_plus1))
        self.transform = T.Compose(tfms)

        # Random horizontal flip, TRAIN SPLIT ONLY (off by default).
        #
        # Train-only is the point: validation and test must stay deterministic or the val
        # curve stops being comparable across epochs and evals stop being reproducible.
        # It is applied here, to x, i.e. BEFORE make_observation -- so a fixed mask
        # (same_mask_across_batch + mask_seed) is still ONE fixed operator A, and the
        # augmentation enlarges the IMAGE distribution rather than the operator.
        self.hflip = bool(self.dcfg.get("augment_hflip", False)) and split == "train"

    def __len__(self) -> int:
        return len(self.files)

    def to_minus1_plus1(self, x):
        return x * 2.0 - 1.0

    def _load_pil(self, path: Path) -> Image.Image:
        img = Image.open(path)
        img = img.convert("L") if self.dcfg.get("grayscale", False) else img.convert("RGB")
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        pil = self._load_pil(path)
        x = self.transform(pil)  # (C,H,W)
        if self.hflip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[-1])
        return x
