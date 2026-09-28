"""ADOBI-protocol fastMRI **brain** multicoil dataset for Mean-Flow.

Why this exists: the InverseBench *knee* train LMDB ships `mvue/` only -- no k-space,
so real ESPIRiT maps cannot be recovered and training had to fall back on synthetic
birdcage maps (see `mri_lmdb.py`). That approximation was measured to cost real
accuracy: the network specialised to the synthetic operator, and the train/test gap grew
to 0.45 dB and was still widening at 20 data passes. This dataset removes the problem --
`smps_hat/` carries precomputed per-slice ESPIRiT maps for EVERY volume, training
included, so train and test share one operator family.

Protocol follows the self-supervised CS-MRI setup of arXiv:2411.16535, so numbers on
this split are directly comparable to that line of work:

    c_i  = F^H k_i                    per-coil images at the acquired FOV
    c_i' = crop(c_i, 320, 320)        crop DEFINES the FOV of the study
    S    = crop(smps_hat, 320, 320)
    x0   = sum_i conj(S_i) c_i'       ground truth (coil-combined, complex)

The crop redefines the problem on the 320x320 grid, which is what makes the sampling
mask well defined there and `A x0 = y` exact.

FFT convention: that setup uses fftshift(fft2(ifftshift(.))) while `methods/mri.py` uses
ifftshift(fft2(fftshift(.))). These coincide for EVEN axis lengths, and every length
here is even (768, 396, 320), so the two agree exactly -- verified in
`tests/`-style checks rather than assumed.

Coil counts vary per volume (ten distinct values; 4, 16 and 20 dominate), so maps are
zero-padded to `max_coils`. A coil with S_c = 0 contributes nothing to A or A^H, so
padding is transparent. It is ALSO safe for the posterior source, but only because
`MRIProblem.apply_pinv_reg` solves the image-space normal equations: A^H annihilates the
padded coils. The measurement-space Gram route would have enlarged the null space of
A A^H and made an already ill-conditioned solve worse.

__getitem__ returns the same dict shape as `MRILMDBDataset`:
    {"target": (2, H, W) float32 real-view x0,
     "maps":   (Cmax, H, W) complex64 sensitivity maps}
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_SPLIT2KEY = {"train": "train", "validation": "val", "val": "val", "test": "test"}

# h5py File objects are neither picklable (spawn) nor fork-safe (a handle inherited
# through fork shares file offsets and will return garbage under concurrent reads).
# Cache by (kind, path, PID) so every worker process opens its own -- same fix as the
# LMDB transaction cache in mri_lmdb.py.
_H5_CACHE: dict = {}


def cache_key(fname: str, sl: int) -> str:
    """Cache filename for one (volume, slice). Split-independent, so train/val/test all
    read from one directory."""
    return f"{os.path.splitext(fname)[0]}_s{sl:02d}.npz"


def _to_real(x: np.ndarray) -> np.ndarray:
    """complex (..., H, W) -> float32 (..., H, W, 2)."""
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def _ifft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def _center_crop(x: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = x.shape[-2:]
    if h < th or w < tw:
        raise ValueError(f"cannot crop {h}x{w} to {th}x{tw}")
    t, l = (h - th) // 2, (w - tw) // 2
    return x[..., t:t + th, l:l + tw]


class FastMRIBrainDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        dcfg = cfg["data"]
        self.raw_dir = dcfg["raw_dir"]
        self.maps_dir = dcfg["maps_dir"]
        self.crop = int(dcfg.get("crop", 320))
        self.max_coils = int(dcfg.get("max_coils", 20))
        # Per-slice intensity normalisation. fastMRI brain has a fairly consistent
        # absolute scale, but a FIXED measurement sigma only corresponds to a fixed SNR
        # if the image scale is fixed too -- and `MRIProblem.sigma` is a scalar (the
        # posterior source folds it into a scalar lambda, see pipeline._draw_sigma).
        # "max" divides by max|x0| per slice, reproducing that setup's `prepare_batch`
        # (`norm = complex_abs(x5).max(); x = x5 / norm`) -- so `sigma_data` and any
        # number quoted against that work refer to the same scale. It also makes the
        # metric's per-image data_range exactly 1. "rms" instead sets sqrt(E|x|^2) == 1;
        # "none" keeps the raw ADOBI scale (~1e-4), which no fixed sigma suits.
        self.normalize = str(dcfg.get("normalize", "max")).lower()
        if self.normalize not in ("max", "rms", "quantile99", "none"):
            raise ValueError(
                f"data.normalize must be max|rms|quantile99|none, got {self.normalize!r}")

        # Optional precomputed cache of the CROPPED arrays (see build_brain_cache.py).
        # Reading raw volumes at train time is ~15.3 s/step against ~0.5 s/step of
        # compute -- 30x I/O bound -- because every sample pulls a full (C,768,396)
        # k-space slice plus maps at random across 769 files. The cache holds raw
        # (unnormalised, unpadded) arrays, so `normalize` and `max_coils` remain
        # changeable without rebuilding it. Falls back to the raw volumes per slice when
        # a cache entry is missing, so a partial cache still trains (just slower).
        self.cache_dir = dcfg.get("cache_dir") or None
        self._warned_miss = False

        idx_path = dcfg["index_json"]
        with open(idx_path) as f:
            index = json.load(f)
        key = _SPLIT2KEY.get(split, "train")
        if key not in index:
            raise KeyError(f"{idx_path} has no {key!r} split (has {list(index)})")
        records: List[dict] = index[key]

        # ADOBI drops the first 4 and last 5 slices of each volume.
        s0 = int(dcfg.get("slice_start", 4))
        s1 = int(dcfg.get("slice_end", -5))
        self.index: List[tuple] = []
        for rec in records:
            n = int(rec["slices"])
            lo = s0 if s0 >= 0 else n + s0
            hi = s1 if s1 >= 0 else n + s1
            for s in range(max(0, lo), min(n, hi)):
                self.index.append((rec["file"], s))

        max_len = dcfg.get("max_len", None)
        if max_len:
            self.index = self.index[:int(max_len)]
        self.H = self.W = self.crop

    def __len__(self) -> int:
        return len(self.index)

    def _h5(self, kind: str, fname: str):
        path = (os.path.join(self.raw_dir, fname) if kind == "kspace"
                else os.path.join(self.maps_dir, "smps_hat", fname))
        key = (kind, path, os.getpid())
        h = _H5_CACHE.get(key)
        if h is None:
            # Drop handles inherited from another PID before opening our own.
            for k in [k for k in _H5_CACHE if k[2] != os.getpid()]:
                try:
                    _H5_CACHE.pop(k).close()
                except Exception:
                    pass
            h = h5py.File(path, "r")
            _H5_CACHE[key] = h
        return h

    def _pad_coils(self, arr: np.ndarray) -> np.ndarray:
        c = arr.shape[0]
        if c > self.max_coils:
            raise ValueError(f"{c} coils exceeds max_coils={self.max_coils}")
        out = np.zeros((self.max_coils,) + arr.shape[1:], dtype=arr.dtype)
        out[:c] = arr
        return out

    def _scale(self, x0: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x0
        if self.normalize == "max":
            s = np.abs(x0).max()
        elif self.normalize == "quantile99":
            s = np.quantile(np.abs(x0), 0.99)
        else:                                    # "rms": sqrt(E|x|^2) -> 1
            s = np.sqrt(np.mean(np.abs(x0) ** 2))
        return x0 / max(float(s), 1e-12)

    def _load_raw(self, fname: str, s: int):
        """(x0, maps) cropped to the study grid, from the cache when available."""
        if self.cache_dir is not None:
            path = os.path.join(self.cache_dir, cache_key(fname, s))
            if os.path.exists(path):
                with np.load(path) as z:
                    return z["x0"], z["maps"]
            if not self._warned_miss:
                self._warned_miss = True
                warnings.warn(
                    f"cache miss for {cache_key(fname, s)} in {self.cache_dir}; falling "
                    "back to the raw volumes, which is ~30x slower. Run "
                    "dataset/build_brain_cache.py to finish the cache.",
                    RuntimeWarning, stacklevel=2)
        k = self.crop
        ksp = self._h5("kspace", fname)["kspace"][s]          # (C,H,W) complex
        smps = _center_crop(self._h5("smps", fname)["smps_hat"][s], k, k)
        coil_img = _center_crop(_ifft2c(ksp), k, k)
        return (coil_img * np.conj(smps)).sum(0), smps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname, s = self.index[idx]
        x0, smps = self._load_raw(fname, s)
        x0 = self._scale(x0)

        target = torch.from_numpy(_to_real(x0)).permute(2, 0, 1).contiguous()   # (2,H,W)
        maps = torch.from_numpy(np.ascontiguousarray(self._pad_coils(smps)))    # (Cmax,H,W)
        return {"target": target, "maps": maps}
