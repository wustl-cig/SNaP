"""Precompute the cropped 320x320 fastMRI-brain cache used by `fastmri_brain.py`.

Why: reading the raw volumes at train time is catastrophically I/O bound. Each sample
pulls a full (C, 768, 396) complex64 k-space slice AND its maps -- ~100 MB -- at random
across 769 files on network storage. Measured 15.3 s/step against ~0.5 s/step of actual
compute, i.e. 183 h for an 80-epoch run. Caching the CROPPED arrays cuts a sample to
~10.7 MiB and returns training to compute-bound (~6 h).

The win comes from access pattern, not just size: this reads each volume ONCE,
sequentially, instead of hitting it repeatedly at random slice offsets.

Stores the RAW (unnormalised, unpadded) arrays, so `data.normalize` and `data.max_coils`
stay changeable without a rebuild:
    x0   complex64 (320, 320)      coil-combined ground truth
    maps complex64 (C, 320, 320)   ESPIRiT maps, C = this volume's true coil count

Resumable: slices whose cache file already exists are skipped, so re-running after an
interruption picks up where it stopped.

    python dataset/build_brain_cache.py --config configs/mri_brain.yaml --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataset.fastmri_brain import _center_crop, _ifft2c, cache_key   # noqa: E402


def _one_volume(job):
    """Process every wanted slice of ONE volume; returns (written, skipped, error)."""
    fname, slices, raw_dir, maps_dir, cache_dir, crop = job
    todo = [s for s in slices
            if not os.path.exists(os.path.join(cache_dir, cache_key(fname, s)))]
    if not todo:
        return (0, len(slices), None)
    try:
        lo, hi = min(todo), max(todo) + 1
        with h5py.File(os.path.join(raw_dir, fname), "r") as fk, \
             h5py.File(os.path.join(maps_dir, "smps_hat", fname), "r") as fs:
            # One contiguous read of the slice window, not per-slice seeks.
            ksp = fk["kspace"][lo:hi]                      # (n, C, H, W) complex
            smp = fs["smps_hat"][lo:hi]
        n = 0
        for s in todo:
            k, m = ksp[s - lo], smp[s - lo]
            coil_img = _center_crop(_ifft2c(k), crop, crop)
            maps = _center_crop(m, crop, crop)
            x0 = (coil_img * np.conj(maps)).sum(0)
            tmp = os.path.join(cache_dir, cache_key(fname, s) + f".tmp{os.getpid()}")
            np.savez(tmp, x0=x0.astype(np.complex64), maps=maps.astype(np.complex64))
            # np.savez appends .npz to a path without it; rename atomically so a killed
            # job never leaves a half-written file that a later run would trust.
            os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp,
                       os.path.join(cache_dir, cache_key(fname, s)))
            n += 1
        return (n, len(slices) - len(todo), None)
    except Exception as e:                                   # keep going on one bad volume
        return (0, 0, f"{fname}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "mri_brain.yaml"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    from core.config import load_config
    dcfg = load_config(args.config)["data"]
    cache_dir = dcfg.get("cache_dir")
    if not cache_dir:
        raise SystemExit("set data.cache_dir in the config first")
    os.makedirs(cache_dir, exist_ok=True)
    crop = int(dcfg.get("crop", 320))
    s0, s1 = int(dcfg.get("slice_start", 4)), int(dcfg.get("slice_end", -5))

    index = json.load(open(dcfg["index_json"]))
    jobs = []
    for sp in args.splits:
        for rec in index[sp]:
            n = int(rec["slices"])
            lo = s0 if s0 >= 0 else n + s0
            hi = s1 if s1 >= 0 else n + s1
            wanted = list(range(max(0, lo), min(n, hi)))
            if wanted:
                jobs.append((rec["file"], wanted, dcfg["raw_dir"], dcfg["maps_dir"],
                             cache_dir, crop))
    total = sum(len(j[1]) for j in jobs)
    print(f"[cache] {len(jobs)} volumes / {total} slices -> {cache_dir}", flush=True)

    done = skipped = 0
    errs = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (w, sk, err) in enumerate(pool.imap_unordered(_one_volume, jobs), 1):
            done += w; skipped += sk
            if err:
                errs.append(err)
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                left = (total - done - skipped) / max(rate, 1e-9)
                print(f"[cache] vol {i}/{len(jobs)} | written {done} skipped {skipped} "
                      f"| {rate:.1f} slice/s | eta {left/60:.0f} min", flush=True)
    print(f"[cache] DONE written={done} skipped={skipped} errors={len(errs)} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    for e in errs[:10]:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
