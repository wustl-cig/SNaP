"""Owner-side: record what a correct dataset install looks like.

Writes scripts/data_manifest.json: per split, the file count, a hash of the sorted file
list, and the sha256 of a few sampled images. scripts/download_data.py checks a fresh
install against this, so a mirror that re-encoded the JPEGs or shuffled the split is
caught loudly instead of quietly changing every number.

    python scripts/make_data_manifest.py --data-root /path/to/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

SPLITS = ("train", "validation", "test")
DATASETS = ("celeba", "afhq_cat")
N_SAMPLES = 4


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def describe(split_dir):
    names = sorted(os.listdir(split_dir))
    idx = [0] if len(names) == 1 else [round(i * (len(names) - 1) / (N_SAMPLES - 1))
                                       for i in range(N_SAMPLES)]
    return {
        "count": len(names),
        "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "first": names[0], "last": names[-1],
        "samples": {names[i]: sha256_file(os.path.join(split_dir, names[i])) for i in idx},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "data_manifest.json"))
    a = ap.parse_args()
    man = {"_comment": "Expected contents of a correct dataset install; see scripts/download_data.py."}
    for ds in DATASETS:
        root = os.path.join(a.data_root, ds)
        if not os.path.isdir(root):
            print(f"[skip] {root}")
            continue
        man[ds] = {s: describe(os.path.join(root, s)) for s in SPLITS
                   if os.path.isdir(os.path.join(root, s))}
        print(f"[ok] {ds}: " + ", ".join(f"{s}={man[ds][s]['count']}" for s in man[ds]))
    with open(a.out, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
