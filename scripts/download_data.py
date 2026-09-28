"""Download and lay out the image datasets this repo trains on.

    python scripts/download_data.py celeba          # 1.4 GB zip -> data/celeba/{train,validation,test}
    python scripts/download_data.py afhq            # 5.8 GB zip -> data/afhq_cat/{train,validation,test}
    python scripts/download_data.py all
    python scripts/download_data.py celeba --verify-only     # check an existing install
    python scripts/download_data.py celeba --archive /path/img_align_celeba_or_Dataset.zip

Both datasets are fetched from public mirrors of the ORIGINAL files, and the result is
checked against scripts/data_manifest.json: file counts, a hash of the sorted file list,
and the sha256 of sampled images. The published numbers were produced on exactly these
bytes, so if a mirror ever re-encodes its images the verification fails loudly rather
than shifting every metric by a fraction of a dB.

Splits
  CelebA   the official partition, which is contiguous in filename order:
           train 000001-162770, validation 162771-182637, test 182638-202599.
  AFHQ-Cat train = afhq/train/cat (5,153), validation = afhq/val/cat (500),
           test = the first 100 of validation in sorted order.

The downloads are resumable: re-running continues a partial archive instead of
restarting it. Pass --keep-archive to leave the zip in place (default: delete it once
extraction succeeds). fastMRI is NOT downloadable here -- it needs credentials; see
docs/DATA.md.

Licensing: CelebA is for non-commercial research only and AFHQ is CC BY-NC 4.0. This
script downloads them for you; it does not relicense them, and this repository ships no
dataset images of its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _download import download, human, sha256_file   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "data_manifest.json")

SOURCES = {
    # Original `img_align_celeba` JPEGs (202,599 files, original names), mirrored on the
    # Hugging Face Hub. Verified byte-identical to the images used for the paper.
    "celeba": dict(
        url="https://huggingface.co/datasets/student/celebA/resolve/main/Dataset.zip",
        archive="celebA_Dataset.zip", size=1414826162,
        member_prefix="Dataset/CelebA_train/img_align_celeba/",
        out="celeba",
        note="CelebA (aligned & cropped), non-commercial research use only."),
    # AFHQ v1 from the StarGAN v2 release (afhq/train|val/{cat,dog,wild}).
    "afhq": dict(
        url="https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1",
        archive="afhq.zip", size=729484677,
        member_prefix="afhq/",
        out="afhq_cat",
        note="AFHQ v1 (StarGAN v2), CC BY-NC 4.0."),
}


# --------------------------------------------------------------------------- extract
def extract_celeba(zpath, out_root):
    """img_align_celeba -> train/validation/test by the official contiguous partition."""
    bounds = (("train", 1, 162770), ("validation", 162771, 182637), ("test", 182638, 202599))
    for s, _, _ in bounds:
        os.makedirs(os.path.join(out_root, s), exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".jpg")]
        if not members:
            raise SystemExit(f"{zpath}: no .jpg members found")
        print(f"    extracting {len(members)} images")
        for i, m in enumerate(sorted(members)):
            name = os.path.basename(m)
            try:
                idx = int(os.path.splitext(name)[0])
            except ValueError:
                raise SystemExit(f"unexpected CelebA file name {name!r} (want 000001.jpg style)")
            split = next(s for s, lo, hi in bounds if lo <= idx <= hi)
            dst = os.path.join(out_root, split, name)
            if not os.path.exists(dst):
                with z.open(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
            if (i + 1) % 20000 == 0:
                print(f"      {i+1}/{len(members)}")


def extract_afhq(zpath, out_root):
    """afhq/train/cat -> train, afhq/val/cat -> validation, first 100 of val -> test."""
    for s in ("train", "validation", "test"):
        os.makedirs(os.path.join(out_root, s), exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        groups = {"train": [m for m in names if m.startswith("afhq/train/cat/") and m.endswith(".jpg")],
                  "validation": [m for m in names if m.startswith("afhq/val/cat/") and m.endswith(".jpg")]}
        if not groups["train"]:
            raise SystemExit(f"{zpath}: no afhq/train/cat/*.jpg members")
        for split, members in groups.items():
            print(f"    extracting {len(members)} -> {split}")
            for m in sorted(members):
                dst = os.path.join(out_root, split, os.path.basename(m))
                if not os.path.exists(dst):
                    with z.open(m) as src, open(dst, "wb") as f:
                        shutil.copyfileobj(src, f)
    # test = the first 100 validation images in sorted order (the fixed evaluation subset)
    val = sorted(os.listdir(os.path.join(out_root, "validation")))[:100]
    print(f"    copying {len(val)} -> test")
    for n in val:
        dst = os.path.join(out_root, "test", n)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(out_root, "validation", n), dst)


# --------------------------------------------------------------------------- verify
def verify(out_root, key):
    man = json.load(open(MANIFEST)).get(key)
    if man is None:
        print(f"[warn] no manifest entry for {key}; skipping verification")
        return True
    ok = True
    for split, exp in man.items():
        d = os.path.join(out_root, split)
        if not os.path.isdir(d):
            print(f"  [FAIL] {split}: missing directory {d}")
            ok = False
            continue
        names = sorted(os.listdir(d))
        if len(names) != exp["count"]:
            print(f"  [FAIL] {split}: {len(names)} files, expected {exp['count']}")
            ok = False
            continue
        got = hashlib.sha256("\n".join(names).encode()).hexdigest()
        if got != exp["names_sha256"]:
            print(f"  [FAIL] {split}: file list differs from the reference split")
            ok = False
            continue
        bad = [n for n, h in exp["samples"].items() if sha256_file(os.path.join(d, n)) != h]
        if bad:
            print(f"  [FAIL] {split}: {len(bad)} sampled image(s) differ byte-for-byte "
                  f"(e.g. {bad[0]}). The mirror re-encoded the data; metrics will not match "
                  f"the published numbers exactly.")
            ok = False
            continue
        print(f"  [ok]   {split}: {len(names)} files, list + samples match")
    return ok


# --------------------------------------------------------------------------- main
def do_dataset(key, args):
    spec = SOURCES[key]
    out_root = os.path.join(args.root, spec["out"])
    print(f"\n=== {key} -> {out_root}")
    print(f"    {spec['note']}")

    if not args.verify_only:
        archive = args.archive or os.path.join(args.cache, spec["archive"])
        if args.archive:
            print(f"    using local archive {archive}")
        else:
            os.makedirs(args.cache, exist_ok=True)
            print(f"    source: {spec['url']}  ({human(spec['size'])})")
            download(spec["url"], archive, spec["size"])
        (extract_celeba if key == "celeba" else extract_afhq)(archive, out_root)
        if not args.archive and not args.keep_archive:
            os.remove(archive)
            print("    removed archive (pass --keep-archive to keep it)")

    print("  verifying against scripts/data_manifest.json")
    return verify(out_root, spec["out"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["celeba", "afhq", "all"])
    ap.add_argument("--root", default=os.path.join(REPO, "data"), help="where datasets live")
    ap.add_argument("--cache", default=os.path.join(REPO, "data", "_archives"),
                    help="where the downloaded zip is stored while extracting")
    ap.add_argument("--archive", default=None,
                    help="use this local zip instead of downloading (offline installs)")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="only check an existing install; download and extract nothing")
    args = ap.parse_args()

    keys = ["celeba", "afhq"] if args.dataset == "all" else [args.dataset]
    ok = all([do_dataset(k, args) for k in keys])
    print("\nall checks passed." if ok else "\nVERIFICATION FAILED (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
