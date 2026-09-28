"""Download the pretrained checkpoints listed in checkpoints/manifest.json.

    python scripts/download_checkpoints.py --list            # what is available
    python scripts/download_checkpoints.py celeba_sr         # one model
    python scripts/download_checkpoints.py --group celeba    # all CelebA models
    python scripts/download_checkpoints.py --all             # everything (~2 GB)

Every file is verified against the sha256 in the manifest, so a truncated or tampered
download fails here rather than three hours into an evaluation. Files already present
and matching are skipped, and interrupted downloads resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _download import download, human, sha256_file   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MANIFEST = os.path.join(REPO, "checkpoints", "manifest.json")
PLACEHOLDER = "REPLACE_ME"


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def show(models):
    w = max(len(k) for k in models)
    print(f"{'model':{w}}  {'size':>8}  {'stage':14}  {'one-step PSNR/SSIM/LPIPS':26}  description")
    for name, e in models.items():
        m = e.get("metrics", {})
        p = m.get("M1")
        score = (f"{p[0]:6.2f} / {p[1]:.4f} / " + (f"{p[2]:.4f}" if p[2] is not None else "  --  ")
                 if p else "")
        print(f"{name:{w}}  {human(e['size']):>8}  {e['stage']:14}  {score:26}  {e['description']}")
    print("\nMetrics are the test-split numbers for a single posterior draw (k=1, M=1);\n"
          "see the README for the sample-averaged (M=100) column.")


def resolve_base_url(manifest, override):
    base = override or os.environ.get("SNAP_WEIGHTS_URL") or manifest.get("base_url", "")
    if not base or PLACEHOLDER in base:
        raise SystemExit(
            "No download URL for the checkpoints yet.\n\n"
            "  checkpoints/manifest.json still has base_url = " + PLACEHOLDER + ".\n"
            "  Set it to wherever the .pt files are hosted (a GitHub Release, a Hugging\n"
            "  Face model repo, Zenodo, ...), or pass one for this run:\n\n"
            "      python scripts/download_checkpoints.py --all \\\n"
            "          --base-url https://github.com/<owner>/<repo>/releases/download/v1.0\n\n"
            "  or set SNAP_WEIGHTS_URL in your environment.")
    return base.rstrip("/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model names from the manifest")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--group", choices=["celeba", "afhq", "brain"], help="download a whole family")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--base-url", default=None, help="override the manifest's base_url")
    ap.add_argument("--out", default=os.path.join(REPO, "checkpoints"))
    ap.add_argument("--force", action="store_true", help="re-download even if the file verifies")
    a = ap.parse_args()

    manifest = load_manifest()
    models = manifest["models"]
    if a.list:
        show(models)
        return 0

    wanted = list(a.models)
    if a.all:
        wanted = list(models)
    elif a.group:
        wanted += [k for k in models if k.startswith(a.group)]
    if not wanted:
        show(models)
        print("\nNothing selected. Pass model names, --group, or --all.")
        return 1
    unknown = [w for w in wanted if w not in models]
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}\navailable: {', '.join(models)}")

    base = resolve_base_url(manifest, a.base_url)
    os.makedirs(a.out, exist_ok=True)
    total = sum(models[w]["size"] for w in dict.fromkeys(wanted))
    print(f"{len(set(wanted))} checkpoint(s), {human(total)} -> {a.out}\n")

    for name in dict.fromkeys(wanted):
        e = models[name]
        dest = os.path.join(a.out, e["file"])
        print(f"=== {name}  ({human(e['size'])})")
        if os.path.exists(dest):
            if not a.force and os.path.getsize(dest) == e["size"] \
                    and sha256_file(dest) == e["sha256"]:
                print("    already present and verified")
                continue
            # A wrong-but-complete file would make download() skip the transfer as
            # "already complete", so remove it and start clean.
            print("    present but does not verify; re-fetching")
            os.remove(dest)
        download(f"{base}/{e['file']}", dest, e["size"])
        print("    verifying sha256")
        got = sha256_file(dest)
        if got != e["sha256"]:
            os.remove(dest)
            raise SystemExit(f"    sha256 mismatch for {name}\n      expected {e['sha256']}\n"
                             f"      got      {got}\n    deleted the file; please retry.")
        print(f"    ok -> {dest}")
    print("\nDone. Try:  python demo.py --model " + list(dict.fromkeys(wanted))[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
