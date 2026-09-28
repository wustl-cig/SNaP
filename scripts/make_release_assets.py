"""Owner-side: turn the curated checkpoint collection into the assets this repo ships.

Source of truth is the provenance-tracked checkpoint tree (`<root>/{celeba,afhq,mri}/`),
where each model sits next to the config it was trained with. For every model listed
below this writes

    configs/pretrained/<model>.yaml   frozen config for that checkpoint (paths relativised)
    checkpoints/manifest.json         file name, size, sha256, config, reference metrics
    <weights-dir>/<model>.pt          the checkpoint, renamed for upload
    <weights-dir>/SHA256SUMS

Checkpoints that carry optimizer/scheduler state (the AFHQ epoch files, ~1.3 GB each) are
stripped to weights for the release and the network tensors are verified identical to the
source afterwards -- inference does not use the optimizer, and a 3x smaller download does.

Then upload everything in <weights-dir> and set `base_url` in checkpoints/manifest.json.
Users never run this script; they run scripts/download_checkpoints.py.

    python scripts/make_release_assets.py --checkpoints-root /path/to/snap/checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# release name -> (file under <checkpoints-root>, description, reference metrics).
#
# Metrics are the published test-split numbers (100 images, one step, k=1), keyed by the
# number of averaged posterior draws M: [PSNR, SSIM, LPIPS]. MRI is scored on the
# magnitude with a per-slice data range (psnr_dr/ssim_dr) and has no LPIPS.
MODELS = {
    "celeba_inpaint": dict(
        src="celeba/meanflow_inpainting",
        desc="CelebA 128 - random inpainting, 70% pixels missing, sigma=0.01",
        metrics={"n": 100, "psnr_in": 11.96, "M1": [31.978, 0.9277, 0.0184],
                 "M4": [33.974, 0.9512, 0.0157], "M16": [34.646, 0.9577, 0.0180],
                 "M100": [34.842, 0.9596, 0.0191]}),
    "celeba_box": dict(
        src="celeba/meanflow_box_inpainting",
        desc="CelebA 128 - centered 40x40 box inpainting, sigma=0.05",
        metrics={"n": 100, "psnr_in": 22.33, "M1": [30.731, 0.9341, 0.0212],
                 "M4": [32.783, 0.9535, 0.0186], "M16": [33.429, 0.9589, 0.0207],
                 "M100": [33.693, 0.9606, 0.0221]}),
    "celeba_deblur": dict(
        src="celeba/meanflow_blur_gauss",
        desc="CelebA 128 - Gaussian deblurring, sigma_b=1.0, k=61, sigma=0.05",
        metrics={"n": 100, "psnr_in": 27.24, "M1": [32.919, 0.9193, 0.0190],
                 "M4": [34.980, 0.9465, 0.0215], "M16": [35.682, 0.9537, 0.0275],
                 "M100": [35.909, 0.9558, 0.0301]}),
    "celeba_sr": dict(
        src="celeba/meanflow_sr",
        desc="CelebA 128 - 2x super-resolution, sigma=0.05",
        metrics={"n": 100, "psnr_in": 11.68, "M1": [31.271, 0.9032, 0.0232],
                 "M4": [33.268, 0.9349, 0.0237], "M16": [33.953, 0.9438, 0.0290],
                 "M100": [34.161, 0.9464, 0.0313]}),
    "afhq_inpaint": dict(
        src="afhq/meanflow_inpainting",
        desc="AFHQ-Cat 256 - random inpainting, 70% pixels missing, sigma=0.01",
        metrics={"n": 100, "psnr_in": 13.25, "M1": [30.486, 0.8536, 0.0664],
                 "M16": [33.047, 0.9097, 0.0564], "M100": [33.252, 0.9133, 0.0586]}),
    "afhq_box": dict(
        src="afhq/meanflow_box_inpainting",
        desc="AFHQ-Cat 256 - centered 80x80 box inpainting, sigma=0.05",
        metrics={"n": 100, "psnr_in": 21.57, "M1": [26.468, 0.8916, 0.0540],
                 "M16": [29.329, 0.9227, 0.0566], "M100": [29.539, 0.9253, 0.0691]}),
    "afhq_deblur": dict(
        src="afhq/meanflow_blur_gauss",
        desc="AFHQ-Cat 256 - Gaussian deblurring, sigma_b=3.0, k=61, sigma=0.05",
        metrics={"n": 100, "psnr_in": 23.58, "M1": [26.249, 0.6744, 0.1596],
                 "M16": [29.098, 0.7798, 0.2315], "M100": [29.328, 0.7889, 0.3201]}),
    "afhq_sr": dict(
        src="afhq/meanflow_sr",
        desc="AFHQ-Cat 256 - 4x super-resolution, sigma=0.05",
        metrics={"n": 100, "psnr_in": 11.97, "M1": [26.075, 0.7036, 0.1307],
                 "M16": [28.777, 0.8029, 0.1573], "M100": [28.993, 0.8106, 0.1870]}),
    # MRI: one model per (acceleration, measurement SNR) cell -- each is only valid at
    # the operating point it was trained for.
    "brain_r4_20db": dict(
        src="mri/meanflow_4x_20",
        desc="fastMRI brain - multi-coil CS-MRI, R=4, 20 dB",
        metrics={"n": 100, "metric": "psnr_dr/ssim_dr", "psnr_in": 25.35,
                 "M1": [32.068, 0.8895, None], "M4": [32.680, 0.9047, None],
                 "M16": [32.852, 0.9084, None], "M100": [32.903, 0.9094, None]}),
    "brain_r4_30db": dict(
        src="mri/meanflow_4x_30",
        desc="fastMRI brain - multi-coil CS-MRI, R=4, 30 dB",
        metrics={"n": 100, "metric": "psnr_dr/ssim_dr", "psnr_in": 25.61,
                 "M1": [32.890, 0.8992, None], "M4": [33.801, 0.9196, None],
                 "M16": [34.058, 0.9246, None], "M100": [34.128, 0.9259, None]}),
    "brain_r8_20db": dict(
        src="mri/meanflow_8x_20",
        desc="fastMRI brain - multi-coil CS-MRI, R=8, 20 dB",
        metrics={"n": 100, "metric": "psnr_dr/ssim_dr", "psnr_in": 21.85,
                 "M1": [28.109, 0.8187, None], "M4": [29.066, 0.8440, None],
                 "M16": [29.346, 0.8506, None]}),
    "brain_r8_30db": dict(
        src="mri/meanflow_8x_30",
        desc="fastMRI brain - multi-coil CS-MRI, R=8, 30 dB",
        metrics={"n": 100, "metric": "psnr_dr/ssim_dr", "psnr_in": 22.00,
                 "M1": [29.123, 0.8451, None], "M4": [29.888, 0.8651, None],
                 "M16": [30.096, 0.8699, None], "M100": [30.152, 0.8712, None]}),
}

# Keys the code actually reads; everything else in a run's config is dropped.
KEEP = {
    "experiment": ["stage", "phase", "seed", "run_id"],
    "paths": ["output_root"],
    "data": ["root", "loader", "size", "image_size", "grayscale", "normalize", "recursive",
             "center_crop", "crop_size", "max_len", "augment_hflip",
             "train_lmdb", "val_lmdb", "test_lmdb", "synthetic_maps", "syn_coils", "syn_seed",
             "syn_support_gate", "syn_support_thr", "syn_support_smooth", "syn_support_feather",
             "raw_dir", "maps_dir", "index_json", "crop", "slice_start", "slice_end",
             "max_coils", "cache_dir"],
    "dataloader": ["batch_size", "num_workers", "pin_memory", "shuffle_train"],
    "distributed": ["use_ddp", "backend", "gpus", "master_addr", "master_port"],
    "logging": ["log_every", "validate", "save_every", "save_img_every", "keep_last_n", "val_seed"],
    "training": ["epochs", "steps_per_epoch", "lr", "lr_min", "lr_schedule", "weight_decay",
                 "grad_clip", "grad_accum_steps", "use_ema", "ema_decay", "ema_start_epoch"],
    "snap": ["source", "cond_anchor", "cond_coverage", "tau", "p_ratio", "corner_frac",
                 "logit_mu", "logit_sigma", "weight_p", "weight_c", "weight_norm",
                 "weight_norm_decay", "beta", "gamma_max", "t_eps"],
    "model": ["resume_path", "in_ch", "out_ch", "input_height", "ch", "ch_mult",
              "num_res_blocks", "attn_resolutions", "dropout", "resamp_with_conv",
              "cond_r", "cond_sigma", "sigma_emb_scale"],
    "sample": ["steps"],
}

MRI_PATHS = {"raw_dir": "./data/fastmri_brain_multicoil",
             "maps_dir": "./data/fastmri_brain_multicoil/real/"
                         "acceleration_rate_1_smps_hat_method_eps",
             "index_json": "./data/index_AXT2_320.json",
             "cache_dir": "./data/fastmri_brain_cache320",
             "train_lmdb": "./data/mri/knee_train_lmdb",
             "val_lmdb": "./data/mri/knee_val_lmdb",
             "test_lmdb": "./data/mri/knee_test_lmdb"}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sanitize(cfg: dict, model: str) -> dict:
    out = {}
    for section, keys in KEEP.items():
        src = cfg.get(section) or ({} if section != "snap" else cfg.get("meanflow") or {})
        blk = {k: src[k] for k in keys if k in src}
        if blk:
            out[section] = blk
    stage = cfg["experiment"]["stage"]
    out["methods"] = {stage: dict(cfg["methods"][stage])}     # only this model's operator
    out["experiment"]["phase"] = "test"
    out["experiment"]["run_id"] = None
    out["paths"]["output_root"] = "./results"
    out["model"]["resume_path"] = None
    out["distributed"].update({"use_ddp": False, "gpus": [0]})
    out["data"]["max_len"] = None
    if model.startswith("celeba"):
        out["data"]["root"] = "./data/celeba"
    elif model.startswith("afhq"):
        out["data"]["root"] = "./data/afhq_cat"
    else:
        for k, v in MRI_PATHS.items():
            if k in out["data"]:
                out["data"][k] = v
    return out


def stage_checkpoint(src, dst, strip=True):
    """Copy a checkpoint, dropping optimizer/scheduler state unless it is already absent.

    Returns (epoch, weights_kind, stripped). When stripping, the saved network tensors are
    compared against the source element by element -- a release checkpoint that is not the
    trained one is the single worst thing this script could produce.
    """
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        done = torch.load(dst, map_location="cpu", weights_only=False)
        if done.get("epoch") == ck.get("epoch") and set(done) <= set(ck):
            return int(ck.get("epoch", -1)), ("ema" if ck.get("ema_model") is not None
                                              else "model"), ("optimizer" in ck)
    extra = [k for k in ("optimizer", "scheduler") if k in ck]
    kind = "ema" if ck.get("ema_model") is not None else "model"
    if not (strip and extra):
        shutil.copy2(src, dst)
        return int(ck.get("epoch", -1)), kind, False
    keep = {k: ck[k] for k in ("epoch", "model", "ema_model", "snap_meta", "meanflow_meta")
            if k in ck}
    torch.save(keep, dst)
    back = torch.load(dst, map_location="cpu", weights_only=False)
    for key in ("model", "ema_model"):
        if key not in keep:
            continue
        a, b = ck[key], back[key]
        assert a.keys() == b.keys(), f"{dst}: {key} tensor names changed"
        for t in a:
            assert torch.equal(a[t], b[t]), f"{dst}: {key}[{t}] changed while stripping"
    return int(ck.get("epoch", -1)), kind, True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints-root", required=True,
                    help="curated tree holding celeba/, afhq/, mri/ (<name>.pt + <name>.yaml)")
    ap.add_argument("--weights-dir", default="dist/weights", help="staging dir for upload")
    ap.add_argument("--base-url", default=None,
                    help="where the weights are served from (default: keep the manifest's)")
    ap.add_argument("--no-strip", action="store_true",
                    help="keep optimizer/scheduler state in the released files")
    a = ap.parse_args()

    os.makedirs(a.weights_dir, exist_ok=True)
    os.makedirs(os.path.join(HERE, "configs", "pretrained"), exist_ok=True)

    entries = {}
    for name, spec in MODELS.items():
        ck_src = os.path.join(a.checkpoints_root, spec["src"] + ".pt")
        cfg_src = os.path.join(a.checkpoints_root, spec["src"] + ".yaml")
        if not (os.path.exists(ck_src) and os.path.exists(cfg_src)):
            print(f"[skip] {name}: missing {ck_src if not os.path.exists(ck_src) else cfg_src}")
            continue
        clean = sanitize(yaml.safe_load(open(cfg_src)), name)

        cfg_rel = f"configs/pretrained/{name}.yaml"
        with open(os.path.join(HERE, cfg_rel), "w") as f:
            f.write(f"# Frozen config for the pretrained checkpoint `{name}`.\n"
                    f"# {spec['desc']}\n"
                    f"# Generated by scripts/make_release_assets.py from the config the model was\n"
                    f"# trained with -- edit the dataset paths, nothing else.\n")
            yaml.safe_dump(clean, f, sort_keys=False, default_flow_style=False)

        dst = os.path.join(a.weights_dir, f"{name}.pt")
        epoch, kind, stripped = stage_checkpoint(ck_src, dst, strip=not a.no_strip)
        digest, size = sha256(dst), os.path.getsize(dst)
        entries[name] = {
            "file": f"{name}.pt", "sha256": digest, "size": size,
            "config": cfg_rel, "stage": clean["experiment"]["stage"],
            "dataset": ("celeba" if name.startswith("celeba") else
                        "afhq_cat" if name.startswith("afhq") else "fastmri_brain"),
            "epoch": epoch, "weights": kind, "source": spec["src"] + ".pt",
            "description": spec["desc"], "metrics": spec["metrics"],
        }
        print(f"[ok] {name:15s} {size/2**20:7.1f} MB  ep={epoch:<4d} {kind}"
              f"{'  (stripped optimizer state)' if stripped else ''}  sha256={digest[:16]}...")

    base_url = a.base_url
    if base_url is None:
        try:
            base_url = json.load(open(os.path.join(HERE, "checkpoints", "manifest.json")))["base_url"]
        except (OSError, KeyError, ValueError):
            base_url = "REPLACE_ME"
    manifest = {
        "_comment": ("Checkpoint index for scripts/download_checkpoints.py. Set `base_url` to "
                     "wherever the .pt files are served from; the downloader fetches "
                     "<base_url>/<file> and verifies sha256."),
        "base_url": base_url,
        "models": entries,
    }
    mpath = os.path.join(HERE, "checkpoints", "manifest.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(os.path.join(a.weights_dir, "SHA256SUMS"), "w") as f:
        for n, e in entries.items():
            f.write(f"{e['sha256']}  {e['file']}\n")
    print(f"\nwrote {mpath}\nwrote {a.weights_dir}/SHA256SUMS ({len(entries)} files, "
          f"{sum(e['size'] for e in entries.values())/2**30:.2f} GB)")


if __name__ == "__main__":
    main()
