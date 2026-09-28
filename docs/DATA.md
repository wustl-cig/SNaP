# Datasets

## CelebA and AFHQ-Cat (automatic)

```bash
python scripts/download_data.py celeba     # 1.4 GB archive -> data/celeba
python scripts/download_data.py afhq       # 0.7 GB archive -> data/afhq_cat
python scripts/download_data.py all --verify-only
```

Layout produced:

```
data/celeba/{train,validation,test}/000001.jpg ...      162,770 / 19,867 / 19,962
data/afhq_cat/{train,validation,test}/flickr_cat_*.jpg    5,153 /    500 /    100
```

The loader crops and resizes on the fly (`data.center_crop`, `data.crop_size`,
`data.size`), so the files on disk stay in their original resolution -- CelebA 178×218
aligned JPEGs, AFHQ 512×512.

**Verification.** After extraction the script compares the install against
`scripts/data_manifest.json`: per split, the file count, the sha256 of the sorted file
list, and the sha256 of four sampled images. Those hashes come from the data the reported
numbers were computed on, so any mirror drift is caught immediately. A count or list
mismatch means a wrong source or a partial extraction; a *sample* mismatch means the
images were re-encoded, which typically moves PSNR by a few hundredths of a dB.

**Offline / mirrored installs.** If your cluster already has the archives:

```bash
python scripts/download_data.py celeba --archive /path/to/Dataset.zip
python scripts/download_data.py afhq   --archive /path/to/afhq.zip
```

Any zip works as long as it contains the original file names (`000001.jpg ...` for CelebA,
`afhq/train/cat/*.jpg` and `afhq/val/cat/*.jpg` for AFHQ); the verification step is what
tells you whether it was the right one.

**Licenses.** CelebA is for non-commercial research only; AFHQ is CC BY-NC 4.0. This
repository ships no dataset images -- the script downloads them from the original
distributors' mirrors, and their terms apply to your copy.

## fastMRI brain (manual)

fastMRI requires an accepted data-use agreement, so it cannot be downloaded by a script.

1. Request access at <https://fastmri.med.nyu.edu/> and download the **brain multicoil**
   set.
2. Put the volumes under `data/fastmri_brain_multicoil/` and the pre-computed ESPIRiT
   sensitivity maps under
   `data/fastmri_brain_multicoil/real/acceleration_rate_1_smps_hat_method_eps/`.
   (`data.maps_dir` in `configs/mri_brain.yaml` points there; maps are per slice.)
3. Provide `data/index_AXT2_320.json`, the volume-level train/validation/test split:
   615 / 77 / 77 AXT2 volumes at 320×320, each file being a distinct patient. Slices 0-3
   and the last 5 of every volume are dropped (`data.slice_start` / `data.slice_end`),
   giving 4,305 / 539 / 539 slices.
4. Build the slice cache -- this is effectively mandatory, not an optimisation: reading
   raw volumes at train time pulls ~100 MB per sample at random offsets and leaves
   training about 30× I/O bound.

   ```bash
   python dataset/build_brain_cache.py --config configs/mri_brain.yaml --workers 8
   ```

   The cache holds raw, unnormalised, unpadded arrays, so `data.normalize` and
   `data.max_coils` can change without rebuilding it.

Then train or evaluate as usual:

```bash
python run.py --config configs/mri_brain.yaml --set experiment.run_id=my_brain
python eval_avg_mri.py --config configs/pretrained/brain_r8_30db.yaml \
    --ckpt checkpoints/brain_r8_30db.pt --ids data_splits/brain_test100.txt \
    --M 1 4 16 --out outputs/eval_brain
```

## fastMRI knee (LMDB)

`configs/mri.yaml` reads the knee data in the LMDB layout used by the InverseBench
benchmark (`mvue/` per split, `s_maps/` for validation and test). No pretrained checkpoint
ships for it: the benchmark's training LMDB has no sensitivity maps, so training there
falls back on synthetic birdcage maps and the network specialises to that operator. The
brain config is the one to use for real multi-coil training. Install the MRI extras with
`pip install -r requirements-mri.txt`.
