# Reproducing the tables

Every number in the README's model table comes from one command. Start with the data and
the weights:

```bash
python scripts/download_data.py all
python scripts/download_checkpoints.py --all
```

## Natural images

```bash
for M in celeba_inpaint celeba_box celeba_deblur celeba_sr \
         afhq_inpaint   afhq_box   afhq_deblur   afhq_sr; do
  python eval_avg.py --config configs/pretrained/$M.yaml \
      --ckpt checkpoints/$M.pt --n 100 --M 1 4 16 100 --out outputs/eval_$M
done
```

Each run prints the PSNR / SSIM / LPIPS row per `M` and writes

```
outputs/eval_<model>/avg_metrics.csv      one row per image, every M side by side
outputs/eval_<model>/k1/test_metrics.csv  M=1 alone, eval.py's columns
outputs/eval_<model>/k1/images/           0000_gt.png 0000_obs.png 0000_output.png
outputs/eval_<model>/k4/ k16/ k100/
```

The `M=1` and `M=100` columns of the README table are the `k1` and `k100` rows.

## MRI

```bash
for M in brain_r4_20db brain_r4_30db brain_r8_20db brain_r8_30db; do
  python eval_avg_mri.py --config configs/pretrained/$M.yaml \
      --ckpt checkpoints/$M.pt --ids data_splits/brain_test100.txt \
      --M 1 4 16 --out outputs/eval_$M
done
```

Each MRI model is tied to one (acceleration, SNR) cell: `brain_r8_20db` evaluated at R=4
is not a weaker result, it is the wrong operator. The config next to each checkpoint pins
`acceleration_ratio` and `input_snr_db`, so use it rather than overriding them.

MRI is scored on the magnitude image with a per-slice data range (`psnr_dr` / `ssim_dr`).
Do not compare those numbers against the RGB `psnr` column: the conventions differ, and
the complex-channel one is roughly 3 dB optimistic.

## What to expect

Reconstructions are stochastic -- each draw is a posterior sample -- so a rerun moves by
a few hundredths of a dB even with the seeds fixed, because the measurement noise and the
source draw consume the RNG in the same stream. Differences larger than ~0.05 dB point at
a different dataset copy (run `python scripts/download_data.py all --verify-only`) or a
different checkpoint (the downloader verifies sha256).

Averaging trades perceptual quality for distortion: PSNR and SSIM rise with `M` while
LPIPS usually gets worse. That is the posterior mean behaving as it should, not a
regression in the model.

## Retraining

```bash
# CelebA, ~200 epochs x 1,000 steps, 4 GPUs; tau per operator, see the config header
python run.py --config configs/celeba.yaml \
    --set experiment.stage=sr snap.tau=0.1 experiment.run_id=repro_sr

# AFHQ-Cat 256, effective batch 32 via grad_accum_steps=4
python run.py --config configs/afhq.yaml \
    --set experiment.stage=sr snap.tau=0.1 training.epochs=120 experiment.run_id=repro_afhq_sr
```

Then evaluate the resulting checkpoint with its own snapshotted config:

```bash
python eval_avg.py --config results/sr/train/repro_sr/config_used.yaml \
    --ckpt results/sr/train/repro_sr/checkpoints/snap_best.pt \
    --n 100 --M 1 4 16 100 --out outputs/eval_repro_sr
```

`snap_best.pt` is selected on the lowest training residual, which falls close to
monotonically -- so it is usually the latest epoch. For careful selection, read the
validation curve in the run's log and load the matching `snap_epoch_*.pt`.
