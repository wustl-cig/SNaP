# Fixed evaluation subsets (MRI)

`brain_test100.txt` and `brain_val20.txt` are dataset indices into the fastMRI brain
`test` and `validation` splits, as comma-separated integers. They are the subsets every
reported MRI number was computed on, and they are stratified over all 77 test volumes.

Pass one with `--ids` instead of `--n`:

```bash
python eval_avg_mri.py --config configs/pretrained/brain_r8_30db.yaml \
    --ckpt checkpoints/brain_r8_30db.pt --ids data_splits/brain_test100.txt \
    --M 1 4 16 --out outputs/eval_brain_r8_30db
```

`--n 100` takes the FIRST 100 slices of the split instead, which on this data is roughly
14 of the 77 patients -- a different, much narrower sample. Use `--ids` for anything
compared against the table in the README.

The indices depend on the split defined by `data.index_json` (615/77/77 volumes, slices
4..-5 dropped); they are meaningless against a differently built split.
