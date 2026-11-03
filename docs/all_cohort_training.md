# All-cohort ThermoNet baseline

This experiment trains a fresh model on `datasets/single_cube`, `datasets/two_cubes`, and `datasets/mixed_cubes` (7,500 specimens when each folder contains 2,500 files). It uses the same ThermoNet architecture and weighted BCE + soft Dice objective as the single-cube baseline.

## Colab

Open `train_thermonet_all_colab.ipynb` and select a GPU runtime. From the project directory, prepare the archive described in the notebook:

```bash
cp ~/Downloads/best.pt best_single.pt
zip -r -0 thermoflow_all.zip thermonet.py train_thermonet.py train_thermonet_all.py best_single.pt datasets/single_cube datasets/two_cubes datasets/mixed_cubes
```

The data must be locally readable when creating the archive. If the Mac copies are iCloud placeholders, use a downloaded copy or create the archive on Moana after transferring the Python files and original checkpoint there. Upload the archive to `My Drive/ThermoFlow/thermoflow_all.zip`. The notebook extracts it onto the runtime disk and writes results to `My Drive/ThermoFlow/runs/all_cohorts`.

## Command line

```bash
python train_thermonet_all.py --data datasets \
  --single-checkpoint best_single.pt \
  --output runs/all_cohorts --epochs 100 --batch-size 16 \
  --device cuda --num-workers 2 --cache
```

Use `--device cpu` for a small pipeline check. `--cache` stores clean video/mask arrays in system RAM (approximately 1.5 GiB for the default 7,500 specimens, plus loader and interpreter overhead).

`train_thermonet_all.py` imports common checkpoint, normalization, augmentation and loss utilities from the existing `train_thermonet.py`; all three Python files are required. Training also requires NumPy, SciPy and PyTorch, already listed in the project requirements.

## Partitions and comparison protocol

- All filenames are cohort-relative, e.g. `two_cubes/sample_0000.npz`, avoiding collisions between folders.
- Exact duplicate conductivity geometries are grouped globally before splitting.
- Supplying the old checkpoint preserves its single-cube training, validation and testing partitions exactly. Single-cube contents and acquisition settings must match the checkpoint manifest. The old weights are never imported.
- Other unique geometries are stratified by true cube count, then divided approximately 70/15/15 with a fixed seed. Actual specimen counts can differ because duplicates stay together.
- Without `--single-checkpoint`, the single-cube split is newly generated too; comparisons with the original single-cube test scores would then use a different holdout.
- All cohorts must share grid, cube size, material properties, recording times and acquisition settings.

Training normalization and class weights use the combined training partition only. No per-video min/max scaling is performed. Training noise is resampled deterministically each epoch; evaluation noise stays fixed. Start a separate output folder for a different noise setting.

The old six visualization examples may be included in the new training set. Use the new held-out test set to quantify performance; those six cases are now illustrations, not necessarily unseen examples.

## Outputs

Each run saves `manifest.json`, split lists, `normalization.json`, `config.json`, `history.json`, `best.pt`, `last.pt`, `test_metrics.json`, and `test_per_sample.csv`. The manifest includes geometry/content hashes, true cube counts, split distributions, and whether the original single-cube split was preserved.

The test report includes aggregate, per-cohort, and per-cube-count metrics. Voxel Dice, IoU, precision, recall and conductivity MSE are equal-weight specimen means. Object precision/recall use pooled one-to-one matches of 6-connected predicted components at IoU >= 0.5. The probability threshold is fixed at 0.5. Whole-volume centroid depth error is omitted because it can conceal missing individual inclusions.

Validation loss selects the best checkpoint and controls early stopping. Resume a run by repeating its command with `--resume runs/all_cohorts/last.pt`; keep `best.pt` alongside it. The notebook does this automatically when a latest checkpoint exists. Existing runs are not overwritten by a fresh start.

CPU fixture checks covered every cube count, original split preservation, duplicate isolation, cohort-safe paths, component scoring, checkpoint resume and exact agreement with uninterrupted CPU training. Full-data GPU training has not been run locally.
