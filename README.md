# ThermoFlow

My ICS 691C project on reconstructing an object's internal thermal conductivity from a video of its surface after heating. The plan is to train a conditional flow matching model and compare it with a deterministic baseline.

For now, this repo contains the simulation, dataset splitting, and visualization tools. Model training is next.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate data

```bash
python construct_toy_dataset.py --count 1000 --cubes 1 --workers 2 --output dataset --resume
```

Each sample contains a conductivity volume and 20 surface temperature frames over the 60 seconds after heating ends. The default slab is 24 × 24 × 6 mm, with 0.5 mm cells and 2 mm cubic inclusions. Samples are saved separately with their seeds and simulation settings. `--resume` checks and skips completed samples.

## Look at the samples

```bash
python contact_sheet.py dataset
python visualize_slabs.py dataset/sample_0000.npz
```

To build the interactive 3D viewer:

```bash
python build_viewer.py dataset --limit 100
```

Open `thermoflow_viewer.html` in a browser. It works offline.

## Dataset split

`dataset_splits/` contains the fixed split for 281 samples: 221 training, 31 validation, and 29 testing. Ten k-means clusters were formed from cube positions, with eight used for training and one each for validation and testing. Identical interiors stay together. This is a spatial holdout split, so it tests recovery in unseen position regions.

The file lists use filenames relative to `dataset/`. New samples are not automatically included. To create a separate split before starting a new experiment:

```bash
python cluster_dataset.py dataset --output dataset_splits_new --seed 42
```

The datasets and generated previews are left out of Git. The split definitions are included so experiments can use the same partitions.
