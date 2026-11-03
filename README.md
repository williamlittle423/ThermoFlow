# ThermoFlow

**Conditional generative inference for three-dimensional thermal tomography.**

ThermoFlow investigates reconstruction of internal thermal conductivity from a surface temperature sequence following a prescribed heat pulse. The research objective is to learn a conditional distribution of conductivity fields using flow matching and evaluate it against deterministic reconstruction methods.

Heat diffusion attenuates internal spatial structure before it reaches the observed boundary. Finite recording duration, spatial resolution, and measurement noise can therefore make reconstruction unstable or ambiguous. The central question is whether conditional generation can recover material structure while representing uncertainty in features that the measurements constrain weakly.

The current implementation provides synthetic data generation, dataset partitioning, visualization, and a deterministic ThermoNet reconstruction baseline. Conditional flow matching and probabilistic evaluation are planned components.

## Physical model and numerical method

Temperature rise relative to a uniform initial state satisfies

$$
C\frac{\partial T}{\partial t}=\nabla\cdot\left(k(\mathbf{x})\nabla T\right),
\qquad T(\mathbf{x},0)=0,
$$

where conductivity $k$ is heterogeneous and locally isotropic, and volumetric heat capacity $C$ is fixed. A uniform inward heat flux is applied at the observed face, $z=0$, for five seconds. The other faces are insulated during heating; all boundaries are insulated after switch-off. Post-pulse surface cooling represents redistribution into the slab, rather than heat loss to the environment.

| Parameter | Default |
| --- | --- |
| Slab dimensions | 24 × 24 × 6 mm |
| Grid | 48 × 48 × 12 cells, with 0.5 mm spacing |
| Inclusion geometry | Axis-aligned cubes, 2 mm side length |
| Background / inclusion conductivity | 0.20 / 0.50 W/(m·K) |
| Volumetric heat capacity | 1.5 × 10⁶ J/(m³·K) |
| Heating | 800 W/m² for 5 s |
| Recording | 20 frames from 0 to 60 s after switch-off |
| Maximum internal timestep | 0.5 s |

The generator constructs geometry with FiPy and integrates a conservative finite-volume discretization using SciPy sparse linear algebra. Harmonic face conductivities represent material interfaces. Backward Euler advances the solution, landing exactly on heating and recording boundaries. The sparse operator is assembled once per specimen, and LU factorizations are reused for each distinct timestep. Surface temperatures are extrapolated from the first two cell layers under the zero-flux boundary condition.

Cubes are sampled on the grid, with a two-cell background layer at every exterior face. Multiple cubes are separated by at least one background cell. Energy checks verify the injected heat and conservation during insulated redistribution. Convection, radiation, emissivity variation, and a detailed camera response are outside the current forward model.

## Installation

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data generation

Generate or resume a single-cube dataset:

```bash
python construct_toy_dataset.py \
  --count 1000 --cubes 1 --workers 2 \
  --output dataset --resume
```

Sample index `i` uses `--seed + i`. `--count 1000` requests indices 0–999; `--resume` verifies existing settings and skips compatible files. Workers simulate independent specimens and write compressed NPZ files through temporary files followed by atomic rename. Stop an earlier run before launching another against the same indices.

`--cubes` (fixed count) or `--cube-count-range MIN MAX` (inclusive uniform count per specimen), `--cube-cells`, `--nx`, `--ny`, `--nz`, `--dt`, `--heat-flux`, and `--heat-capacity` control the simulation. The CLI defaults to two cubes; the command above explicitly selects the initial single-cube benchmark. Use a separate output directory when changing physical or numerical settings. Larger timesteps require an accuracy comparison, particularly for the early frames.

### Concurrent cohort benchmark

Run the three planned cohorts together: one cube, two cubes, and a uniformly sampled count from 3 through 10. Each cohort has its own output directory and independent seed offset. Counts and positions are deterministic across worker counts and resumed runs. The realized count is saved as `n_cubes` and in `config_json`; mixed samples also record the sampling range and method. Placement failures are reported, never replaced with a smaller count.

On a compute node with at least six CPU cores available (eight gives room for controller overhead):

```bash
python benchmark_cluster.py --samples 50 --workers-per-group 2 \
  --target 2500 --output cluster_benchmark
```

This generates **50 specimens per cohort**, with six solver workers running concurrently. Numerical-library threads are limited to one per worker. Separate logs and a `report.json` record wall time, throughput, realized cube-count histograms, and projected durations for 2,500 specimens per cohort. The combined projection is the maximum cohort duration because all three run concurrently. Extrapolations assume comparable resource and filesystem conditions; they are not guaranteed completion times.

The script prints resume commands that extend the same cohort folders to 2,500 specimens, reusing the benchmark samples. Launch those commands concurrently inside the same kind of allocation if extending the run. An existing benchmark output directory is rejected to avoid misleading timings. Use a new directory for each throughput experiment. The driver does not request scheduler resources and stays on one node; a cluster's total core count is not the same as an individual job's allocation.

### Sample format

Each `sample_XXXX.npz` contains:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `video` | `(20, ny, nx)`, float32 | Surface temperature rise in K |
| `conductivity` | `(nz, ny, nx)`, float32 | Conductivity in W/(m·K) |
| `frame_times` | `(20,)` | Seconds after heater switch-off |
| `cell_size` | Scalar | Cell width in metres |
| `seed`, `sample_index` | Scalars | Specimen identifiers |
| `n_cubes` | Scalar | Realized inclusion count (newly generated samples) |
| `config_json` | JSON string | Acquisition and simulation settings |

Depth increases away from the observed surface. Frame zero is the end of the heating pulse, after the interior has already influenced the thermal response. The generator saves clean observations; measurement noise can be added during training or evaluation without changing the underlying specimen.

## Dataset partitions

The committed split covers 281 specimens and 276 unique interiors:

| Partition | Clusters | Samples |
| --- | --- | --- |
| Training | 8 | 221 |
| Validation | 1 | 31 |
| Testing | 1 | 29 |

Ten k-means++ clusters are fitted to standardized single-cube center coordinates, using 30 initializations and seed 42. Duplicate interiors share an assignment. One cluster each is reserved for validation and testing; the remaining eight form the training set.

This is a **spatial holdout benchmark**, rather than an IID random split. It evaluates recovery in held-out position regions, although neighboring positions can lie on opposite sides of a cluster boundary. Cluster sizes are not constrained to be equal.

`dataset_splits/` contains partition lists, individual cluster lists, specimen assignments, geometry hashes, and clustering metadata. List entries are filenames relative to `dataset/`. This snapshot is fixed; newly generated specimens are not automatically included. To construct a separate partition before a new experiment:

```bash
python cluster_dataset.py dataset --output dataset_splits_new --seed 42
```

Preserve the selected partitions throughout model comparison. Fit model preprocessing on training data and use validation data for tuning.

## ThermoNet baseline training

`ThermoNet` is a compact 2D U-Net that maps recording times to input channels and depth slices to output channels: `(B, 20, 48, 48)` videos become `(B, 12, 48, 48)` inclusion logits. `predict_proba()` applies sigmoid; with the default materials, a conductivity estimate is `0.20 + 0.30 * probability`.

```bash
python train_thermonet.py --data datasets/single_cube \
  --output runs/single_cube --epochs 100 --batch-size 16 --cache
```

The script uses CUDA, Apple MPS, or CPU according to availability. Set `--device cuda` to require an NVIDIA GPU. `--cache` stores the clean videos and masks in system RAM. For a short CPU pipeline check, use a new output directory with `--limit 32 --epochs 2 --base-channels 8 --device cpu`.

Training constructs a new **geometry-grouped random 70/15/15 split** of the supplied single-cube dataset, keeping duplicate interiors together. It does not reuse the earlier 281-specimen spatial holdout. The frozen split, specimen content hashes and acquisition settings are saved in `manifest.json`. All specimens must contain one solid axis-aligned cube with consistent grid, material, and acquisition settings.

A global temperature mean and standard deviation are fitted on training data only; there is no per-video min/max scaling. The objective is class-weighted binary cross-entropy with logits plus soft Dice. Class weighting is calculated from the training voxel counts. `--noise-std 0.05` adds Gaussian observation noise in kelvin before normalization: training noise changes each epoch, while validation/test noise remains fixed. Start a separate run when changing the noise setting.

Validation loss controls best-checkpoint selection and early stopping. The test set is evaluated after training using the best validation checkpoint. Outputs include `best.pt`, `last.pt`, `history.json`, final test metrics, per-specimen scores, and a breakdown by true inclusion front depth. Reported centroid depth error is conditional on a nonempty thresholded prediction; the evaluated fraction is recorded so missed defects are not counted as successful localization.

Resume an interrupted run with matching training settings:

```bash
python train_thermonet.py --data datasets/single_cube \
  --output runs/single_cube --epochs 100 --batch-size 16 --cache \
  --resume runs/single_cube/last.pt
```

The checkpoint restores model and optimizer state, history, and the validation stopping state. It must match the dataset and training settings. A new run refuses to overwrite a nonempty output directory. GPU execution is not claimed to be bitwise reproducible across hardware.

For Colab, open `train_thermonet_colab.ipynb` and follow its setup cells. The notebook stages an archive onto the runtime's local disk and saves checkpoints to Drive. Only NumPy and PyTorch are needed for the training script; FiPy is not imported. Training outputs are excluded from Git.

### All-cohort training

Train a fresh model on the single-, two-, and 3–10-cube cohorts:

```bash
python train_thermonet_all.py --data datasets \
  --single-checkpoint best_single.pt --output runs/all_cohorts \
  --epochs 100 --batch-size 16 --cache
```

The original checkpoint supplies split metadata only; its weights are not loaded. This preserves the earlier single-cube holdout. Other geometries are split by cube count with duplicate interiors kept together. Final metrics are reported per cohort and per cube count, including object recovery. Set `--data cluster_benchmark_2w` when using the original cohort folders on Moana. For CPU training, add `--device cpu --cpu-threads 8`; the default device selection also supports CUDA and MPS.

See [the all-cohort protocol](docs/all_cohort_training.md) and `train_thermonet_all_colab.ipynb` for setup and resume instructions. Checkpoints and datasets are transferred separately from Git.

## Visualization

```bash
# Paginated surface-contrast and inclusion-depth contact sheet
python contact_sheet.py dataset

# Inspect one specimen with a rotatable 3D interior and a frame slider
python visualize_slabs.py dataset/sample_0000.npz

# Export the surface sequence and interior as an animated GIF
python visualize_slabs.py dataset/sample_0000.npz --gif specimen.gif --no-show
```

The single-specimen viewer shows the 3D inclusion geometry, surface temperature rise, spatial contrast, and a surface-temperature trace. Drag the 3D view to rotate it and use the frame slider to inspect the recording. Temperature and contrast scales remain fixed across frames. Spatial contrast subtracts the frame mean; it is not a defect-free reference subtraction. Interior geometry and depth displays are simulation ground truth. The contact sheet uses shared scales across its displayed specimens.

To measure generation throughput and timestep sensitivity:

```bash
python benchmark_generation.py --count 4 --output benchmark_results
```

The benchmark compares one, two, and four workers, checks agreement across worker counts, and evaluates 1 s and 2 s timesteps against the 0.5 s reference at three inclusion depths. It writes a JSON report with timings and temperature/contrast errors. Use a new output directory for each benchmark.

Generated datasets, model artifacts, rendered previews, and local environments are excluded from version control. Source code and fixed partition definitions are tracked.
