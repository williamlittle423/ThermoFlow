# ThermoFlow

**Conditional generative inference for three-dimensional thermal tomography.**

ThermoFlow investigates reconstruction of internal thermal conductivity from a surface temperature sequence following a prescribed heat pulse. The research objective is to learn a conditional distribution of conductivity fields using flow matching and evaluate it against deterministic reconstruction methods.

Heat diffusion attenuates internal spatial structure before it reaches the observed boundary. Finite recording duration, spatial resolution, and measurement noise can therefore make reconstruction unstable or ambiguous. The central question is whether conditional generation can recover material structure while representing uncertainty in features that the measurements constrain weakly.

The current implementation provides synthetic data generation, geometry-based dataset partitioning, and visualization. Model training and probabilistic evaluation are planned components.

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

`--cubes`, `--cube-cells`, `--nx`, `--ny`, `--nz`, `--dt`, `--heat-flux`, and `--heat-capacity` control the simulation. The CLI defaults to two cubes; the command above explicitly selects the initial single-cube benchmark. Use a separate output directory when changing physical or numerical settings. Larger timesteps require an accuracy comparison, particularly for the early frames.

### Sample format

Each `sample_XXXX.npz` contains:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `video` | `(20, ny, nx)`, float32 | Surface temperature rise in K |
| `conductivity` | `(nz, ny, nx)`, float32 | Conductivity in W/(m·K) |
| `frame_times` | `(20,)` | Seconds after heater switch-off |
| `cell_size` | Scalar | Cell width in metres |
| `seed`, `sample_index` | Scalars | Specimen identifiers |
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
