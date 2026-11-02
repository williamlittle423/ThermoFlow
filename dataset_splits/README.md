# ThermoFlow clustered dataset split

The splitter uses k-means++ on the cube-center coordinates (x, y, z), standardized to give each axis equal variance. It fits ten clusters to **unique conductivity geometries**, with 30 initializations. Duplicate specimens inherit the same cluster and split. A seeded permutation selects one cluster for validation and one for testing; the remaining eight train the model. Cluster choices are independent of model results.

This is a **spatial holdout experiment**, rather than a conventional IID random 80/10/10 split. Validation/test cubes occupy held-out geometric regions. K-means does not impose equal group sizes and does not enforce a minimum distance between training and held-out positions at cluster boundaries. Adjacent grid positions may be on opposite sides of a boundary. For an ordinary interpolation benchmark, use a separate geometry-grouped random split instead.

Ground-truth positions define the benchmark split; they are not model inputs. Clustering uses no thermal-video features or reconstruction results. Model normalization and tuning must still use the training and validation partitions only.

Files under `dataset_splits`:

- `groups/cluster_00.txt` through `cluster_09.txt`: the ten groups.
- `train.txt`, `validation.txt`, `test.txt`: combined partition lists. Each line is a specimen filename relative to the original `dataset` directory.
- `assignments.csv`: specimen, full conductivity-geometry hash, cluster, split, and cube-center coordinates in millimetres.
- `summary.json`: counts, feature scaling, centroids, seed and selected clusters.
- `split_overview.png`: geometric cluster and partition views.

The original NPZ files are unchanged. The lists are a frozen snapshot of files present when the script starts. New samples are not silently added. Do not refit or repeatedly choose holdout clusters based on model performance. Preserve the published split for comparisons.

To construct a separate new split before model evaluation:

```bash
python cluster_dataset.py dataset --output dataset_splits_new --seed 42
```

The script refuses to overwrite an existing split. It requires exactly one solid axis-aligned cube of a common size and consistent simulation settings. A dataset with multiple inclusions will need a richer geometry descriptor or another grouping scheme.
