# Troubleshooting

This page covers common setup and reproduction issues for MoNet-Phy.

## PyG or Torch-Scatter Fails to Install

`torch-geometric` and `torch-scatter` must match the installed PyTorch and CUDA versions. If `pip install -r requirements.txt` fails on these packages, install the matching PyG wheels for your PyTorch/CUDA build first, then rerun:

```bash
pip install -r requirements.txt
```

Run the release smoke checks after installation:

```bash
python scripts/validate_release.py --skip_dry_run
```

## Dataset File Is Missing

Typical error:

```text
FileNotFoundError: ... data/sd/sd_rn_adj.npy
```

Check the expected layout:

```bash
python scripts/check_data.py --datasets SD
```

See [DATASETS.md](DATASETS.md) for required paths, node counts, feature dimensions, and adjacency files.

## Wrong Input Dimension

The model enforces dataset-specific feature dimensions:

```text
PEMS-BAY: 3
SD: 3
KnowAir: 15
BJAir: 18
```

`experiments/monet/main.py` overrides `--input_dim` from the dataset name. This is intentional because the air-quality datasets use additional covariates.

## CUDA Complex Kernel Error During Laplacian Initialization

If an older CUDA/PyTorch combination fails around complex phase construction or Laplacian eigendecomposition, use the current code path. The directed Laplacian initialization is performed on CPU before tensors are moved back to the model device.

Verify that core imports and the release checks pass:

```bash
python scripts/validate_release.py
```

## Training Appears Silent or Stuck

Long paper-scale runs can take a while, especially on large datasets. Use the reproduction runner rather than an IDE run button when possible:

```bash
python experiments/monet/run_best_experiments.py --datasets PEMS-BAY --device cuda:0
```

The runner streams stdout and writes logs under:

```text
eval/monet/best_runs/
```

Single-dataset logs are also written under:

```text
eval/monet/<DATASET>/
```

## KnowAir or BJAir Reports Boundary Index Warnings

`scripts/check_data.py` may report boundary-index warnings for KnowAir or BJAir. This means a small number of split indices fall outside the `seq_len=12`, `horizon=12` sampling window.

The runtime DataLoader filters these samples before batching, so the default data check treats them as warnings. Use strict mode only when auditing regenerated split files:

```bash
python scripts/check_data.py --datasets KnowAir BJAir --strict_indices
```

## Results Differ Slightly From the Paper

Small metric differences are expected across hardware, CUDA/PyTorch versions, random seeds, and early-stopping points. Use:

```bash
python scripts/summarize_results.py --no_write
```

Report the observed values and deltas transparently. The strict release check validates provenance, not exact numerical equality:

```bash
python scripts/validate_release.py --strict_results
```

## Final Release Still Fails Strict Results

`--strict_results` requires the latest completed paper-dataset runs to include the current best-validation reload path. If it fails for a dataset, rerun that dataset with the current code and then summarize again.
