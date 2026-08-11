# PhySTA / MoNet-Phy

Reference implementation for **PhySTA**, a physics-inspired spatio-temporal learning framework for graph-structured forecasting and arbitrary inference.

**Paper:** Enabling Arbitrary Inference in Spatio-Temporal Dynamic Systems: A Physics-Inspired Perspective

**Authors:** Yan Ge, Zhengyang Zhou, Qihe Huang, Yuxuan Liang, and Yang Wang

This repository contains the cleaned experimental code used for the ICLR 2026 submission. It focuses on the `Mask=0` forecasting setting and includes scripts for reproducing the main results on PEMS-BAY, SD, KnowAir, and BJAir.

## Repository Layout

```text
.
|-- experiments/monet/
|   |-- main.py                  # Main train/evaluation entrypoint
|   |-- config.yaml              # Base model configuration
|   `-- run_best_experiments.py  # Paper/best-parameter reproduction runner
|-- src/
|   |-- base/                    # Training engine and model base classes
|   |-- engines/                 # MoNet/PhySTA training engine
|   |-- models/                  # Model components
|   `-- utils/                   # Data loading, metrics, logging
|-- data/                        # Local datasets; ignored by git
|-- eval/                        # Local logs/results; ignored by git
|-- CITATION.cff                 # GitHub citation metadata
|-- CITATION.md                  # Citation and result-reporting guidance
|-- CONTRIBUTING.md              # Contribution and artifact-boundary guidance
|-- DATASETS.md                  # Dataset layout and validation guide
|-- RELEASE.md                   # Final release checklist
|-- TROUBLESHOOTING.md           # Common setup and reproduction issues
|-- requirements.txt
`-- REPRODUCIBILITY.md
```

Large datasets, checkpoints, logs, generated figures, and exploratory notebooks are intentionally excluded from version control.

## Environment

Python 3.9+ is recommended. The experiments in this workspace were run with a Conda environment named `monet`.

```bash
conda create -n monet-phy python=3.9
conda activate monet-phy
pip install -r requirements.txt
```

The final local release checks used Python 3.9.21, PyTorch 2.0.1, CUDA 11.7,
PyG 2.6.1, NumPy 1.26.4, and SciPy 1.13.1 on an NVIDIA RTX 3060 Laptop GPU.
These versions are a tested reference, not a requirement for exact metric equality.

`torch-scatter` and `torch-geometric` are sensitive to the installed PyTorch/CUDA versions. If the generic install fails, install the matching PyG wheels from the official PyG instructions first, then rerun `pip install -r requirements.txt`.

## Data Layout

Put each dataset under `data/<DATASET>/`. Each directory must contain:

```text
data/<DATASET>/
|-- his.npz          # contains data, mean, std
|-- idx_train.npy
|-- idx_val.npy
|-- idx_test.npy
`-- adjacency file
```

Supported dataset keys and required input dimensions:

| Dataset | Input dim | Adjacency file |
|---|---:|---|
| `PEMS-BAY` | 3 | `data/PEMS-BAY/adj_mx_bay.pkl` |
| `SD` | 3 | `data/sd/sd_rn_adj.npy` |
| `KnowAir` | 15 | `data/KnowAir/adj_matrix.npy` |
| `BJAir` | 18 | `data/BJAir/BJAir.npy` |

`input_dim` is selected automatically from the dataset name in `experiments/monet/main.py`. This is intentional: traffic datasets use value/time features, while air-quality datasets include additional covariates.

See [DATASETS.md](DATASETS.md) for detailed dataset shape, adjacency, validation, and git-tracking guidance.

Before running long experiments, validate the local files and shapes:

```bash
python scripts/check_data.py --datasets PEMS-BAY SD KnowAir BJAir
```

## Quick Smoke Test

Use this only to verify that the environment and data are wired correctly:

```bash
python experiments/monet/main.py \
  --dataset PEMS-BAY \
  --device cpu \
  --bs 2 \
  --max_epochs 1 \
  --patience 1 \
  --emd_dim 4 \
  --gfno_hidden 8 \
  --energy_splits 0.3 0.7
```

This is not a paper reproduction setting.

## Reproduce Paper-Scale Runs

Long-running experiments should be launched manually in your own terminal. Run the curated best-known configurations with:

```bash
python experiments/monet/run_best_experiments.py --datasets PEMS-BAY SD KnowAir
```

The runner also supports BJAir:

```bash
python experiments/monet/run_best_experiments.py --datasets BJAir
```

Useful options:

```bash
python experiments/monet/run_best_experiments.py --dry_run
python experiments/monet/run_best_experiments.py --datasets SD --device cuda:0
python experiments/monet/run_best_experiments.py --datasets KnowAir BJAir --continue_on_error
```

Outputs are written to:

```text
eval/monet/best_runs/results.csv
eval/monet/best_runs/results.json
eval/monet/best_runs/<DATASET>_stdout.log
```

To summarize completed runs and compare them with the paper Table 1 `Mask=0` metrics:

```bash
python scripts/summarize_results.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact Table 1 comparison, dataset-specific commands, and currently verified results.

Small numerical differences from the paper are expected across hardware, CUDA/PyTorch versions, random seeds, and early-stopping points. The provided scripts report deltas for transparency; they do not enforce exact metric equality.

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common environment, dataset, and reproduction issues.

## Single-Dataset Training

Example:

```bash
python experiments/monet/main.py \
  --dataset PEMS-BAY \
  --device cuda:0 \
  --bs 32 \
  --seq_len 12 \
  --horizon 12 \
  --max_epochs 80 \
  --patience 15 \
  --lrate 0.002 \
  --wdecay 0.00001 \
  --cl_epoch 3 \
  --warm_epoch 30 \
  --emd_dim 32 \
  --gfno_hidden 32 \
  --energy_splits 0.7 0.95 \
  --topk_edges 3 \
  --ecc_layers 1 \
  --mask_ratio 0
```

## Notes for Open-Source Use

- Experiment logs are written to `eval/<model>/<dataset>/`.
- The training engine reloads the best validation state before final test evaluation.
- The directed Laplacian eigendecomposition is initialized on CPU to avoid CUDA complex-kernel compatibility issues.
- Air-quality datasets require the covariate side branch in `src/models/monet.py`; do not remove it when simplifying the model.
- Run `python scripts/validate_release.py` before publishing to check required files, Python syntax, reproduction dry-run, and accidental tracked artifacts.
- Run `python scripts/validate_release.py --check_data` on the machine that has the datasets to include local data integrity checks.
- Run `python scripts/validate_release.py --strict_results` after manually completing final Table 1 reruns to ensure paper datasets were evaluated with best-validation-state reload. This check validates result provenance, not exact metric equality.
- See [RELEASE.md](RELEASE.md) for the final automated and manual release gates.
- The repository currently includes an MIT license; replace it before release if a different license is required.

## Citation and Contributions

Use [CITATION.md](CITATION.md) for the current submission citation and replace it with the proceedings entry once public. GitHub citation metadata is provided in `CITATION.cff`. See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes or reporting reproduced metrics.
