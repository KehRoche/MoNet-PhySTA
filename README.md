# MoNet-Phy

Reference implementation for the ICLR submission codebase around **MoNet-Phy**, a spatio-temporal forecasting model that combines graph neural operators and multi-scale graph diffusion modules.

This repository is cleaned for code release. Large datasets, checkpoints, generated visualizations, logs, and exploratory notebooks are intentionally excluded from version control.

## Repository Layout

```text
.
├── experiments/monet/        # Main training and evaluation entrypoint
├── src/base/                 # Shared engine/model abstractions
├── src/engines/              # MoNet training engine
├── src/models/               # MoNet-Phy model components
├── src/utils/                # Data loading, metrics, graph utilities, logging
└── data/                     # Dataset folders, not included in the release
```

## Environment

Python 3.9+ is recommended.

```bash
conda create -n monet-phy python=3.9
conda activate monet-phy
pip install -r requirements.txt
```

`torch-scatter` and `torch-geometric` are CUDA/PyTorch-version sensitive. If the generic install fails, install the matching wheels from the official PyG instructions, then rerun `pip install -r requirements.txt`.

## Data

Put each dataset under `data/<DATASET_NAME>/`. The training code expects:

```text
data/<DATASET_NAME>/
├── his.npz
├── idx_train.npy
├── idx_val.npy
├── idx_test.npy
└── adjacency file
```

`his.npz` should contain `data`, `mean`, and `std`. The release entrypoint currently supports these dataset keys and input dimensions:

| Dataset | `input_dim` |
| --- | ---: |
| `PEMS-BAY` | 3 |
| `SD` | 3 |
| `KnowAir` | 15 |
| `BJAir` | 18 |

`input_dim` is selected automatically from the dataset name because the model assumes fixed positions for target value, temporal features, and covariates.

Adjacency filenames are configured in `src/utils/dataloader.py`. For example, `PEMS-BAY` uses `data/PEMS-BAY/adj_mx_bay.pkl`.

## Run

From the repository root:

```bash
python experiments/monet/main.py --dataset PEMS-BAY --device cuda:0 --mode train
```

For CPU smoke tests:

```bash
python experiments/monet/main.py --dataset PEMS-BAY --device cpu --bs 2 --max_epochs 1 --patience 1
```

To evaluate with the same entrypoint:

```bash
python experiments/monet/main.py --dataset PEMS-BAY --device cuda:0 --mode test
```

Model and training defaults live in `experiments/monet/config.yaml`. Command-line arguments override the YAML values.

## Notes

- Experiment logs are written to `eval/<model>/<dataset>/`.
- Checkpoints, logs, generated figures, and raw data are ignored by `.gitignore`.
- Optional hyperparameter tuning code uses `optuna` and `swanlab`; install them separately if needed.
- Add a project license before publishing the repository publicly.
