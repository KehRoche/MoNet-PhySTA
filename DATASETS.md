# Dataset Guide

Datasets are not distributed with this repository. Place local copies under `data/` before running experiments.

## Required Layout

| Dataset | Directory | Nodes | Required features | Adjacency file |
|---|---|---:|---:|---|
| `PEMS-BAY` | `data/PEMS-BAY` | 325 | 3 | `adj_mx_bay.pkl` |
| `SD` | `data/sd` | 716 | 3 | `sd_rn_adj.npy` |
| `KnowAir` | `data/KnowAir` | 184 | 15 | `adj_matrix.npy` |
| `BJAir` | `data/BJAir` | 35 | 18 | `BJAir.npy` |

Each dataset directory must contain:

```text
his.npz
idx_train.npy
idx_val.npy
idx_test.npy
<adjacency file>
```

`his.npz` must contain:

- `data`: array shaped `[time, nodes, features]`
- `mean`: scaler mean or minimum, depending on the dataset preprocessing
- `std`: scaler standard deviation or maximum, depending on the dataset preprocessing

The model uses only the first `input_dim` channels for each dataset. These dimensions are fixed in `experiments/monet/main.py` and are intentionally enforced at runtime.

## Validation

Run this before launching long experiments:

```bash
python scripts/check_data.py --datasets PEMS-BAY SD KnowAir BJAir
```

The checker validates:

- required files
- `his.npz` keys
- node count
- minimum feature dimension
- train/validation/test index files
- adjacency matrix shape

For KnowAir and BJAir, the current local index files include a few boundary indices that are outside the `seq_len=12`, `horizon=12` sampling window. This is reported as a warning by default because the runtime DataLoader filters those samples before batching.

Use strict mode only when auditing regenerated indices:

```bash
python scripts/check_data.py --datasets KnowAir BJAir --strict_indices
```

## Git Policy

Do not commit dataset files, generated arrays, checkpoints, logs, or notebooks. The `.gitignore` and `scripts/validate_release.py` checks are configured to reject these artifacts when they are tracked by git.

Allowed tracked files under `data/` are limited to preprocessing scripts such as `generate_data_for_training.py`, `generate_training_data.py`, and `generate_adj_mx.py`.
