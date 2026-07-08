# Reproducibility Guide

This document records the commands and expected results for reproducing the PhySTA/MoNet-Phy experiments in the paper.

The paper table used here is Table 1 from `ICLR26_PhySTA.pdf`. Current local runs use `mask_ratio=0.0`, so the direct comparison is against the `Mask=0` columns.

## Dataset Requirements

Expected files:

```text
data/PEMS-BAY/his.npz
data/PEMS-BAY/idx_train.npy
data/PEMS-BAY/idx_val.npy
data/PEMS-BAY/idx_test.npy
data/PEMS-BAY/adj_mx_bay.pkl

data/sd/his.npz
data/sd/idx_train.npy
data/sd/idx_val.npy
data/sd/idx_test.npy
data/sd/sd_rn_adj.npy

data/KnowAir/his.npz
data/KnowAir/idx_train.npy
data/KnowAir/idx_val.npy
data/KnowAir/idx_test.npy
data/KnowAir/adj_matrix.npy

data/BJAir/his.npz
data/BJAir/idx_train.npy
data/BJAir/idx_val.npy
data/BJAir/idx_test.npy
data/BJAir/BJAir.npy
```

The supported input dimensions are fixed:

```python
{
    "PEMS-BAY": 3,
    "SD": 3,
    "KnowAir": 15,
    "BJAir": 18,
}
```

The entrypoint overrides `--input_dim` from this mapping to avoid invalid covariate layouts.

## Recommended Reproduction Runner

Long-running experiments should be launched manually in your own terminal. Run all Table 1 datasets:

```bash
python experiments/monet/run_best_experiments.py --datasets PEMS-BAY SD KnowAir
```

Run the additional BJAir experiment:

```bash
python experiments/monet/run_best_experiments.py --datasets BJAir
```

The runner writes:

```text
eval/monet/best_runs/results.csv
eval/monet/best_runs/results.json
eval/monet/best_runs/<DATASET>_stdout.log
```

After runs finish, summarize the latest completed run for each dataset:

```bash
python scripts/summarize_results.py
```

To include every completed run found in `eval/monet`:

```bash
python scripts/summarize_results.py --all_runs
```

For final release verification, require the latest Table 1 runs to come from code that reloads the best validation state before test:

```bash
python scripts/summarize_results.py --strict_paper --no_write
```

This strict check verifies that the latest completed paper-dataset runs used the current best-validation reload path. It does not require exact numerical equality with the paper. Small metric differences are acceptable and expected across hardware, dependency versions, and stochastic training.

## Curated Hyperparameters

| Dataset | `emd_dim` | `gfno_hidden` | `energy_splits` | `topk_edges` | `ecc_layers` | `bs` | `seed` |
|---|---:|---:|---|---:|---:|---:|---:|
| PEMS-BAY | 32 | 32 | `[0.7, 0.95]` | 3 | 1 | 32 | 2023 |
| SD | 32 | 32 | `[0.7, 0.95]` | 3 | 1 | 32 | 2023 |
| KnowAir | 16 | 8 | `[0.7, 0.95]` | 3 | 1 | 32 | 2023 |
| BJAir | 16 | 8 | `[0.7, 0.95]` | 3 | 1 | 32 | 2023 |

Common training options:

```text
seq_len=12
horizon=12
max_epochs=80
patience=15
lrate=0.002
wdecay=0.00001
cl_epoch=3
warm_epoch=30
mask_ratio=0.0
```

## Current Verified Results

Current local results from `eval/monet`:

| Dataset | Current MAE | Current MAPE | Current RMSE | Paper Table 1 MAE | Paper Table 1 MAPE | Paper Table 1 RMSE | MAE Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| KnowAir | 20.3364 | 0.5358 | 32.8287 | 20.55 | 0.55 | 33.05 | -0.2136 |
| PEMS-BAY | 1.6848 | 0.0392 | 3.7207 | 1.66 | 0.04 | 3.61 | +0.0248 |
| SD | 21.2282 | 0.1592 | 32.6871 | 20.64 | 0.15 | 33.05 | +0.5882 |

BJAir is not included in Table 1 of the paper. Current local result:

| Dataset | Current MAE | Current MAPE | Current RMSE |
|---|---:|---:|---:|
| BJAir | 49.9713 | 1.7079 | 73.1515 |

## Interpretation

- KnowAir currently matches and slightly improves the paper Table 1 `Mask=0` result.
- SD has better RMSE than Table 1 but worse MAE/MAPE. The MAE gap is about 2.85%.
- PEMS-BAY was run before the best-validation-state reload fix and should be rerun with the current code before final release claims.
- BJAir is an additional experiment and should not be presented as a Table 1 reproduction unless the paper is updated to include it.
- The comparison table is intended as a reproducibility guide, not a hard pass/fail threshold. Report observed deltas honestly in release notes or experiment logs.

## Single-Dataset Commands

PEMS-BAY:

```bash
python experiments/monet/main.py --dataset PEMS-BAY --device cuda:0 --bs 32 --seq_len 12 --horizon 12 --mode train --max_epochs 80 --patience 15 --lrate 0.002 --wdecay 0.00001 --cl_epoch 3 --warm_epoch 30 --mask_ratio 0 --emd_dim 32 --gfno_hidden 32 --energy_splits 0.7 0.95 --topk_edges 3 --ecc_layers 1
```

SD:

```bash
python experiments/monet/main.py --dataset SD --device cuda:0 --bs 32 --seq_len 12 --horizon 12 --mode train --max_epochs 80 --patience 15 --lrate 0.002 --wdecay 0.00001 --cl_epoch 3 --warm_epoch 30 --mask_ratio 0 --emd_dim 32 --gfno_hidden 32 --energy_splits 0.7 0.95 --topk_edges 3 --ecc_layers 1
```

KnowAir:

```bash
python experiments/monet/main.py --dataset KnowAir --device cuda:0 --bs 32 --seq_len 12 --horizon 12 --mode train --max_epochs 80 --patience 15 --lrate 0.002 --wdecay 0.00001 --cl_epoch 3 --warm_epoch 30 --mask_ratio 0 --emd_dim 16 --gfno_hidden 8 --energy_splits 0.7 0.95 --topk_edges 3 --ecc_layers 1
```

BJAir:

```bash
python experiments/monet/main.py --dataset BJAir --device cuda:0 --bs 32 --seq_len 12 --horizon 12 --mode train --max_epochs 80 --patience 15 --lrate 0.002 --wdecay 0.00001 --cl_epoch 3 --warm_epoch 30 --mask_ratio 0 --emd_dim 16 --gfno_hidden 8 --energy_splits 0.7 0.95 --topk_edges 3 --ecc_layers 1
```

## Release Checklist

- Verify `python -m py_compile` passes for `src` and `experiments`.
- Run `python experiments/monet/run_best_experiments.py --dry_run`.
- Run `python scripts/summarize_results.py --no_write`.
- Run `python scripts/validate_release.py`.
- Manually rerun PEMS-BAY with the current best-validation reload logic before reporting final paper comparison.
- After manually rerunning PEMS-BAY, run `python scripts/validate_release.py --strict_results`. This verifies provenance, not exact metric equality.
- Confirm datasets are documented but not committed.
- Confirm the MIT license is appropriate for the public release, or replace it with the intended license.
