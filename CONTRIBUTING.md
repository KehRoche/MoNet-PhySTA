# Contributing

Contributions should keep this repository focused on the clean paper-reproduction code path.

## Before Opening a Change

Run the lightweight release checks:

```bash
python scripts/validate_release.py
```

If you have the datasets locally, also run:

```bash
python scripts/check_data.py --datasets PEMS-BAY SD KnowAir BJAir
python scripts/validate_release.py --check_data
```

Do not run long training jobs as part of routine validation. For experiment changes, run the relevant reproduction command locally and include the command, seed, commit, hardware, and summarized metrics in the pull request or issue.

## Repository Boundaries

Do not commit:

- datasets or generated arrays such as `.npz`, `.npy`, `.pkl`, `.h5`, `.mat`, `.nc`
- checkpoints, logs, result folders, or screenshots
- exploratory notebooks
- one-off visualization, ablation, or local launcher scripts

The release checker rejects these artifacts when they are tracked by git.

## Code Expectations

- Keep changes scoped to the paper reproduction path unless the issue explicitly asks for broader refactoring.
- Preserve dataset-specific input dimensions: `PEMS-BAY=3`, `SD=3`, `KnowAir=15`, `BJAir=18`.
- Keep the air-quality covariate branch in `src/models/monet.py`.
- Prefer logger output for training code; avoid unconditional debug prints in model components.
- Update `README.md` or `REPRODUCIBILITY.md` whenever commands, data layout, or expected results change.

## Reporting Reproduction Results

Use:

```bash
python scripts/summarize_results.py --no_write
```

Small metric differences from the paper are acceptable. Report the observed values and deltas rather than treating exact equality as a pass/fail condition.
