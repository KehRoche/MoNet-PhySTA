# Release Checklist

Use this checklist before publishing the repository or making final reproduction claims.

## Automated Gates

These checks are lightweight and should pass before every release:

```bash
python scripts/validate_release.py
```

On a machine with the datasets installed, also run:

```bash
python scripts/check_data.py --datasets PEMS-BAY SD KnowAir BJAir
python scripts/validate_release.py --check_data
```

The release validator checks:

- required project files and documentation
- Python syntax for git-tracked Python files without writing `.pyc` files
- smoke imports for the main experiment entrypoint, core model, and utility scripts
- local Markdown links, including troubleshooting and dataset guides
- reproduction-runner dry-run
- result summarization script
- accidental tracked datasets, checkpoints, logs, notebooks, visualization outputs, and exploratory scripts

The validator intentionally reasons about git-tracked files for release hygiene. Local ignored notebooks, datasets, logs, and exploratory scripts may exist in a working directory, but they must not be tracked by git.

## Manual Experiment Gates

Long experiments are intentionally manual. Before reporting final paper-comparison numbers:

1. Run the curated reproduction jobs from `REPRODUCIBILITY.md`.
2. Summarize the logs:

   ```bash
   python scripts/summarize_results.py --no_write
   ```

3. For final Table 1 reporting, verify provenance:

   ```bash
   python scripts/validate_release.py --strict_results
   ```

The strict result gate verifies that the latest completed paper-dataset runs used the current best-validation reload path. It does not enforce exact metric equality with the paper.

## Current Known Release State

At the time this checklist was added:

- `scripts/validate_release.py` passes.
- `scripts/validate_release.py --check_data` passes on the local machine with all four datasets.
- `scripts/validate_release.py --strict_results` passes for the latest completed paper-dataset runs.
- Current parsed local results are documented in `REPRODUCIBILITY.md`.
- PEMS-BAY has been rerun with the current best-validation reload path.
- Manuscript title and authors are recorded in `CITATION.md` and `CITATION.cff`; replace the submission citation with the proceedings entry once public.

## Final Human Checks

- Confirm the intended public license.
- Confirm the final venue/proceedings metadata and update the submission citation when it becomes public.
- Confirm that no local-only data access instructions or private paths are required for reproduction.
- Confirm that result deltas are reported transparently rather than hidden behind exact-match claims.
