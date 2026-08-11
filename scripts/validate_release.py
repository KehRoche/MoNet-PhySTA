import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "README.md",
    "REPRODUCIBILITY.md",
    "RELEASE.md",
    "CITATION.md",
    "CONTRIBUTING.md",
    "DATASETS.md",
    "TROUBLESHOOTING.md",
    "LICENSE",
    "requirements.txt",
    "run.sh",
    "experiments/monet/config.yaml",
    "experiments/monet/main.py",
    "experiments/monet/run_best_experiments.py",
    "scripts/check_data.py",
    "scripts/summarize_results.py",
    "scripts/validate_release.py",
    "src/models/monet.py",
]

EXPECTED_BASE_CONFIG = {
    "model_name": "monet",
    "emd_dim": 32,
    "gfno_hidden": 32,
    "energy_splits": [0.7, 0.95],
    "topk_edges": 3,
    "ecc_layers": 1,
}

TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}
PRIVATE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", re.IGNORECASE),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
]

FORBIDDEN_TRACKED_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".html",
    ".ipynb",
    ".log",
    ".mat",
    ".nc",
    ".npy",
    ".npz",
    ".pkl",
    ".png",
    ".pdf",
    ".pt",
    ".pth",
    ".zip",
}

FORBIDDEN_TRACKED_DIR_PARTS = {
    "eval",
    "exp_vis",
    "iclr2026",
    "screenshots",
    "tgssp_trace",
    "tgssp_viz",
}

FORBIDDEN_TRACKED_NAMES = {
    "abl.py",
    "alpha_vis.py",
    "capture.py",
    "gata_vis.py",
    "run_exp.py",
    "vis_tgssp.py",
}

ALLOWED_DATA_SCRIPT_NAMES = {
    "generate_adj_mx.py",
    "generate_data_for_training.py",
    "generate_training_data.py",
}

MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

SMOKE_IMPORTS = [
    "experiments.monet.main",
    "src.models.monet",
    "src.utils.dataloader",
    "scripts.check_data",
    "scripts.summarize_results",
]


def run(command):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def git_tracked_files():
    result = run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "ls-files"])
    if result.returncode != 0:
        raise RuntimeError("Unable to list git-tracked files:\n" + result.stdout)
    return [Path(line) for line in result.stdout.splitlines()]


def check_required_files(errors):
    for rel_path in REQUIRED_FILES:
        if not (REPO_ROOT / rel_path).exists():
            errors.append(f"Missing required file: {rel_path}")


def check_base_config(errors):
    config_path = REPO_ROOT / "experiments/monet/config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["config1"]
    except Exception as exc:
        errors.append(f"Unable to parse base configuration: {exc}")
        return

    for key, expected in EXPECTED_BASE_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            errors.append(
                f"Base configuration drift for {key}: expected {expected!r}, got {actual!r}"
            )


def check_citation_metadata(errors):
    citation_path = REPO_ROOT / "CITATION.cff"
    try:
        citation = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Unable to parse CITATION.cff: {exc}")
        return

    required_values = {
        "cff-version": "1.2.0",
        "title": "PhySTA / MoNet-Phy",
        "type": "software",
        "license": "MIT",
    }
    for key, expected in required_values.items():
        actual = citation.get(key)
        if str(actual) != expected:
            errors.append(
                f"Citation metadata drift for {key}: expected {expected!r}, got {actual!r}"
            )
    if len(citation.get("authors", [])) != 5:
        errors.append("CITATION.cff must list all five manuscript authors.")


def check_tracked_files(errors):
    try:
        tracked_files = git_tracked_files()
    except RuntimeError as exc:
        errors.append(str(exc))
        return

    for path in tracked_files:
        line = path.as_posix()
        parts = set(path.parts)
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            errors.append(f"Large/generated artifact is tracked: {line}")
        if parts & FORBIDDEN_TRACKED_DIR_PARTS:
            errors.append(f"Generated output directory is tracked: {line}")
        if path.name in FORBIDDEN_TRACKED_NAMES:
            errors.append(f"Exploratory local script is tracked: {line}")
        if path.parts and path.parts[0] == "data" and path.suffix == ".py" and path.name not in ALLOWED_DATA_SCRIPT_NAMES:
            errors.append(f"Unexpected tracked data-side script: {line}")


def check_private_paths(errors):
    try:
        tracked_files = git_tracked_files()
    except RuntimeError as exc:
        errors.append(str(exc))
        return

    for path in tracked_files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = (REPO_ROOT / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRIVATE_PATH_PATTERNS:
            match = pattern.search(content)
            if match:
                errors.append(
                    f"Private absolute path is tracked in {path.as_posix()}: {match.group(0)}"
                )
                break


def check_python_compile(errors):
    try:
        files = [path for path in git_tracked_files() if path.suffix == ".py"]
    except RuntimeError as exc:
        errors.append(str(exc))
        return
    for path in files:
        try:
            source = (REPO_ROOT / path).read_text(encoding="utf-8")
            compile(source, path.as_posix(), "exec")
        except Exception as exc:
            errors.append(f"Python syntax check failed for {path.as_posix()}:\n{exc}")


def check_smoke_imports(errors):
    code = "; ".join(f"import {module}" for module in SMOKE_IMPORTS)
    result = run([sys.executable, "-c", code])
    if result.returncode != 0:
        errors.append("Smoke imports failed:\n" + result.stdout)


def check_markdown_links(errors):
    for markdown_file in REPO_ROOT.glob("*.md"):
        text = markdown_file.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not (markdown_file.parent / target).exists():
                rel_file = markdown_file.relative_to(REPO_ROOT)
                errors.append(f"Broken markdown link in {rel_file}: {match.group(1)}")


def check_dry_run(errors):
    result = run(
        [
            sys.executable,
            "experiments/monet/run_best_experiments.py",
            "--dry_run",
            "--datasets",
            "PEMS-BAY",
        ]
    )
    if result.returncode != 0:
        errors.append("Reproduction runner dry-run failed:\n" + result.stdout)


def check_result_summary(errors, strict_results=False):
    command = [sys.executable, "scripts/summarize_results.py", "--no_write"]
    if strict_results:
        command.append("--strict_paper")
    result = run(command)
    if result.returncode != 0:
        errors.append("Result summary script failed:\n" + result.stdout)


def check_local_data(errors):
    result = run([sys.executable, "scripts/check_data.py"])
    if result.returncode != 0:
        errors.append("Local dataset check failed:\n" + result.stdout)


def check_license(warnings):
    if not any((REPO_ROOT / name).exists() for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"]):
        warnings.append("No LICENSE file found. Add one before publishing publicly.")


def main():
    parser = argparse.ArgumentParser(description="Run lightweight release-readiness checks.")
    parser.add_argument("--skip_dry_run", action="store_true", help="Skip reproduction runner dry-run.")
    parser.add_argument(
        "--strict_results",
        action="store_true",
        help="Require latest paper-dataset runs to include best-validation reload logs; does not enforce metric equality.",
    )
    parser.add_argument(
        "--check_data",
        action="store_true",
        help="Also validate local dataset files. This is off by default because datasets are not tracked in git.",
    )
    args = parser.parse_args()

    errors = []
    warnings = []
    check_required_files(errors)
    check_base_config(errors)
    check_citation_metadata(errors)
    check_tracked_files(errors)
    check_private_paths(errors)
    check_python_compile(errors)
    check_smoke_imports(errors)
    check_markdown_links(errors)
    if not args.skip_dry_run:
        check_dry_run(errors)
    check_result_summary(errors, strict_results=args.strict_results)
    if args.check_data:
        check_local_data(errors)
    check_license(warnings)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Release validation passed.")


if __name__ == "__main__":
    main()
