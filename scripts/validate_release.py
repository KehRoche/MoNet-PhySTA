import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "experiments/monet/main.py",
    "experiments/monet/run_best_experiments.py",
    "src/models/monet.py",
]

FORBIDDEN_TRACKED_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".html",
    ".log",
    ".mat",
    ".nc",
    ".npy",
    ".npz",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".zip",
}

FORBIDDEN_TRACKED_DIR_PARTS = {
    "eval",
    "exp_vis",
    "screenshots",
    "tgssp_trace",
    "tgssp_viz",
}


def run(command):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def check_required_files(errors):
    for rel_path in REQUIRED_FILES:
        if not (REPO_ROOT / rel_path).exists():
            errors.append(f"Missing required file: {rel_path}")


def check_tracked_files(errors):
    result = run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "ls-files"])
    if result.returncode != 0:
        errors.append("Unable to list git-tracked files:\n" + result.stdout)
        return

    for line in result.stdout.splitlines():
        path = Path(line)
        parts = set(path.parts)
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            errors.append(f"Large/generated artifact is tracked: {line}")
        if parts & FORBIDDEN_TRACKED_DIR_PARTS:
            errors.append(f"Generated output directory is tracked: {line}")


def check_python_compile(errors):
    files = [
        str(path.relative_to(REPO_ROOT))
        for base in ["src", "experiments"]
        for path in (REPO_ROOT / base).rglob("*.py")
    ]
    result = run([sys.executable, "-m", "py_compile", *files])
    if result.returncode != 0:
        errors.append("Python compilation failed:\n" + result.stdout)


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


def check_license(warnings):
    if not any((REPO_ROOT / name).exists() for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"]):
        warnings.append("No LICENSE file found. Add one before publishing publicly.")


def main():
    parser = argparse.ArgumentParser(description="Run lightweight release-readiness checks.")
    parser.add_argument("--skip_dry_run", action="store_true", help="Skip reproduction runner dry-run.")
    args = parser.parse_args()

    errors = []
    warnings = []
    check_required_files(errors)
    check_tracked_files(errors)
    check_python_compile(errors)
    if not args.skip_dry_run:
        check_dry_run(errors)
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
