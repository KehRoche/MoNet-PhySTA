import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPT = REPO_ROOT / "experiments" / "monet" / "main.py"
RESULT_DIR = REPO_ROOT / "eval" / "monet" / "best_runs"

METRIC_RE = re.compile(
    r"Average Test MAE:\s*([0-9.]+),\s*Test RMSE:\s*([0-9.]+),\s*Test MAPE:\s*([0-9.]+)"
)

BEST_CONFIGS = {
    "PEMS-BAY": {
        "input_dim": 3,
        "emd_dim": 32,
        "gfno_hidden": 32,
        "energy_splits": [0.7, 0.95],
        "topk_edges": 3,
        "ecc_layers": 1,
        "reference_mae": 1.6523,
    },
    "SD": {
        "input_dim": 3,
        "emd_dim": 32,
        "gfno_hidden": 32,
        "energy_splits": [0.7, 0.95],
        "topk_edges": 3,
        "ecc_layers": 1,
        "reference_mae": 20.5886,
    },
    "KnowAir": {
        "input_dim": 15,
        "emd_dim": 16,
        "gfno_hidden": 8,
        "energy_splits": [0.7, 0.95],
        "topk_edges": 3,
        "ecc_layers": 1,
        "reference_mae": 20.4873,
    },
    "BJAir": {
        "input_dim": 18,
        "emd_dim": 16,
        "gfno_hidden": 8,
        "energy_splits": [0.7, 0.95],
        "topk_edges": 3,
        "ecc_layers": 1,
        "reference_mae": 55.5130,
    },
}


def build_command(args, dataset):
    config = BEST_CONFIGS[dataset]
    energy_low, energy_high = config["energy_splits"]
    command = [
        args.python,
        "-u",
        str(MAIN_SCRIPT),
        "--dataset",
        dataset,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--bs",
        str(args.batch_size),
        "--seq_len",
        str(args.seq_len),
        "--horizon",
        str(args.horizon),
        "--input_dim",
        str(config["input_dim"]),
        "--output_dim",
        "1",
        "--mode",
        "train",
        "--max_epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--lrate",
        str(args.learning_rate),
        "--wdecay",
        str(args.weight_decay),
        "--cl_epoch",
        str(args.cl_epoch),
        "--warm_epoch",
        str(args.warm_epoch),
        "--mask_ratio",
        str(args.mask_ratio),
        "--emd_dim",
        str(config["emd_dim"]),
        "--gfno_hidden",
        str(config["gfno_hidden"]),
        "--energy_splits",
        str(energy_low),
        str(energy_high),
        "--topk_edges",
        str(config["topk_edges"]),
        "--ecc_layers",
        str(config["ecc_layers"]),
    ]
    return command


def run_one(args, dataset):
    command = build_command(args, dataset)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n===== Running {dataset} =====", flush=True)
    print(" ".join(command), flush=True)

    if args.dry_run:
        return {
            "dataset": dataset,
            "status": "dry_run",
            "command": command,
            "started_at": started_at,
            **BEST_CONFIGS[dataset],
        }

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    last_metric = None
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
        match = METRIC_RE.search(line)
        if match:
            mae, rmse, mape = map(float, match.groups())
            last_metric = {"mae": mae, "rmse": rmse, "mape": mape}

    return_code = process.wait()
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "dataset": dataset,
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command,
        **BEST_CONFIGS[dataset],
    }
    if last_metric is not None:
        result.update(last_metric)
        result["mae_gap_to_reference"] = last_metric["mae"] - BEST_CONFIGS[dataset]["reference_mae"]
    else:
        result["error"] = "No Average Test metric found in process output."

    output_path = RESULT_DIR / f"{dataset}_stdout.log"
    output_path.write_text("".join(output_lines), encoding="utf-8")
    return result


def write_results(results):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "results.json"
    csv_path = RESULT_DIR / "results.csv"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "dataset",
        "status",
        "mae",
        "rmse",
        "mape",
        "reference_mae",
        "mae_gap_to_reference",
        "emd_dim",
        "gfno_hidden",
        "energy_splits",
        "topk_edges",
        "ecc_layers",
        "input_dim",
        "seed",
        "return_code",
        "started_at",
        "finished_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved summary to {csv_path}")
    print(f"Saved details to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MoNet best-known configurations on all datasets.")
    parser.add_argument("--datasets", nargs="+", choices=BEST_CONFIGS.keys(), default=list(BEST_CONFIGS.keys()))
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch each run.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--cl_epoch", type=int, default=3)
    parser.add_argument("--warm_epoch", type=int, default=30)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and write configs without training.")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset in args.datasets:
        result = run_one(args, dataset)
        result["seed"] = args.seed
        results.append(result)
        write_results(results)
        if result.get("status") == "failed" and not args.continue_on_error:
            raise SystemExit(f"{dataset} failed. Re-run with --continue_on_error to keep going.")


if __name__ == "__main__":
    main()
