import argparse
import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "eval" / "monet"

PAPER_MASK0 = {
    "KnowAir": {"mae": 20.55, "mape": 0.55, "rmse": 33.05},
    "PEMS-BAY": {"mae": 1.66, "mape": 0.04, "rmse": 3.61},
    "SD": {"mae": 20.64, "mape": 0.15, "rmse": 33.05},
}

LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (.*)$")
PARAM_RE = re.compile(r"Param:\s*(\{.*\})")
EPOCH_RE = re.compile(r"Epoch:\s*(\d+).*?Valid Loss:\s*([0-9.]+)")
EARLY_RE = re.compile(r"Early stop at epoch\s+(\d+),\s+loss\s+=\s+([0-9.]+)")
LOADED_RE = re.compile(r"Loaded best validation model,\s+loss\s+=\s+([0-9.]+)")
METRIC_RE = re.compile(
    r"Average Test MAE:\s*([0-9.]+),\s*Test RMSE:\s*([0-9.]+),\s*Test MAPE:\s*([0-9.]+)"
)


def parse_param(raw):
    try:
        return json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        return {"raw": raw}


def parse_runs(log_root):
    runs = []
    for path in sorted(log_root.rglob("record_s*.log")):
        current = None
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line_match = LINE_RE.match(raw_line)
            timestamp = line_match.group(1) if line_match else ""
            message = line_match.group(2) if line_match else raw_line

            param_match = PARAM_RE.search(message)
            if param_match:
                if current is not None:
                    runs.append(current)
                params = parse_param(param_match.group(1))
                current = {
                    "dataset": params.get("dataset", path.parent.name),
                    "start_time": timestamp,
                    "log_file": str(path),
                    "params": params,
                    "epochs": 0,
                    "best_val": None,
                    "early_stop_epoch": None,
                    "loaded_best_val": None,
                    "metrics": [],
                }
                continue

            if current is None:
                continue

            epoch_match = EPOCH_RE.search(message)
            if epoch_match:
                current["epochs"] += 1
                valid_loss = float(epoch_match.group(2))
                current["best_val"] = (
                    valid_loss if current["best_val"] is None else min(current["best_val"], valid_loss)
                )

            early_match = EARLY_RE.search(message)
            if early_match:
                current["early_stop_epoch"] = int(early_match.group(1))

            loaded_match = LOADED_RE.search(message)
            if loaded_match:
                current["loaded_best_val"] = float(loaded_match.group(1))

            metric_match = METRIC_RE.search(message)
            if metric_match:
                mae, rmse, mape = map(float, metric_match.groups())
                current["metrics"].append(
                    {
                        "timestamp": timestamp,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                    }
                )

        if current is not None:
            runs.append(current)

    return [run for run in runs if run["metrics"]]


def latest_by_dataset(runs):
    latest = {}
    for run in runs:
        dataset = run["dataset"]
        if dataset not in latest or run["start_time"] > latest[dataset]["start_time"]:
            latest[dataset] = run
    return latest


def flatten_run(run):
    metric = run["metrics"][-1]
    paper = PAPER_MASK0.get(run["dataset"], {})
    row = {
        "dataset": run["dataset"],
        "start_time": run["start_time"],
        "epochs": run["epochs"],
        "best_val": run["best_val"],
        "early_stop_epoch": run["early_stop_epoch"],
        "loaded_best_val": run["loaded_best_val"],
        "mae": metric["mae"],
        "rmse": metric["rmse"],
        "mape": metric["mape"],
        "paper_mae": paper.get("mae"),
        "paper_rmse": paper.get("rmse"),
        "paper_mape": paper.get("mape"),
        "mae_gap_to_paper": None,
        "rmse_gap_to_paper": None,
        "mape_gap_to_paper": None,
        "params": json.dumps(run["params"], ensure_ascii=False, sort_keys=True),
        "log_file": run["log_file"],
    }
    if paper:
        row["mae_gap_to_paper"] = metric["mae"] - paper["mae"]
        row["rmse_gap_to_paper"] = metric["rmse"] - paper["rmse"]
        row["mape_gap_to_paper"] = metric["mape"] - paper["mape"]
    return row


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_markdown(rows):
    headers = [
        "Dataset",
        "MAE",
        "Paper MAE",
        "Delta MAE",
        "MAPE",
        "Paper MAPE",
        "RMSE",
        "Paper RMSE",
        "Epochs",
        "Best Val",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["dataset"],
            format_value(row["mae"]),
            format_value(row["paper_mae"]),
            format_value(row["mae_gap_to_paper"]),
            format_value(row["mape"]),
            format_value(row["paper_mape"]),
            format_value(row["rmse"]),
            format_value(row["paper_rmse"]),
            format_value(row["epochs"]),
            format_value(row["best_val"]),
        ]
        print("| " + " | ".join(values) + " |")


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    headers = ["Dataset", "MAE", "Paper MAE", "Delta MAE", "MAPE", "RMSE", "Epochs"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["dataset"],
            format_value(row["mae"]),
            format_value(row["paper_mae"]),
            format_value(row["mae_gap_to_paper"]),
            format_value(row["mape"]),
            format_value(row["rmse"]),
            format_value(row["epochs"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Summarize MoNet experiment logs and compare with paper Table 1.")
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--all_runs", action="store_true", help="Report every completed run instead of latest per dataset.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_LOG_ROOT / "best_runs")
    parser.add_argument("--no_write", action="store_true", help="Only print the summary table.")
    args = parser.parse_args()

    runs = parse_runs(args.log_root)
    if args.all_runs:
        selected = runs
    else:
        selected = list(latest_by_dataset(runs).values())

    rows = [flatten_run(run) for run in sorted(selected, key=lambda item: item["dataset"])]
    if not rows:
        print(f"No completed runs found under {args.log_root}.")
        return

    print_markdown(rows)
    if not args.no_write:
        paths = write_outputs(rows, args.output_dir)
        print("\nWrote summary files:")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
