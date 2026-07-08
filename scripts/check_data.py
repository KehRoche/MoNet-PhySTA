import argparse
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "PEMS-BAY": {
        "data_dir": "data/PEMS-BAY",
        "adj_path": "data/PEMS-BAY/adj_mx_bay.pkl",
        "input_dim": 3,
        "nodes": 325,
    },
    "SD": {
        "data_dir": "data/sd",
        "adj_path": "data/sd/sd_rn_adj.npy",
        "input_dim": 3,
        "nodes": 716,
    },
    "KnowAir": {
        "data_dir": "data/KnowAir",
        "adj_path": "data/KnowAir/adj_matrix.npy",
        "input_dim": 15,
        "nodes": 184,
    },
    "BJAir": {
        "data_dir": "data/BJAir",
        "adj_path": "data/BJAir/BJAir.npy",
        "input_dim": 18,
        "nodes": 35,
    },
}

INDEX_FILES = {
    "train": "idx_train.npy",
    "val": "idx_val.npy",
    "test": "idx_test.npy",
}


def resolve_path(data_root, rel_path):
    path = Path(rel_path)
    if path.parts and path.parts[0].lower() == "data":
        path = Path(*path.parts[1:])
    return data_root / path


def load_pickle(path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with path.open("rb") as f:
            return pickle.load(f, encoding="latin1")


def adjacency_array(path):
    if path.suffix == ".pkl":
        value = load_pickle(path)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            value = value[2]
    else:
        value = np.load(path, allow_pickle=True)
    return np.asarray(value).squeeze()


def check_dataset(name, spec, data_root, seq_len, horizon, strict_indices):
    errors = []
    warnings = []
    data_dir = resolve_path(data_root, spec["data_dir"])
    adj_path = resolve_path(data_root, spec["adj_path"])
    his_path = data_dir / "his.npz"

    required_paths = [his_path, adj_path] + [data_dir / filename for filename in INDEX_FILES.values()]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        return {
            "dataset": name,
            "status": "FAIL",
            "shape": "-",
            "index": "-",
            "adj": "-",
            "errors": [f"Missing file: {path}" for path in missing],
            "warnings": [],
        }

    try:
        with np.load(his_path) as ptr:
            keys = set(ptr.files)
            for key in ["data", "mean", "std"]:
                if key not in keys:
                    errors.append(f"{his_path} missing key '{key}'")
            if "data" not in keys:
                raise ValueError("his.npz has no data array")
            data = ptr["data"]
            shape = tuple(data.shape)
    except Exception as exc:
        return {
            "dataset": name,
            "status": "FAIL",
            "shape": "-",
            "index": "-",
            "adj": "-",
            "errors": [f"Unable to load {his_path}: {exc}"],
            "warnings": [],
        }

    if len(shape) < 3:
        errors.append(f"data shape must be at least 3D, got {shape}")
    else:
        if shape[1] != spec["nodes"]:
            errors.append(f"node count mismatch: expected {spec['nodes']}, got {shape[1]}")
        if shape[-1] < spec["input_dim"]:
            errors.append(f"feature dim mismatch: expected at least {spec['input_dim']}, got {shape[-1]}")

    index_summary = []
    for split, filename in INDEX_FILES.items():
        path = data_dir / filename
        try:
            idx = np.asarray(np.load(path)).reshape(-1)
        except Exception as exc:
            errors.append(f"Unable to load {path}: {exc}")
            continue
        if idx.size == 0:
            errors.append(f"{path} is empty")
            continue
        min_idx = int(idx.min())
        max_idx = int(idx.max())
        valid_mask = (idx - seq_len + 1 >= 0) & (idx + horizon < shape[0])
        invalid_count = int(idx.size - np.count_nonzero(valid_mask))
        if invalid_count:
            message = (
                f"{split} has {invalid_count} boundary indices outside seq_len={seq_len}, "
                f"horizon={horizon}; DataLoader filters them at runtime"
            )
            if strict_indices:
                errors.append(message)
            else:
                warnings.append(message)
        index_summary.append(f"{split}:{idx.size}")

    try:
        adj = adjacency_array(adj_path)
        adj_shape = tuple(adj.shape)
    except Exception as exc:
        adj_shape = "-"
        errors.append(f"Unable to load adjacency {adj_path}: {exc}")
    else:
        expected_adj = (spec["nodes"], spec["nodes"])
        if adj_shape != expected_adj:
            errors.append(f"adjacency shape mismatch: expected {expected_adj}, got {adj_shape}")

    return {
        "dataset": name,
        "status": "FAIL" if errors else "OK",
        "shape": str(shape),
        "index": ", ".join(index_summary) if index_summary else "-",
        "adj": str(adj_shape),
        "errors": errors,
        "warnings": warnings,
    }


def print_table(rows):
    headers = ["Dataset", "Status", "Data shape", "Index counts", "Adj shape"]
    table = [[row["dataset"], row["status"], row["shape"], row["index"], row["adj"]] for row in rows]
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *table)
    ]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description="Check local dataset files before running MoNet experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=sorted(DATASETS),
        help="Datasets to check.",
    )
    parser.add_argument("--seq_len", type=int, default=12, help="Input sequence length used by experiments.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon used by experiments.")
    parser.add_argument(
        "--strict_indices",
        action="store_true",
        help="Fail if split indices include boundary values that the runtime DataLoader would filter.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root directory containing dataset folders.",
    )
    args = parser.parse_args()

    rows = [
        check_dataset(
            name,
            DATASETS[name],
            args.data_root.resolve(),
            args.seq_len,
            args.horizon,
            args.strict_indices,
        )
        for name in args.datasets
    ]
    print_table(rows)

    errors = []
    warnings = []
    for row in rows:
        for warning in row["warnings"]:
            warnings.append(f"{row['dataset']}: {warning}")
        for error in row["errors"]:
            errors.append(f"{row['dataset']}: {error}")

    if warnings:
        print()
        for warning in warnings:
            print(f"WARNING: {warning}")

    if errors:
        print()
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print()
    print("Dataset check passed.")


if __name__ == "__main__":
    main()
