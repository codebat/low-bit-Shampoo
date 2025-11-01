#!/usr/bin/env python3
"""
Plot wall-clock accuracy curves from metrics logged by train_cifar.py.

Example:
    python plot_metrics.py \
        --title "ResNet34 on CIFAR-100" \
        --output plots/resnet34_cifar100.png \
        --runs "SGDM=logs/sgdm_metrics.csv" \
                "SGDM+32-bit Shampoo=logs/shampoo32_metrics.csv" \
                "SGDM+4-bit Shampoo=logs/shampoo4_metrics.csv"
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs. wall-clock curves from metrics CSV files.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of label=metrics.csv entries. Each metrics file should be produced by train_cifar.py",
    )
    parser.add_argument("--output", default="metrics_plot.png", type=str, help="Output image path")
    parser.add_argument("--title", default=None, type=str, help="Figure title")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Which split to plot")
    parser.add_argument("--smooth", default=1, type=int, help="Moving-average window (>=1) applied to accuracy")
    parser.add_argument("--x-field", default="wall_clock_min", type=str, help="CSV field for the x-axis")
    parser.add_argument("--y-field", default="top1", type=str, help="CSV field for the y-axis")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, help="Optional y-axis limits")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="Optional x-axis limits")
    parser.add_argument("--legend-loc", default="best", type=str, help="matplotlib legend location")
    return parser.parse_args()


EXPECTED_FIELDS = [
    "run",
    "epoch",
    "split",
    "top1",
    "top5",
    "loss",
    "wall_clock_min",
    "lr",
    "optimizer",
    "prec_bits",
    "device",
    "model",
    "batch_size",
    "amp",
]


def _load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames and all(field in reader.fieldnames for field in ("split", "wall_clock_min", "top1")):
            return rows

    # Fall back to headerless files (older logs) by re-reading as plain rows.
    with path.open("r", newline="") as handle:
        reader_plain = csv.reader(handle)
        rows_plain = [row for row in reader_plain if row]

    parsed_rows: List[Dict[str, str]] = []
    for row in rows_plain:
        if len(row) != len(EXPECTED_FIELDS):
            raise ValueError(
                f"Metrics file {path} has {len(row)} columns but expected {len(EXPECTED_FIELDS)}. "
                "Delete it and re-run training to regenerate with headers."
            )
        parsed_rows.append(dict(zip(EXPECTED_FIELDS, row)))
    return parsed_rows


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if window <= 1 or n == 0:
        return values
    w = min(window, n)

    c = np.cumsum(np.insert(values, 0, 0.0))
    tail = (c[w:] - c[:-w]) / w                 # length n - w + 1
    head = (c[1:w] - c[0]) / np.arange(1, w)    # lengths 1..w-1
    return np.concatenate([head, tail])



def _prepare_series(
    rows: Iterable[Dict[str, str]],
    split: str,
    x_field: str,
    y_field: str,
    smooth_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        if row.get("split") != split:
            continue
        try:
            x_val = float(row[x_field])
            y_val = float(row[y_field])
        except (KeyError, ValueError):
            continue
        xs.append(x_val)
        ys.append(y_val)

    if not xs:
        raise ValueError(f"No rows found for split='{split}' with fields '{x_field}' and '{y_field}'.")

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

    order = np.argsort(xs_arr)
    xs_arr = xs_arr[order]
    ys_arr = ys_arr[order]

    ys_smoothed = _moving_average(ys_arr, smooth_window)
    return xs_arr, ys_smoothed


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for run_entry in args.runs:
        if "=" not in run_entry:
            raise ValueError(f"Run definition '{run_entry}' is missing '='. Expected 'label=path/to/metrics.csv'.")
        label, path_str = run_entry.split("=", 1)
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Metrics file does not exist: {path}")

        rows = _load_metrics(path)
        xs, ys = _prepare_series(rows, args.split, args.x_field, args.y_field, args.smooth)

        ax.plot(xs, ys, label=label, linewidth=2.0)

    ax.set_xlabel("Wall-clock Time (min)" if args.x_field == "wall_clock_min" else args.x_field)
    ax.set_ylabel("Top-1 Accuracy (%)" if args.y_field == "top1" else args.y_field)
    if args.title:
        ax.set_title(args.title)
    if args.ylim:
        ax.set_ylim(*args.ylim)
    if args.xlim:
        ax.set_xlim(*args.xlim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc=args.legend_loc)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
