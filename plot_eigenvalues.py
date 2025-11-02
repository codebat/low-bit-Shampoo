#!/usr/bin/env python3

"""Utility to visualize Shampoo preconditioner eigenvalue spectra."""

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import List, Tuple


def load_eigenvalues(path: Path) -> List[float]:
    values: List[float] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return values
        header_lower = [h.lower() for h in header]
        if header_lower == ["category", "value"]:
            kept: List[float] = []
            discarded: List[float] = []
            for row in reader:
                if not row:
                    continue
                label, value = row[0], float(row[1])
                if label == "kept":
                    kept.append(value)
                elif label == "discarded":
                    discarded.append(value)
            values = sorted(kept + discarded)
        elif header_lower == ["rank", "value"]:
            for row in reader:
                if not row:
                    continue
                _, value = row
                values.append(float(value))
        else:
            raise ValueError(f"Unrecognized header in {path}: {header}")
    return sorted(values)


def truncation_curve(eigenvalues: List[float], s: float = -0.25) -> Tuple[List[float], List[float]]:
    n = len(eigenvalues)
    if n == 0:
        return [0.0], [float("nan")]
    eps = 1e-12
    exponent = 2 * s
    weights = [max(abs(v), eps) ** exponent for v in eigenvalues]
    denom = sum(weights)
    if denom == 0:
        return [0.0], [float("nan")]
    tail = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        tail[i] = tail[i + 1] + weights[i]

    fractions: List[float] = []
    ratios: List[float] = []
    for k in range(n + 1):
        complement = tail[k]
        ratio = math.sqrt(complement / denom)
        fractions.append(k / n)
        ratios.append(ratio)
    return fractions, ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="logs/eigenvalues_*.csv", help="Glob pattern for eigenvalue CSV files.")
    parser.add_argument("--output", default="plots/eigenvalues", help="Directory to store generated plots.")
    parser.add_argument("--s", type=float, default=-0.25, help="Exponent s used in the lemma (default -0.25).")
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern {args.pattern}.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting; install it or run within an environment that provides it.") from exc

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        eigenvalues = load_eigenvalues(path)
        fractions, ratios = truncation_curve(eigenvalues, s=args.s)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fractions, ratios, marker="o", markersize=3)
        ax.set_title(f"{path.stem}")
        ax.set_xlabel("Fraction of eigenvalues kept")
        ax.set_ylabel("Lemma bound")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        outfile = output_dir / f"{path.stem}.png"
        fig.savefig(outfile)
        plt.close(fig)


if __name__ == "__main__":
    main()
