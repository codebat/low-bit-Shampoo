#!/usr/bin/env python3

"""Utility to visualize Shampoo preconditioner eigenvalue spectra."""

import argparse
import csv
import glob
import math
from pathlib import Path


def load_spectrum(path: Path):
    kept, discarded = [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            label, value = row[0], float(row[1])
            if label == "kept":
                kept.append(value)
            elif label == "discarded":
                discarded.append(value)
    return kept, discarded


def lemma_ratio(kept, discarded, s=-0.25):
    if not kept and not discarded:
        return float("nan")
    exponent = 2 * s
    eps = 1e-12
    all_vals = kept + discarded
    denom = sum(max(abs(v), eps) ** exponent for v in all_vals)
    if denom == 0:
        return float("nan")
    numer = sum(max(abs(v), eps) ** exponent for v in discarded)
    return math.sqrt(numer / denom)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="logs/eigenvalues_*.csv", help="Glob pattern for eigenvalue CSV files.")
    parser.add_argument("--output", default="plots/eigenvalues", help="Directory to store generated plots.")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins for kept/discarded spectra.")
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
        kept, discarded = load_spectrum(path)
        ratio = lemma_ratio(kept, discarded)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist([kept, discarded], bins=args.bins, label=["kept", "discarded"], alpha=0.7)
        ax.set_title(f"{path.stem} | ratio={ratio:.4f}")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()

        outfile = output_dir / f"{path.stem}.png"
        fig.savefig(outfile)
        plt.close(fig)


if __name__ == "__main__":
    main()
