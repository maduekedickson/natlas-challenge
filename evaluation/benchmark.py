"""Generate the base-vs-adapted comparison (comparison.csv + comparison.png)
from evaluation/run_benchmark.py's output files.

No numbers are hand-typed: everything here is read directly from the JSON
results produced by actually running the model.

Usage:
    python evaluation/benchmark.py \
        --base evaluation/results/base_results.json \
        --adapted evaluation/results/adapted_results.json \
        --output-dir evaluation/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRIC_KEYS = [
    "parse_success_rate",
    "exact_match_rate",
    "mean_precision",
    "mean_recall",
    "mean_f1",
]


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_comparison(base: dict, adapted: dict | None) -> pd.DataFrame:
    rows = [{"model": "base", **base["metrics"]}]
    if adapted is not None:
        rows.append({"model": "adapted", **adapted["metrics"]})
    df = pd.DataFrame(rows).set_index("model")
    return df[METRIC_KEYS]


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    ax = df.T.plot(kind="bar", figsize=(9, 5), rot=30)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("N-ATLaS: Base vs. LoRA-Adapted — Health Report Extraction Benchmark")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapted", default=None, help="Optional: omit to report base-only")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    base = load_result(args.base)
    adapted = load_result(args.adapted) if args.adapted else None

    df = build_comparison(base, adapted)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path)
    print(f"Wrote {csv_path}")
    print(df)

    if adapted is not None:
        png_path = out_dir / "comparison.png"
        plot_comparison(df, png_path)
        print(f"Wrote {png_path}")

        delta = df.loc["adapted"] - df.loc["base"]
        print("\nImprovement (adapted - base):")
        print(delta)
    else:
        print("\nOnly base results provided — skipping chart until adapted results exist.")


if __name__ == "__main__":
    main()
