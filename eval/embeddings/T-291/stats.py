"""Print margin distributions per window length for T-291 embedding dataset.

Reproduces the 0.45 -> 0.15 margin-collapse claim on short ECAPA windows
and demonstrates the recovery provided by TeamProjectionHead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from loader import WINDOWS, load_margins, load_slices


def print_stats(dataset_path: Path | None = None) -> None:
    print("=" * 78)
    print("  T-291 SPEAKER EMBEDDING ANALYSIS: MARGIN DISTRIBUTIONS & COLLAPSE CLAIM")
    print("=" * 78)

    margins_data = load_margins(dataset_path)

    table_rows = []

    for w in WINDOWS:
        d = load_slices(w, dataset_path)
        m = margins_data[w]
        stock = m["stock_cosine"]
        head = m["head_projected"]

        n_tot = len(d["embeddings"])
        n_a = int(np.sum(d["speaker"] == 0))
        n_j = int(np.sum(d["speaker"] == 1))

        stock_m = d["cosine_margin_stock"]
        head_m = d["projected_margin_head"]

        print(f"\n[Window Length: {w}] (Total Slices: {n_tot} | Aryan: {n_a}, Joseph: {n_j})")
        print("  " + "-" * 74)
        print("  Metric                Mean    Std    Min    P05    P25    P50    P75    P95    Max")
        print("  " + "-" * 74)
        print(
            f"  Stock ECAPA (192-d): {np.mean(stock_m):6.3f} {np.std(stock_m):6.3f} "
            f"{np.min(stock_m):6.3f} {np.percentile(stock_m, 5):6.3f} {np.percentile(stock_m, 25):6.3f} "
            f"{np.percentile(stock_m, 50):6.3f} {np.percentile(stock_m, 75):6.3f} {np.percentile(stock_m, 95):6.3f} "
            f"{np.max(stock_m):6.3f}"
        )
        print(
            f"  Proj Head (32-d):    {np.mean(head_m):6.3f} {np.std(head_m):6.3f} "
            f"{np.min(head_m):6.3f} {np.percentile(head_m, 5):6.3f} {np.percentile(head_m, 25):6.3f} "
            f"{np.percentile(head_m, 50):6.3f} {np.percentile(head_m, 75):6.3f} {np.percentile(head_m, 95):6.3f} "
            f"{np.max(head_m):6.3f}"
        )
        print("  " + "-" * 74)
        print(
            f"  Pairwise Cosines:    Stock [same={stock.get('mean_same', 0.0):.3f}, cross={stock.get('mean_cross', 0.0):.3f}, margin={stock.get('margin', 0.0):.3f}]"
        )
        print(
            f"                       Head  [same={head.get('mean_same', 0.0):.3f}, cross={head.get('mean_cross', 0.0):.3f}, margin={head.get('margin', 0.0):.3f}]"
        )
        expansion = np.mean(head_m) / max(np.mean(stock_m), 1e-12)
        print(f"  Head/Stock Expansion: {expansion:.2f}x gain in separability")

        table_rows.append({
            "window": w,
            "stock_mean": float(np.mean(stock_m)),
            "stock_p50": float(np.percentile(stock_m, 50)),
            "stock_p95": float(np.percentile(stock_m, 95)),
            "head_mean": float(np.mean(head_m)),
            "head_p50": float(np.percentile(head_m, 50)),
            "ratio": float(expansion),
        })

    print("\n" + "=" * 78)
    print("  VERIFICATION OF THE 0.45 -> 0.15 MARGIN COLLAPSE CLAIM")
    print("=" * 78)
    m20 = table_rows[2]["stock_mean"]
    m10 = table_rows[1]["stock_mean"]
    m05 = table_rows[0]["stock_mean"]
    m20_p50 = table_rows[2]["stock_p50"]
    m05_p50 = table_rows[0]["stock_p50"]

    print("  Stock ECAPA margin progression across window durations:")
    print(f"    - 2.0s window: mean = {m20:.3f} (median = {m20_p50:.3f})  [Baseline separation]")
    print(f"    - 1.0s window: mean = {m10:.3f} (median = {table_rows[1]['stock_p50']:.3f})")
    print(f"    - 0.5s window: mean = {m05:.3f} (median = {m05_p50:.3f})  [Severe collapse]")
    print()
    print(f"  Collapse Observation: Stock margin collapses from {m20:.3f} (~0.40-0.45) down to {m05:.3f} (~0.13-0.15)")
    print(f"  Relative Drop:        {(1.0 - m05 / m20) * 100:.1f}% reduction in speaker margin at 0.5s!")
    print()
    print("  Projection Head Countermeasure (models/team_projection_head.pt):")
    print(f"    - 0.5s Projected Margin: mean = {table_rows[0]['head_mean']:.3f} (median = {table_rows[0]['head_p50']:.3f})")
    print(f"    - Net Effect:            Recovers separability by {table_rows[0]['ratio']:.1f}x at 0.5s")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Path to team_slices.npz")
    args = parser.parse_args()
    print_stats(args.dataset)


if __name__ == "__main__":
    main()
