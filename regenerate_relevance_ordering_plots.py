#!/usr/bin/env python3
"""
Regenerate relevance-ordering plots (PNG + SVG) from saved summary CSVs.

The relevance-ordering runs save, in every results folder, a
``relevance_ordering_summary<sfx>.csv`` that contains the mean (and, for the
similarity curves, the std) of each ordering at every revealed-pixel fraction.
This script walks a directory tree, finds every such summary CSV, and re-plots
the figures *without re-running the model* — so you can add SVG versions (or
restore PNGs) after the fact.

What can / cannot be reproduced from the summary CSV
----------------------------------------------------
The summary CSV columns are::

    fraction,
    mean_prp_sim, mean_proto_sim, mean_rand_sim,
    std_prp_sim,  std_proto_sim,  std_rand_sim,
    mean_prp_pred, mean_proto_pred, mean_rand_pred

So this script can faithfully regenerate, in PNG and SVG:
  - relevance_ordering_similarity<sfx>            (±std shaded)   -> has mean + std
  - relevance_ordering_similarity<sfx>_mean_only                 -> has mean
  - relevance_ordering_prediction<sfx>_mean_only                 -> has mean

It does NOT regenerate the *prediction* ±std shaded plot, because the
per-prototype prediction std is not stored in the summary CSV (nor in the
per-image CSV, which only keeps similarity fraction columns).

Usage
-----
    cd /path/to/CleverHansRegression
    conda activate new_insight_env

    python3 regenerate_relevance_ordering_plots.py \
        --root bld_art_25_Apr/class3_v14_fix_43/exp1/DR_25_Jan_2026_1/Fold0_DR_03_May_3/prp/relevance_ordering_class3_1

    # Several roots at once:
    python3 regenerate_relevance_ordering_plots.py --root DIR_A DIR_B

    # Only PNG, or only SVG:
    python3 regenerate_relevance_ordering_plots.py --root DIR --formats svg
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from relevance_ordering_general import plot_mean_curves  # noqa: E402

_SUMMARY_RE = re.compile(r"^relevance_ordering_summary(?P<sfx>.*)\.csv$")
_PROTO_RE = re.compile(r"prototype_(\d+)")


def _proto_label_from_path(csv_path: Path) -> Optional[int]:
    """Look for a 'prototype_<n>' component in the path and return <n>."""
    for part in csv_path.parts:
        m = _PROTO_RE.fullmatch(part)
        if m:
            return int(m.group(1))
    return None


def regenerate_one(csv_path: Path, formats: List[str]) -> List[str]:
    """Regenerate plots next to one summary CSV. Returns list of written stems."""
    m = _SUMMARY_RE.match(csv_path.name)
    sfx = m.group("sfx") if m else ""
    out_dir = csv_path.parent
    proto_label = _proto_label_from_path(csv_path)

    df = pd.read_csv(csv_path)
    fractions = df["fraction"].to_numpy()

    written: List[str] = []

    def _emit(target_png: Path):
        # plot_mean_curves always writes PNG + SVG; unwanted formats are pruned
        # afterwards by _prune_unwanted_formats().
        written.append(str(target_png.with_suffix("")))

    # --- Similarity: ±std shaded ---
    sim_png = out_dir / f"relevance_ordering_similarity{sfx}.png"
    plot_mean_curves(
        fractions,
        df["mean_prp_sim"].to_numpy(),
        df["mean_proto_sim"].to_numpy(),
        df["mean_rand_sim"].to_numpy(),
        df["std_prp_sim"].to_numpy(),
        df["std_proto_sim"].to_numpy(),
        df["std_rand_sim"].to_numpy(),
        ylabel="Prototype Similarity (higher = better)",
        title="Relevance Ordering: Prototype Similarity",
        savepath=str(sim_png),
        prototype_plot_label=proto_label,
    )
    _emit(sim_png)

    # --- Similarity: mean only ---
    sim_mean_png = out_dir / f"relevance_ordering_similarity{sfx}_mean_only.png"
    plot_mean_curves(
        fractions,
        df["mean_prp_sim"].to_numpy(),
        df["mean_proto_sim"].to_numpy(),
        df["mean_rand_sim"].to_numpy(),
        None, None, None,
        ylabel="Prototype Similarity (higher = better)",
        title="Relevance Ordering: Prototype Similarity",
        savepath=str(sim_mean_png),
        prototype_plot_label=proto_label,
    )
    _emit(sim_mean_png)

    # --- Prediction: mean only (std for prediction is not stored) ---
    pred_mean_png = out_dir / f"relevance_ordering_prediction{sfx}_mean_only.png"
    plot_mean_curves(
        fractions,
        df["mean_prp_pred"].to_numpy(),
        df["mean_proto_pred"].to_numpy(),
        df["mean_rand_pred"].to_numpy(),
        None, None, None,
        ylabel="Regression Prediction (continuous)",
        title="Relevance Ordering: Regression Output",
        savepath=str(pred_mean_png),
        prototype_plot_label=proto_label,
    )
    _emit(pred_mean_png)

    return written


def _prune_unwanted_formats(out_dir: Path, formats: List[str]) -> None:
    """If user asked for a single format, drop the sibling produced by plot_mean_curves."""
    if set(formats) == {"png", "svg"}:
        return
    unwanted = {"png", "svg"} - set(formats)
    for ext in unwanted:
        for f in out_dir.glob(f"relevance_ordering_*.{ext}"):
            f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", nargs="+", required=True, help="Directory(ies) to scan recursively.")
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "svg"],
        help="Formats to keep (default: png svg).",
    )
    args = ap.parse_args()

    formats = [f.lower() for f in args.formats]
    total_csvs = 0
    touched_dirs = set()

    for root in args.root:
        root_path = Path(root)
        if not root_path.exists():
            print(f"WARNING: root does not exist, skipping: {root_path}", file=sys.stderr)
            continue
        summaries = sorted(root_path.rglob("relevance_ordering_summary*.csv"))
        if not summaries:
            print(f"No summary CSVs found under {root_path}")
            continue
        for csv_path in summaries:
            try:
                regenerate_one(csv_path, formats)
                total_csvs += 1
                touched_dirs.add(csv_path.parent)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR regenerating from {csv_path}: {e!r}", file=sys.stderr)

    for d in touched_dirs:
        _prune_unwanted_formats(d, formats)

    print(
        f"\nDone. Regenerated plots from {total_csvs} summary CSV(s) "
        f"across {len(touched_dirs)} folder(s). Formats: {', '.join(formats)}."
    )
    print(
        "Note: the prediction ±std (shaded) plot is NOT regenerated — prediction "
        "std is not stored in the summary CSV. Similarity (shaded + mean_only) and "
        "prediction mean_only are reproduced faithfully."
    )


if __name__ == "__main__":
    main()
