"""
plrp_ext/relevance_ordering_prp_vs_plrp.py
==========================================

Relevance Ordering Test (Insertion variant) — Baseline PRP vs PLRP-PRP.

Adapts the methodology of relevance_ordering_proto.py (and the underlying
relevance_ordering_paper.py) by replacing the *prototype-heatmap* baseline
with **PLRP-PRP**. The random ordering is kept as a control.

What this measures
------------------
**Faithfulness**, not localization. If PLRP-PRP's top-ranked pixels are
truly the most important for prototype similarity, similarity should rise
faster when we insert PLRP's top pixels first than when we insert PRP's
top pixels first.

Protocol (same insertion mechanic as the original test)
-------------------------------------------------------
  1. Start from a uniform-noise baseline image (same shape as the real image).
  2. Gradually REPLACE pixels in the baseline with real-image pixels, in the
     order defined by the explanation map (most important first).
  3. After each fraction, run the (uncanonized) model forward and record:
       - prototype similarity s[pno]
       - regression prediction
  4. Compare the three curves: Baseline-PRP / PLRP-PRP / Random.

Sparsity caveat (read before reporting numbers)
-----------------------------------------------
PLRP-PRP zeroes out a large fraction of pixels by design. When sorting
pixels by |relevance|, those zeros pile up at the bottom of the ranking
and their internal order is arbitrary (raster scan). Therefore:

  * The primary, **sparsity-aware** signal is the *partial AUC over the first
    10 % of pixels* (``partial_auc10_*``). This is where PLRP's pruning
    can actually pay off because both methods have meaningful order there.
  * The full-range AUC is still computed and saved, but is partially driven
    by the arbitrary tie-breaking inside PLRP's zero-bucket. Both curves
    necessarily converge at fraction = 1 (the entire image is revealed).

Both metrics are written to ``relevance_ordering_per_image*.csv`` and
summarized in the on-disk + console outputs.

Isolation guarantees (same as the rest of plrp_ext/)
----------------------------------------------------
  * Does NOT modify any existing file outside plrp_ext/.
  * Reuses the method-agnostic helpers ``load_ppnet``, ``load_image``,
    ``run_insertion_curve``, ``importance_order`` from
    ``relevance_ordering_paper`` so the insertion mechanic is identical to
    your existing tests.
  * Uses the unmodified ``insight_prp.generate_prp_image`` for baseline PRP.
  * Uses ``plrp_ext.insight_prp_plrp.generate_prp_image`` for PLRP-PRP.
  * ``set_plrp_params(...)`` only mutates state in
    ``plrp_ext.lrp_general6_plrp``; the baseline pipeline does not import
    that module and is therefore unaffected.
  * Loads THREE independent PPNet instances (plain, baseline canonized,
    PLRP canonized) so in-place canonization cannot corrupt either path.

Usage (artifact-stratified — same flag style as your existing run)
------------------------------------------------------------------

    cd /sc/home/akshay.gudi/code/CleverHansRegression
    conda activate new_insight_env

    python3 plrp_ext/relevance_ordering_prp_vs_plrp.py \\
        --artifact_labels_csv /path/to/test_labeled_data.csv \\
        --image_dir /path/to/test/jpeg \\
        --output_dir /path/to/results_dir \\
        --model_path /path/to/Epoch_50_after_protopushing.pth \\
        --param_jsonpath config/params_example_ordinal.json \\
        --prototypes 22 27 33 \\
        --num_random_images 50 \\
        --num_fractions 21 \\
        --plrp_p_pos 0.25 --plrp_p_neg 0.125 \\
        --seed 42

    # Built-in sanity: p_pos = p_neg = 0 must give PLRP curve == PRP curve.
    python3 plrp_ext/relevance_ordering_prp_vs_plrp.py \\
        --artifact_labels_csv ... --image_dir ... --output_dir .../sanity \\
        --model_path ... --param_jsonpath config/params_example_ordinal.json \\
        --prototypes 22 --num_random_images 5 \\
        --plrp_p_pos 0.0 --plrp_p_neg 0.0
"""

# ---------------------------------------------------------------------------
# Make top-level modules importable when running as
# ``python3 plrp_ext/relevance_ordering_prp_vs_plrp.py``.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from typing import List

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers import load_json
from define_parameters import NetworkParams

# Method-agnostic helpers — same as the original relevance ordering test.
from relevance_ordering_paper import (
    load_ppnet,
    load_image,
    run_insertion_curve,
    importance_order,
)
from relevance_ordering_artifact_split import load_image_paths_from_artifact_csv

# Baseline PRP pipeline (unmodified).
from insight_prp import (
    PRPCanonizedModel as build_baseline_canon,
    generate_prp_image as generate_prp_baseline,
)

# PLRP-PRP pipeline (isolated under plrp_ext/).
from plrp_ext.lrp_general6_plrp import set_plrp_params, get_plrp_params
from plrp_ext.insight_prp_plrp import (
    PRPCanonizedModel as build_plrp_canon,
    generate_prp_image as generate_prp_plrp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _partial_auc(values, fractions, max_fraction):
    """Trapezoidal AUC over the prefix [0, max_fraction] of an insertion curve."""
    arr = np.asarray(values, dtype=np.float64)
    f = np.asarray(fractions, dtype=np.float64)
    mask = f <= max_fraction + 1e-12
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(arr[mask], f[mask]))


def _sample_random_paths(paths, n, rng):
    """Subsample up to ``n`` paths deterministically (sorted by chosen index)."""
    paths = list(paths)
    if len(paths) <= n:
        return paths
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image_prp_vs_plrp(
    ppnet_plain,
    ppnet_baseline_canon,
    ppnet_plrp_canon,
    image_tensor,
    pno,
    fractions,
    device,
    seed=0,
):
    """For one image + one prototype, run the insertion test with three orderings.

    Orderings:
      1) Baseline PRP   -> insight_prp.generate_prp_image
      2) PLRP-PRP       -> plrp_ext.insight_prp_plrp.generate_prp_image
      3) Random         -> control
    """
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    n_pixels = H * W

    prp_hm = generate_prp_baseline(image_tensor, pno, ppnet_baseline_canon, device)
    plrp_hm = generate_prp_plrp(image_tensor, pno, ppnet_plrp_canon, device)

    order_prp = importance_order(prp_hm)
    order_plrp = importance_order(plrp_hm)
    rng = np.random.default_rng(seed + 999)
    order_rand = rng.permutation(n_pixels)

    sim_prp, pred_prp = run_insertion_curve(
        ppnet_plain, image_tensor, pno, order_prp, fractions, device, seed=seed,
    )
    sim_plrp, pred_plrp = run_insertion_curve(
        ppnet_plain, image_tensor, pno, order_plrp, fractions, device, seed=seed,
    )
    sim_rand, pred_rand = run_insertion_curve(
        ppnet_plain, image_tensor, pno, order_rand, fractions, device, seed=seed,
    )

    return {
        "prp_sim": sim_prp,
        "plrp_sim": sim_plrp,
        "rand_sim": sim_rand,
        "prp_pred": pred_prp,
        "plrp_pred": pred_plrp,
        "rand_pred": pred_rand,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mean_curves(
    fractions,
    mean_prp,
    mean_plrp,
    mean_rand,
    std_prp=None,
    std_plrp=None,
    std_rand=None,
    ylabel="Prototype Similarity",
    title="Relevance Ordering Test: PRP vs PLRP-PRP",
    savepath="relevance_ordering.png",
    prototype_plot_label=None,
):
    """Three-curve insertion plot for PRP, PLRP-PRP, and Random orderings.

    A dotted vertical line at 10 % marks the partial-AUC boundary.
    """
    if prototype_plot_label is not None:
        title = f"Prototype {prototype_plot_label} - {title}"
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [f * 100 for f in fractions]

    ax.plot(pcts, mean_prp,  "o-",  color="#1f77b4",
            label="Baseline PRP ordering", linewidth=2, markersize=4)
    ax.plot(pcts, mean_plrp, "s-",  color="#d62728",
            label="PLRP-PRP ordering",     linewidth=2, markersize=4)
    ax.plot(pcts, mean_rand, "^--", color="#7f7f7f",
            label="Random ordering",       linewidth=1.5, markersize=4)

    if std_prp is not None:
        ax.fill_between(pcts, mean_prp - std_prp, mean_prp + std_prp, color="#1f77b4", alpha=0.15)
    if std_plrp is not None:
        ax.fill_between(pcts, mean_plrp - std_plrp, mean_plrp + std_plrp, color="#d62728", alpha=0.15)
    if std_rand is not None:
        ax.fill_between(pcts, mean_rand - std_rand, mean_rand + std_rand, color="#7f7f7f", alpha=0.10)

    ax.axvline(x=10.0, color="black", linestyle=":", alpha=0.4)
    y_top = ax.get_ylim()[1]
    y_bot = ax.get_ylim()[0]
    ax.text(10.5, y_bot + 0.05 * max(y_top - y_bot, 1e-9),
            " 10% partial-AUC", fontsize=8, color="black")

    ax.set_xlabel("Pixels revealed (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {savepath}")


# ---------------------------------------------------------------------------
# Core run loop (per prototype, per image-group)
# ---------------------------------------------------------------------------

def run_ro_prp_vs_plrp_core(
    ppnet_plain,
    ppnet_baseline_canon,
    ppnet_plrp_canon,
    image_paths,
    out_dir,
    fractions,
    img_size,
    device,
    *,
    seed,
    fixed_prototype_index,
    prototype_plot_label,
    output_filename_suffix="",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sfx = output_filename_suffix or ""
    detail_csv = out_dir / f"relevance_ordering_per_image{sfx}.csv"

    fraction_cols_prp = [f"prp_sim_{f:.2f}" for f in fractions]
    fraction_cols_plrp = [f"plrp_sim_{f:.2f}" for f in fractions]
    fraction_cols_rand = [f"rand_sim_{f:.2f}" for f in fractions]

    all_columns = (
        ["image", "prototype",
         "auc_prp", "auc_plrp", "auc_rand",
         "auc_delta_plrp_minus_prp", "plrp_better_full",
         "partial_auc10_prp", "partial_auc10_plrp", "partial_auc10_rand",
         "partial_auc10_delta_plrp_minus_prp", "plrp_better_partial10"]
        + fraction_cols_prp + fraction_cols_plrp + fraction_cols_rand
    )

    if not detail_csv.exists():
        pd.DataFrame(columns=all_columns).to_csv(detail_csv, index=False)
        print(f"Created per-image CSV: {detail_csv}")
    else:
        print(f"Appending to existing per-image CSV: {detail_csv}")

    all_prp_sim, all_plrp_sim, all_rand_sim = [], [], []
    all_prp_pred, all_plrp_pred, all_rand_pred = [], [], []

    n_total = len(image_paths)
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[{img_idx+1}/{n_total}] {img_path.name}")
        image_tensor = load_image(str(img_path), img_size=img_size)

        print(f"  Prototype {fixed_prototype_index}: computing PRP + PLRP heatmaps + insertion curves ...")
        result = process_image_prp_vs_plrp(
            ppnet_plain,
            ppnet_baseline_canon,
            ppnet_plrp_canon,
            image_tensor,
            fixed_prototype_index,
            fractions,
            device,
            seed=seed + img_idx,
        )

        all_prp_sim.append(result["prp_sim"])
        all_plrp_sim.append(result["plrp_sim"])
        all_rand_sim.append(result["rand_sim"])
        all_prp_pred.append(result["prp_pred"])
        all_plrp_pred.append(result["plrp_pred"])
        all_rand_pred.append(result["rand_pred"])

        auc_prp  = float(np.trapz(result["prp_sim"],  fractions))
        auc_plrp = float(np.trapz(result["plrp_sim"], fractions))
        auc_rand = float(np.trapz(result["rand_sim"], fractions))
        delta_auc_full = auc_plrp - auc_prp

        p10_prp  = _partial_auc(result["prp_sim"],  fractions, 0.10)
        p10_plrp = _partial_auc(result["plrp_sim"], fractions, 0.10)
        p10_rand = _partial_auc(result["rand_sim"], fractions, 0.10)
        delta_p10 = p10_plrp - p10_prp

        row = {
            "image": img_path.name,
            "prototype": fixed_prototype_index,
            "auc_prp": round(auc_prp, 6),
            "auc_plrp": round(auc_plrp, 6),
            "auc_rand": round(auc_rand, 6),
            "auc_delta_plrp_minus_prp": round(delta_auc_full, 6),
            "plrp_better_full": delta_auc_full > 0,
            "partial_auc10_prp":  round(p10_prp, 6),
            "partial_auc10_plrp": round(p10_plrp, 6),
            "partial_auc10_rand": round(p10_rand, 6),
            "partial_auc10_delta_plrp_minus_prp": round(delta_p10, 6),
            "plrp_better_partial10": delta_p10 > 0,
            **{f"prp_sim_{f:.2f}":  s for f, s in zip(fractions, result["prp_sim"])},
            **{f"plrp_sim_{f:.2f}": s for f, s in zip(fractions, result["plrp_sim"])},
            **{f"rand_sim_{f:.2f}": s for f, s in zip(fractions, result["rand_sim"])},
        }
        pd.DataFrame([row]).to_csv(detail_csv, mode="a", header=False, index=False)
        print(f"  Saved row.")

    prp_sim_arr   = np.array(all_prp_sim)
    plrp_sim_arr  = np.array(all_plrp_sim)
    rand_sim_arr  = np.array(all_rand_sim)
    prp_pred_arr  = np.array(all_prp_pred)
    plrp_pred_arr = np.array(all_plrp_pred)
    rand_pred_arr = np.array(all_rand_pred)

    mean_prp_sim  = prp_sim_arr.mean(axis=0)
    mean_plrp_sim = plrp_sim_arr.mean(axis=0)
    mean_rand_sim = rand_sim_arr.mean(axis=0)
    std_prp_sim   = prp_sim_arr.std(axis=0)
    std_plrp_sim  = plrp_sim_arr.std(axis=0)
    std_rand_sim  = rand_sim_arr.std(axis=0)

    mean_prp_pred  = prp_pred_arr.mean(axis=0)
    mean_plrp_pred = plrp_pred_arr.mean(axis=0)
    mean_rand_pred = rand_pred_arr.mean(axis=0)
    std_prp_pred   = prp_pred_arr.std(axis=0)
    std_plrp_pred  = plrp_pred_arr.std(axis=0)
    std_rand_pred  = rand_pred_arr.std(axis=0)

    detail_df = pd.read_csv(detail_csv)
    n_full = int(detail_df["plrp_better_full"].sum())
    n_p10  = int(detail_df["plrp_better_partial10"].sum())
    nn = len(detail_df)
    print(f"\nFull-range AUC: PLRP > PRP in {n_full}/{nn} pairs "
          f"({100 * n_full / max(nn, 1):.1f}%).")
    print(f"Partial AUC @10%: PLRP > PRP in {n_p10}/{nn} pairs "
          f"({100 * n_p10 / max(nn, 1):.1f}%).")
    print(f"Means - full AUC:        PRP={detail_df['auc_prp'].mean():.4f} | "
          f"PLRP={detail_df['auc_plrp'].mean():.4f} | "
          f"Random={detail_df['auc_rand'].mean():.4f}")
    print(f"Means - partial AUC@10%: PRP={detail_df['partial_auc10_prp'].mean():.4f} | "
          f"PLRP={detail_df['partial_auc10_plrp'].mean():.4f} | "
          f"Random={detail_df['partial_auc10_rand'].mean():.4f}")

    summary_df = pd.DataFrame({
        "fraction": fractions,
        "mean_prp_sim":  mean_prp_sim,
        "mean_plrp_sim": mean_plrp_sim,
        "mean_rand_sim": mean_rand_sim,
        "std_prp_sim":   std_prp_sim,
        "std_plrp_sim":  std_plrp_sim,
        "std_rand_sim":  std_rand_sim,
        "mean_prp_pred":  mean_prp_pred,
        "mean_plrp_pred": mean_plrp_pred,
        "mean_rand_pred": mean_rand_pred,
    })
    summary_csv = out_dir / f"relevance_ordering_summary{sfx}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    plot_mean_curves(
        fractions,
        mean_prp_sim, mean_plrp_sim, mean_rand_sim,
        std_prp_sim, std_plrp_sim, std_rand_sim,
        ylabel="Prototype Similarity (higher = better)",
        title="Relevance Ordering: Prototype Similarity (PRP vs PLRP-PRP)",
        savepath=str(out_dir / f"relevance_ordering_similarity{sfx}.png"),
        prototype_plot_label=prototype_plot_label,
    )
    plot_mean_curves(
        fractions,
        mean_prp_pred, mean_plrp_pred, mean_rand_pred,
        std_prp_pred, std_plrp_pred, std_rand_pred,
        ylabel="Regression Prediction (continuous)",
        title="Relevance Ordering: Regression Output (PRP vs PLRP-PRP)",
        savepath=str(out_dir / f"relevance_ordering_prediction{sfx}.png"),
        prototype_plot_label=prototype_plot_label,
    )

    sim_png = out_dir / f"relevance_ordering_similarity{sfx}.png"
    pred_png = out_dir / f"relevance_ordering_prediction{sfx}.png"
    plot_mean_curves(
        fractions,
        mean_prp_sim, mean_plrp_sim, mean_rand_sim,
        None, None, None,
        ylabel="Prototype Similarity (higher = better)",
        title="Relevance Ordering: Prototype Similarity (PRP vs PLRP-PRP)",
        savepath=str(sim_png.with_name(sim_png.stem + "_mean_only" + sim_png.suffix)),
        prototype_plot_label=prototype_plot_label,
    )
    plot_mean_curves(
        fractions,
        mean_prp_pred, mean_plrp_pred, mean_rand_pred,
        None, None, None,
        ylabel="Regression Prediction (continuous)",
        title="Relevance Ordering: Regression Output (PRP vs PLRP-PRP)",
        savepath=str(pred_png.with_name(pred_png.stem + "_mean_only" + pred_png.suffix)),
        prototype_plot_label=prototype_plot_label,
    )

    print(f"\nDone. Results in: {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Relevance Ordering Test (Insertion) - Baseline PRP vs PLRP-PRP.",
    )
    parser.add_argument("--artifact_labels_csv", required=True,
                        help="CSV with image_name + artifact_label (1=artifact, 0=clean).")
    parser.add_argument("--image_dir", required=True, help="Folder of .jpeg images.")
    parser.add_argument("--output_dir", required=True, help="Root output directory.")
    parser.add_argument("--model_path", required=True, help="Checkpoint .pth or .ckpt.")
    parser.add_argument("--param_jsonpath", required=True, help="Network parameters JSON.")

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--prototype_index", type=int, default=None,
                   help="Single prototype index (0-based).")
    g.add_argument("--prototypes", type=int, nargs="+", default=None,
                   help="Several prototype indices.")

    parser.add_argument("--num_random_images", type=int, default=50,
                        help="Max images per artifact group (default 50).")
    parser.add_argument("--num_fractions", type=int, default=21,
                        help="Insertion curve resolution (default 21).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")

    parser.add_argument("--plrp_p_pos", type=float, default=0.25,
                        help="PLRP-lambda: proportion of positive relevance to prune. Default 0.25.")
    parser.add_argument("--plrp_p_neg", type=float, default=0.125,
                        help="PLRP-lambda: proportion of negative relevance to prune. Default 0.125.")

    args = parser.parse_args()

    if not 0.0 <= args.plrp_p_pos < 1.0:
        parser.error("--plrp_p_pos must be in [0, 1).")
    if not 0.0 <= args.plrp_p_neg < 1.0:
        parser.error("--plrp_p_neg must be in [0, 1).")

    # PLRP state only mutates plrp_ext.lrp_general6_plrp; baseline pipeline
    # uses lrp_general6 (different module) and is unaffected.
    set_plrp_params(p_pos=args.plrp_p_pos, p_neg=args.plrp_p_neg)
    eff_p_pos, eff_p_neg = get_plrp_params()
    print(f"PLRP-lambda state: p_pos={eff_p_pos}, p_neg={eff_p_neg}")
    if eff_p_pos == 0.0 and eff_p_neg == 0.0:
        print("  -> PLRP DISABLED; PLRP curve should match baseline PRP curve.")
    else:
        print("  -> PLRP ENABLED; baseline canonization is unaffected by this state.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    img_size = network_params.img_size
    num_proto = int(network_params.proto_shape[0])
    print(f"Architecture: {network_params.base_architecture}, "
          f"img_size={img_size}, num_prototypes={num_proto}")

    plist: List[int] = (
        [args.prototype_index] if args.prototype_index is not None
        else list(dict.fromkeys(args.prototypes))
    )
    bad = [p for p in plist if p < 0 or p >= num_proto]
    if bad:
        raise SystemExit(f"Invalid prototype index(es) {bad}; valid 0..{num_proto - 1}.")

    # Load THREE independent PPNet instances.
    print("Loading three PPNet instances:")
    print("  1/3 -> ppnet_plain         (uncanonized; forward passes during insertion)")
    ppnet_plain = load_ppnet(args.model_path, network_params, device).to(device)

    print("  2/3 -> ppnet_baseline      (canonized for baseline PRP)")
    ppnet_baseline = load_ppnet(args.model_path, network_params, device)
    ppnet_baseline.base_architecture = network_params.base_architecture
    ppnet_baseline_canon = build_baseline_canon(ppnet_baseline).to(device)

    print("  3/3 -> ppnet_plrp          (canonized for PLRP-PRP)")
    ppnet_plrp = load_ppnet(args.model_path, network_params, device)
    ppnet_plrp.base_architecture = network_params.base_architecture
    ppnet_plrp_canon = build_plrp_canon(ppnet_plrp).to(device)
    print("Canonization complete.\n")

    with_art, without_art = load_image_paths_from_artifact_csv(
        args.artifact_labels_csv, args.image_dir,
    )
    rng = np.random.default_rng(args.seed)
    samp_with    = _sample_random_paths(with_art,    args.num_random_images, rng)
    samp_without = _sample_random_paths(without_art, args.num_random_images, rng)
    print(f"Sampled {len(samp_with)} with-artifact and "
          f"{len(samp_without)} without-artifact image(s) "
          f"(requested up to {args.num_random_images} each).")

    fractions = np.linspace(0.0, 1.0, args.num_fractions).tolist()
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    for pno in plist:
        proto_root = base_out / f"prototype_{pno}"
        print(f"\n========== Prototype {pno} -> {proto_root} ==========")

        if samp_with:
            sub = proto_root / "with_artifact"
            print(f"  Running with_artifact ({len(samp_with)} images) ...")
            run_ro_prp_vs_plrp_core(
                ppnet_plain,
                ppnet_baseline_canon,
                ppnet_plrp_canon,
                samp_with,
                sub,
                fractions,
                img_size,
                device,
                seed=args.seed,
                fixed_prototype_index=pno,
                prototype_plot_label=pno,
                output_filename_suffix="_with_artifact",
            )
        else:
            print("  [with_artifact] No images - skipped.")

        if samp_without:
            sub = proto_root / "without_artifact"
            print(f"  Running without_artifact ({len(samp_without)} images) ...")
            run_ro_prp_vs_plrp_core(
                ppnet_plain,
                ppnet_baseline_canon,
                ppnet_plrp_canon,
                samp_without,
                sub,
                fractions,
                img_size,
                device,
                seed=args.seed + 10_000,
                fixed_prototype_index=pno,
                prototype_plot_label=pno,
                output_filename_suffix="_without_artifact",
            )
        else:
            print("  [without_artifact] No images - skipped.")

    print(f"\nDone. Results under: {base_out}")


if __name__ == "__main__":
    main()
