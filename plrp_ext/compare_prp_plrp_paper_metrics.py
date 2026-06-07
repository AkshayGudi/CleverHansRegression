#!/usr/bin/env python3
"""
Compare **Baseline PRP vs PLRP-PRP** using Sparsity and Perturbation Faithfulness metrics
These both are used in original paper

Neither metric requires a ground-truth artifact mask, so this script
complements the existing localization comparison and the relevance-ordering
insertion test rather than duplicating either.

Outputs (under ``--output_dir``)
--------------------------------
  - ``per_image_metrics.csv``       — one row per (image, prototype, method)
  - ``summary_per_prototype.csv``   — mean / std per prototype × method
  - ``paired_deltas_per_image.csv`` — per (image, prototype) Δ = PLRP − PRP
                                      for each metric
  - ``comparison_paired.json``      — overall means/std per method +
                                      paired summary (PLRP − PRP
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json  # noqa: E402

from prp.relevance_ordering_general import (  # noqa: E402
    load_ppnet,
    load_image,
    get_prototype_heatmap,
    get_similarity_and_prediction,
)

from prp.insight_prp import PRPCanonizedModel as PRPCanonizedModelBaseline  # noqa: E402
from prp.insight_prp import generate_prp_image as generate_prp_image_baseline  # noqa: E402

from plrp_ext.lrp_general6_plrp import set_plrp_params, get_plrp_params  # noqa: E402
from plrp_ext.insight_prp_plrp import PRPCanonizedModel as PRPCanonizedModelPlrp  # noqa: E402
from plrp_ext.insight_prp_plrp import generate_prp_image as generate_prp_image_plrp  # noqa: E402

from prp.localization_metrics import discover_evaluation_images  # noqa: E402


# Metric keys (all "higher is better").
METRIC_KEYS: List[str] = [
    "gini",
    "aopc_drop_at_5pct",
    "aopc_drop_at_10pct",
    "aopc_mean_drop",
    "aopc_partial_auc_10",
    "aopc_full_auc",
]


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def gini_coefficient(heatmap: np.ndarray) -> float:
    """Gini coefficient on |heatmap|, computed with the sorted formula.

    Definition:
        Gini = (2 * sum_{i=1..n}(i * x_i) - (n + 1) * sum(x)) / (n * sum(x))
        where x is the sorted-ascending sequence of |relevance| values.

    Properties:
        - Range [0, 1].
        - 1 = perfectly concentrated (one pixel has all the mass).
        - 0 = uniform.
        - Returns 0.0 when the heatmap is all zero.

    Notes:
        - Operates on absolute values, so negative relevance is handled.
        - Robust to a few negative outliers via abs().
    """
    flat = np.abs(np.asarray(heatmap, dtype=np.float64)).ravel()
    total = flat.sum()
    if total <= 0:
        return 0.0
    flat.sort()  # ascending
    n = flat.size
    # 1-indexed positions for the standard Gini formula
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(i, flat) - (n + 1) * total) / (n * total))


def _flat_pixel_order_morf(heatmap: np.ndarray) -> np.ndarray:
    """Return flat indices sorted MOST-RELEVANT-FIRST by |R|.
    """
    flat_abs = np.abs(heatmap).ravel()
    return np.argsort(flat_abs)[::-1].copy()


@torch.no_grad()
def deletion_morf_curve(
    ppnet_plain: torch.nn.Module,
    real_image: torch.Tensor,
    pno: int,
    heatmap: np.ndarray,
    fractions: List[float],
    noise_tensor: torch.Tensor,
    device: torch.device,
) -> List[float]:
    """Compute the MoRF deletion curve for one (image, prototype, heatmap).

    Algorithm
    ---------
    1. Sort flat pixel positions by |R| descending (most relevant first).
    2. For each fraction p in ``fractions``:
         - Start from the **real** image.
         - Replace the top-k positions (k = round(p * H*W)) across **all 3
           channels** with the corresponding positions of ``noise_tensor``.
         - Run the (uncanonized) model forward and record pooled prototype
           similarity ``s[pno]``.
    3. Return the list of similarities, one per fraction.

    Args
    ----
    ppnet_plain: uncanonized PPNet (gradient-free forward).
    real_image: ``(1, 3, H, W)`` tensor in [0, 1] on any device.
    pno: prototype index.
    heatmap: 2D numpy ``(H, W)`` heatmap (signed values OK; uses |R| for order).
    fractions: e.g. [0.0, 0.05, ..., 1.0].
    noise_tensor: ``(1, 3, H, W)`` tensor of uniform-noise replacement values,
        already on ``device``. The same noise is used for all fractions of
        this image so that the only variable is the deletion ordering.
    device: torch device.
    """
    assert real_image.dim() == 4 and real_image.size(0) == 1, \
        f"Expected (1, 3, H, W); got {tuple(real_image.shape)}"
    assert noise_tensor.shape == real_image.shape, \
        f"Noise shape {tuple(noise_tensor.shape)} must equal image shape {tuple(real_image.shape)}"
    C, H, W = real_image.shape[1], real_image.shape[2], real_image.shape[3]
    n_pixels = H * W

    real_on = real_image.to(device).contiguous()
    noise_on = noise_tensor.to(device).contiguous()

    order = _flat_pixel_order_morf(heatmap)
    assert order.shape == (n_pixels,), \
        f"Order length {order.shape[0]} does not match H*W={n_pixels} (heatmap shape={heatmap.shape})"
    order_t = torch.from_numpy(order.astype(np.int64)).to(device)

    similarities: List[float] = []
    for p in fractions:
        k = int(round(float(p) * n_pixels))
        k = max(0, min(k, n_pixels))

        current = real_on.clone()
        if k > 0:
            current_flat = current.view(1, C, -1)
            noise_flat = noise_on.view(1, C, -1)
            idx = order_t[:k]
            current_flat[:, :, idx] = noise_flat[:, :, idx]
            current = current_flat.view(1, C, H, W)

        sim, _pred = get_similarity_and_prediction(ppnet_plain, current, pno, device)
        similarities.append(float(sim))

    return similarities


def aopc_metrics_from_drops(drops: List[float], fractions: List[float]) -> Dict[str, float]:
    """Summarize a deletion drop curve into the six headline metrics.

    Inputs
    ------
    drops: per-fraction similarity drop = s(real) - s(perturbed).
    fractions: same length as drops, monotonically increasing, in [0, 1].

    Outputs (dict)
    --------------
    aopc_drop_at_5pct, aopc_drop_at_10pct
        Drop after replacing the top 5% / 10% most relevant pixels.
        Picked by nearest-fraction lookup (no interpolation) so the
        numbers are exactly the on-disk per-fraction values for fairness.
    aopc_mean_drop
        Mean drop across all fractions (classical AOPC; includes the
        fraction-0 zero-drop for normalization consistency).
    aopc_full_auc
        Trapezoidal AUC of the drop curve over [0, 1]. Equivalent (up to a
        constant) to the classical AOPC integral.
    aopc_partial_auc_10
        Trapezoidal AUC of the drop curve over [0, 0.10]. This is the
        sparsity-aware summary: it focuses on the regime where both PRP and
        PLRP have meaningfully ordered pixels (their non-zero buckets).
    """
    arr = np.asarray(drops, dtype=np.float64)
    f = np.asarray(fractions, dtype=np.float64)
    n = arr.size
    assert n == f.size, f"drops/fractions length mismatch ({n} vs {f.size})"

    def _nearest(target: float) -> float:
        idx = int(np.argmin(np.abs(f - target)))
        return float(arr[idx])

    full_auc = float(np.trapz(arr, f))
    mask10 = f <= 0.10 + 1e-12
    if int(mask10.sum()) >= 2:
        partial_auc_10 = float(np.trapz(arr[mask10], f[mask10]))
    else:
        partial_auc_10 = 0.0
    mean_drop = float(arr.mean())

    return {
        "aopc_drop_at_5pct": _nearest(0.05),
        "aopc_drop_at_10pct": _nearest(0.10),
        "aopc_mean_drop": mean_drop,
        "aopc_partial_auc_10": partial_auc_10,
        "aopc_full_auc": full_auc,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not rows:
        return out
    for k in keys:
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "n": int(vals.size),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare baseline PRP vs PLRP-PRP on PLRP-paper metrics: "
            "Gini coefficient (sparsity) and AOPC-MoRF (perturbation faithfulness)."
        ),
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to Epoch_*_after_protopushing.pth (or Lightning .ckpt).")
    p.add_argument("--param_jsonpath", type=str, required=True,
                   help="Path to params JSON (e.g. config/params_example_ordinal.json).")
    p.add_argument("--artifact_dir", type=str, required=True,
                   help="Folder of artifact-augmented .jpeg images.")
    p.add_argument("--artifact_csv", type=str, required=True,
                   help="CSV with image_name + artifact_label (only label==1 rows used).")
    p.add_argument("--original_dir", type=str, default="",
                   help="Optional: folder of original images. Defaults to --artifact_dir.")

    p.add_argument("--prototypes", type=int, nargs="+", default=[],
                   help="Prototype indices to evaluate.")
    p.add_argument("--num_images", type=int, default=0,
                   help="If > 0, randomly subsample this many images.")
    p.add_argument("--num_fractions", type=int, default=21,
                   help="Deletion-curve resolution (default 21: 0%%, 5%%, ..., 100%%).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling + noise.")

    p.add_argument("--plrp_p_pos", type=float, default=0.25,
                   help="PLRP-lambda: proportion of positive relevance to prune. Default 0.25.")
    p.add_argument("--plrp_p_neg", type=float, default=0.125,
                   help="PLRP-lambda: proportion of negative relevance to prune. Default 0.125.")

    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--skip_prototype_activation", action="store_true",
                   help="Skip the upsampled prototype_activation method (faster; PRP vs PLRP only).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if not args.prototypes:
        raise SystemExit("Provide at least one --prototypes index.")
    if not 0.0 <= args.plrp_p_pos < 1.0:
        raise SystemExit("--plrp_p_pos must be in [0, 1).")
    if not 0.0 <= args.plrp_p_neg < 1.0:
        raise SystemExit("--plrp_p_neg must be in [0, 1).")
    if args.num_fractions < 3:
        raise SystemExit("--num_fractions must be >= 3.")

    include_proto = not args.skip_prototype_activation

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}", flush=True)

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    img_size = int(network_params.img_size)
    print(f"Architecture: {network_params.base_architecture}, img_size={img_size}", flush=True)

    discovery_original = Path(args.original_dir) if args.original_dir else Path(args.artifact_dir)
    items: List[Tuple[str, Path, Path]] = discover_evaluation_images(
        artifact_csv=Path(args.artifact_csv),
        original_dir=discovery_original,
        artifact_dir=Path(args.artifact_dir),
    )
    if not items:
        raise SystemExit("No artifact images matched (artifact_label==1 with both files present).")
    rng = np.random.default_rng(args.seed)
    if args.num_images > 0 and len(items) > args.num_images:
        sel = rng.choice(len(items), size=args.num_images, replace=False)
        items = [items[i] for i in sorted(sel)]
    print(f"Evaluating {len(items)} image(s) on prototypes {sorted(set(args.prototypes))}", flush=True)

    proto_indices = sorted(set(args.prototypes))

    # --- Three independent PPNet instances --------------------------------
    print("Loading 3 PPNet instances:", flush=True)
    print("  1/3 -> ppnet_proto         (uncanonized; activation map + forward)", flush=True)
    ppnet_proto = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet_proto.base_architecture = network_params.base_architecture

    print("  2/3 -> ppnet_baseline      (canonized for baseline PRP)", flush=True)
    ppnet_baseline = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet_baseline.base_architecture = network_params.base_architecture
    prp_baseline = PRPCanonizedModelBaseline(ppnet_baseline).to(device).eval()

    set_plrp_params(args.plrp_p_pos, args.plrp_p_neg)
    eff_p_pos, eff_p_neg = get_plrp_params()
    print(f"  3/3 -> ppnet_plrp          (canonized for PLRP-PRP; p_pos={eff_p_pos}, p_neg={eff_p_neg})",
          flush=True)
    ppnet_plrp = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet_plrp.base_architecture = network_params.base_architecture
    prp_plrp = PRPCanonizedModelPlrp(ppnet_plrp).to(device).eval()
    if eff_p_pos == 0.0 and eff_p_neg == 0.0:
        print("  NOTE: p_pos=p_neg=0 -> PLRP curves should match baseline PRP exactly (sanity mode).",
              flush=True)

    # --- Validate prototype indices ---------------------------------------
    n_proto_eff = int(ppnet_proto.num_prototypes)
    bad = [p for p in proto_indices if p < 0 or p >= n_proto_eff]
    if bad:
        raise SystemExit(f"Invalid prototype index(es) {bad}; valid range 0..{n_proto_eff - 1}.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = np.linspace(0.0, 1.0, int(args.num_fractions)).tolist()

    methods_used: List[str] = (
        ["prototype_activation", "prp_baseline", "plrp_prp"]
        if include_proto else ["prp_baseline", "plrp_prp"]
    )

    per_image_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    n_items = len(items)
    for i, (stem, _orig_path, art_path) in enumerate(items):
        img_t = load_image(art_path, img_size=img_size).to(device)
        # Reproducible per-image noise (independent of method).
        noise_np = rng.uniform(0.0, 1.0, size=(img_t.shape[1], img_size, img_size)).astype(np.float32)
        noise_t = torch.from_numpy(noise_np).unsqueeze(0).contiguous().to(device)

        for pno in proto_indices:
            # Always compute baseline-PRP and PLRP-PRP heatmaps; optionally prototype activation.
            heatmaps: Dict[str, np.ndarray] = {}

            if include_proto:
                heatmaps["prototype_activation"] = get_prototype_heatmap(
                    ppnet_proto, img_t, pno, device,
                )

            heatmaps["prp_baseline"] = generate_prp_image_baseline(
                img_t.clone(), pno, prp_baseline, device,
            )
            heatmaps["plrp_prp"] = generate_prp_image_plrp(
                img_t.clone(), pno, prp_plrp, device,
            )

            # Sanity checks on heatmap shapes (catches silent transposes etc.)
            for name, hm in heatmaps.items():
                if hm.ndim != 2 or hm.shape != (img_size, img_size):
                    raise SystemExit(
                        f"{name} heatmap shape {hm.shape} != expected ({img_size}, {img_size})"
                    )

            # Per-method metrics.
            row_pair: Dict[str, Any] = {"image": stem, "prototype": pno}
            method_metrics: Dict[str, Dict[str, float]] = {}
            for name, hm in heatmaps.items():
                g = gini_coefficient(hm)
                sims = deletion_morf_curve(
                    ppnet_proto, img_t, pno, hm, fractions, noise_t, device,
                )
                s_real = sims[0]  # similarity at fraction 0 (no deletion)
                drops = [s_real - s for s in sims]
                aopc = aopc_metrics_from_drops(drops, fractions)
                method_metrics[name] = {"gini": float(g), **aopc}

                per_image_rows.append({
                    "image": stem,
                    "prototype": pno,
                    "method": name,
                    "n_fractions": int(len(fractions)),
                    "similarity_real": float(s_real),
                    **method_metrics[name],
                })

            # Paired delta PLRP - PRP for each metric (only when both present).
            if "prp_baseline" in method_metrics and "plrp_prp" in method_metrics:
                for k in METRIC_KEYS:
                    row_pair[f"delta_{k}"] = float(
                        method_metrics["plrp_prp"][k] - method_metrics["prp_baseline"][k]
                    )
                paired_rows.append(row_pair)

        if (i + 1) % 5 == 0 or (i + 1) == n_items:
            print(
                f"  [{i+1}/{n_items}] {stem}  ({time.perf_counter() - t0:.1f}s elapsed)",
                flush=True,
            )

    if not per_image_rows:
        raise SystemExit("No rows collected — exiting.")

    # ----- per_image_metrics.csv ------------------------------------------
    fieldnames = ["image", "prototype", "method", "n_fractions", "similarity_real"] + METRIC_KEYS
    per_csv = out_dir / "per_image_metrics.csv"
    with per_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_image_rows:
            w.writerow(r)
    print(f"Wrote {per_csv}", flush=True)

    # ----- summary_per_prototype.csv --------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for pno in proto_indices:
        for method in methods_used:
            sel = [r for r in per_image_rows if r["prototype"] == pno and r["method"] == method]
            agg = _aggregate(sel, METRIC_KEYS)
            row: Dict[str, Any] = {"prototype": pno, "method": method, "n_images": len(sel)}
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = agg[k]["mean"] if agg else None
                row[f"{k}_std"] = agg[k]["std"] if agg else None
            summary_rows.append(row)
    sum_csv = out_dir / "summary_per_prototype.csv"
    with sum_csv.open("w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    print(f"Wrote {sum_csv}", flush=True)

    # ----- paired_deltas_per_image.csv ------------------------------------
    paired_csv = out_dir / "paired_deltas_per_image.csv"
    pd_keys = ["image", "prototype"] + [f"delta_{k}" for k in METRIC_KEYS]
    with paired_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pd_keys)
        w.writeheader()
        for r in paired_rows:
            w.writerow(r)
    print(f"Wrote {paired_csv}", flush=True)

    # ----- comparison_paired.json -----------------------------------------
    overall: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods_used:
        sel = [r for r in per_image_rows if r["method"] == method]
        overall[method] = _aggregate(sel, METRIC_KEYS)

    delta_summary: Dict[str, Dict[str, float]] = {}
    for k in METRIC_KEYS:
        dk = f"delta_{k}"
        arr = np.array([float(r[dk]) for r in paired_rows], dtype=np.float64) if paired_rows else np.array([])
        delta_summary[k] = {
            "mean_delta_plrp_minus_prp": float(arr.mean()) if arr.size else 0.0,
            "std_delta": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "fraction_strictly_positive": float((arr > 0).mean()) if arr.size else 0.0,
            "fraction_non_negative": float((arr >= 0).mean()) if arr.size else 0.0,
            "n_pairs": int(arr.size),
        }

    comparison = {
        "config": {
            "ckpt": str(args.ckpt),
            "param_jsonpath": str(args.param_jsonpath),
            "plrp_p_pos": float(eff_p_pos),
            "plrp_p_neg": float(eff_p_neg),
            "prototypes": proto_indices,
            "num_images_requested": int(args.num_images),
            "num_images_in_loop": int(n_items),
            "num_fractions": int(args.num_fractions),
            "include_prototype_activation": bool(include_proto),
        },
        "metrics_overall_mean": overall,
        "paired_prp_vs_plrp": delta_summary,
    }
    comp_path = out_dir / "comparison_paired.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    print(f"Wrote {comp_path}", flush=True)

    # ----- Console summary ------------------------------------------------
    print("\n=== Overall mean (all image × prototype pairs) ===", flush=True)
    hdr = f"{'metric':<22} {'PRP':>12} {'PLRP':>12} {'Δ(PLRP-PRP)':>14}  {'PLRP>PRP':>9}"
    if include_proto:
        hdr = f"{'metric':<22} {'proto_act':>10} {'PRP':>12} {'PLRP':>12} {'Δ(PLRP-PRP)':>14}  {'PLRP>PRP':>9}"
    print(hdr, flush=True)
    for k in METRIC_KEYS:
        b = overall["prp_baseline"][k]["mean"]
        l = overall["plrp_prp"][k]["mean"]
        d = delta_summary[k]["mean_delta_plrp_minus_prp"]
        fr = delta_summary[k]["fraction_strictly_positive"]
        line_tail = f"{b:>12.4f} {l:>12.4f} {d:>14.4f}  {100*fr:>8.1f}%"
        if include_proto:
            pa = overall["prototype_activation"][k]["mean"]
            print(f"  {k:<20} {pa:>10.4f} {line_tail}", flush=True)
        else:
            print(f"  {k:<20} {line_tail}", flush=True)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
