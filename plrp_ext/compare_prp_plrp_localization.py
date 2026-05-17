#!/usr/bin/env python3
"""
Compare standard PRP vs PLRP-PRP on the same artifact images using the
existing localization metrics (Pointing Game, RMA, RRA, ROC AUC, Top-K IoU).

This script lives entirely under ``plrp_ext/`` and does not modify
``evaluation/localization_metrics.py``. It imports the same mask builders,
image discovery, and ``all_metrics`` from that module, then computes heatmaps
with:
  - ``insight_prp`` + ``generate_prp_image``  (baseline PRP, unpruned)
  - ``plrp_ext.insight_prp_plrp`` + ``generate_prp_image`` after
    ``set_plrp_params(plrp_p_pos, plrp_p_neg)``  (PLRP-λ)

Optional third method: ``prototype_activation`` (upsampled similarity map),
same as ``localization_metrics.py``.

Outputs (under ``--output_dir``):
  - ``per_image_metrics.csv``     — one row per (image, prototype, method)
  - ``summary_per_prototype.csv`` — mean/std per prototype × method
  - ``paired_deltas_per_image.csv`` — per (image, prototype) Δ = PLRP − PRP
                                      for each metric
  - ``comparison_paired.json``    — overall mean/std per method
                                    (``metrics_overall_mean``) AND mean paired
                                    delta (PLRP − PRP) with the fraction of
                                    pairs where PLRP is strictly better
                                    (``paired_prp_vs_plrp``).

Example (position CSV from overlay pipeline; same flags style as localization_metrics):

  cd /sc/home/akshay.gudi/code/CleverHansRegression
  conda activate new_insight_env

  python3 plrp_ext/compare_prp_plrp_localization.py \\
      --ckpt bld_art_25_Apr/class3_v14_2_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_03_May_3/saved_models/Epoch_50_after_protopushing.pth \\
      --param_jsonpath config/params_example_ordinal.json \\
      --artifact_dir /path/to/artifact/test/jpeg \\
      --artifact_csv /path/to/test_labeled_data.csv \\
      --position_csv /path/to/artifact_pos_test.csv \\
      --prototypes 22 27 33 \\
      --num_images 50 \\
      --plrp_p_pos 0.25 \\
      --plrp_p_neg 0.125 \\
      --output_dir plrp_ext/outputs/loc_compare_class3_v14_2_p025
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Repo root (parent of plrp_ext/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json  # noqa: E402
from insight_prp import PRPCanonizedModel as PRPCanonizedModelBaseline  # noqa: E402
from insight_prp import generate_prp_image as generate_prp_image_baseline  # noqa: E402
from relevance_ordering_paper import (  # noqa: E402
    get_prototype_heatmap,
    load_image,
    load_ppnet,
)
from plrp_ext.lrp_general6_plrp import set_plrp_params  # noqa: E402
from plrp_ext.insight_prp_plrp import PRPCanonizedModel as PRPCanonizedModelPlrp  # noqa: E402
from plrp_ext.insight_prp_plrp import generate_prp_image as generate_prp_image_plrp  # noqa: E402

from evaluation.localization_metrics import (  # noqa: E402
    all_metrics,
    discover_evaluation_images,
    derive_gt_mask_from_diff,
    derive_gt_mask_from_position,
    load_position_csv,
)


METRIC_KEYS = [
    "pointing_game",
    "relevance_mass_accuracy",
    "relevance_rank_accuracy",
    "localization_auc",
    "topk_iou",
]


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


def _filter_by_target_class(
    items: List[Tuple[str, Path, Path]],
    target_class: str,
    test_config_path: Path,
) -> List[Tuple[str, Path, Path]]:
    from helpers import load_json as _lj

    cfg = _lj(str(test_config_path))
    files = cfg["files"]
    labels = cfg["labels"]
    if len(files) != len(labels):
        raise ValueError("test_config files/labels length mismatch")
    label_for_stem = {str(s).strip(): str(l).strip() for s, l in zip(files, labels)}
    target = str(target_class).strip()
    return [it for it in items if label_for_stem.get(it[0], None) == target]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline PRP vs PLRP-PRP on localization metrics (artifact GT mask).",
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to Epoch_*_after_protopushing.pth")
    p.add_argument("--param_jsonpath", type=str, required=True)
    p.add_argument("--artifact_dir", type=str, required=True)
    p.add_argument("--artifact_csv", type=str, required=True)
    p.add_argument("--original_dir", type=str, default="", help="Required for diff-based GT mask")
    p.add_argument(
        "--position_csv",
        type=str,
        default="",
        help="artifact_pos_*.csv for position-based GT mask (recommended)",
    )
    p.add_argument("--feather_amount", type=int, default=15)
    p.add_argument("--target_class", type=str, default="")
    p.add_argument("--test_config", type=str, default="")
    p.add_argument("--prototypes", type=int, nargs="+", default=[], help="Prototype indices (required)")
    p.add_argument("--num_images", type=int, default=0, help="0 = all matched images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diff_threshold", type=int, default=15)
    p.add_argument("--min_blob_area_frac", type=float, default=0.001)
    p.add_argument("--morph_kernel", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--plrp_p_pos", type=float, default=0.25)
    p.add_argument("--plrp_p_neg", type=float, default=0.125)
    p.add_argument(
        "--skip_prototype_activation",
        action="store_true",
        help="Skip upsampled prototype_activation metrics (faster; only PRP vs PLRP).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prototypes:
        raise SystemExit("Provide at least one --prototypes index.")

    include_proto = not args.skip_prototype_activation

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    img_size = int(network_params.img_size)

    if not args.position_csv and not args.original_dir:
        raise SystemExit("Provide --position_csv and/or --original_dir (for diff masks).")

    discovery_original = Path(args.original_dir) if args.original_dir else Path(args.artifact_dir)
    items = discover_evaluation_images(
        artifact_csv=Path(args.artifact_csv),
        original_dir=discovery_original,
        artifact_dir=Path(args.artifact_dir),
    )

    position_index: Dict[str, Dict[str, int]] = {}
    if args.position_csv:
        position_index = load_position_csv(Path(args.position_csv))
        print(f"Loaded {len(position_index)} positions from {args.position_csv}", flush=True)

    if args.target_class:
        if not args.test_config:
            raise SystemExit("--target_class requires --test_config")
        items = _filter_by_target_class(items, args.target_class, Path(args.test_config))
    if not items:
        raise SystemExit("No artifact images matched filters.")
    if args.num_images > 0 and len(items) > args.num_images:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(items), size=args.num_images, replace=False)
        items = [items[i] for i in sorted(idx)]
    print(f"Evaluating {len(items)} image(s), prototypes {sorted(set(args.prototypes))}", flush=True)

    proto_indices = sorted(set(args.prototypes))

    # --- Load three independent model instances (canonization mutates in place) ---
    print("Loading baseline PRP model (insight_prp)...", flush=True)
    ppnet_proto = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet_proto.base_architecture = network_params.base_architecture

    ppnet_baseline = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet_baseline.base_architecture = network_params.base_architecture
    prp_baseline = PRPCanonizedModelBaseline(ppnet_baseline).to(device).eval()

    set_plrp_params(args.plrp_p_pos, args.plrp_p_neg)
    print(
        f"Loading PLRP-PRP model (plrp_ext) with p_pos={args.plrp_p_pos}, p_neg={args.plrp_p_neg}...",
        flush=True,
    )
    ppnet_plrp = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet_plrp.base_architecture = network_params.base_architecture
    prp_plrp = PRPCanonizedModelPlrp(ppnet_plrp).to(device).eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image", "prototype", "method", "gt_mask_pixels"] + METRIC_KEYS
    per_image_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    skipped = 0

    for i, (stem, orig_path, art_path) in enumerate(items):
        gt_mask: Optional[np.ndarray] = None
        if position_index:
            pos = position_index.get(stem)
            if pos is not None:
                gt_mask = derive_gt_mask_from_position(
                    center_x=pos["center_x"],
                    center_y=pos["center_y"],
                    target_size=pos["target_size"],
                    image_w=pos["image_w"],
                    image_h=pos["image_h"],
                    img_size=img_size,
                    feather_amount=args.feather_amount,
                )
        if gt_mask is None and args.original_dir:
            gt_mask = derive_gt_mask_from_diff(
                orig_path,
                art_path,
                img_size=img_size,
                diff_threshold=args.diff_threshold,
                min_blob_area_frac=args.min_blob_area_frac,
                morph_kernel=args.morph_kernel,
            )
        if gt_mask is None or int(gt_mask.sum()) == 0:
            skipped += 1
            continue

        img_t = load_image(art_path, img_size=img_size).to(device)

        for pno in proto_indices:
            if pno < 0 or pno >= int(ppnet_proto.num_prototypes):
                raise SystemExit(f"prototype {pno} out of range [0, {ppnet_proto.num_prototypes - 1}]")

            if include_proto:
                proto_hm = get_prototype_heatmap(ppnet_proto, img_t, pno, device)
                m = all_metrics(proto_hm, gt_mask)
                per_image_rows.append(
                    {
                        "image": stem,
                        "prototype": pno,
                        "method": "prototype_activation",
                        "gt_mask_pixels": int(gt_mask.sum()),
                        **m,
                    }
                )

            prp_hm = generate_prp_image_baseline(img_t.clone(), pno, prp_baseline, device)
            m_prp = all_metrics(prp_hm, gt_mask)
            per_image_rows.append(
                {
                    "image": stem,
                    "prototype": pno,
                    "method": "prp_baseline",
                    "gt_mask_pixels": int(gt_mask.sum()),
                    **m_prp,
                }
            )

            plrp_hm = generate_prp_image_plrp(img_t.clone(), pno, prp_plrp, device)
            m_plrp = all_metrics(plrp_hm, gt_mask)
            per_image_rows.append(
                {
                    "image": stem,
                    "prototype": pno,
                    "method": "plrp_prp",
                    "gt_mask_pixels": int(gt_mask.sum()),
                    **m_plrp,
                }
            )

            pair: Dict[str, Any] = {"image": stem, "prototype": pno}
            for k in METRIC_KEYS:
                pair[f"delta_{k}"] = float(m_plrp[k] - m_prp[k])
            paired_rows.append(pair)

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(
                f"  [{i+1}/{len(items)}] {stem}  (skipped_no_mask={skipped}, {time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if not per_image_rows:
        raise SystemExit("No metrics computed (all images skipped?).")

    per_csv = out_dir / "per_image_metrics.csv"
    with per_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_image_rows:
            w.writerow(r)
    print(f"Wrote {per_csv}", flush=True)

    # Per-prototype summary
    methods = ["prototype_activation", "prp_baseline", "plrp_prp"] if include_proto else ["prp_baseline", "plrp_prp"]
    summary_rows: List[Dict[str, Any]] = []
    for pno in proto_indices:
        for method in methods:
            sel = [r for r in per_image_rows if r["prototype"] == pno and r["method"] == method]
            agg = _aggregate(sel, METRIC_KEYS)
            row: Dict[str, Any] = {"prototype": pno, "method": method, "n_images": len(sel)}
            for k in METRIC_KEYS:
                row[k + "_mean"] = agg[k]["mean"] if agg else None
                row[k + "_std"] = agg[k]["std"] if agg else None
            summary_rows.append(row)

    sum_csv = out_dir / "summary_per_prototype.csv"
    with sum_csv.open("w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    print(f"Wrote {sum_csv}", flush=True)

    paired_csv = out_dir / "paired_deltas_per_image.csv"
    pd_keys = ["image", "prototype"] + [f"delta_{k}" for k in METRIC_KEYS]
    with paired_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pd_keys)
        w.writeheader()
        for r in paired_rows:
            w.writerow(r)
    print(f"Wrote {paired_csv}", flush=True)

    # Overall means per method
    overall: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods:
        sel = [r for r in per_image_rows if r["method"] == method]
        overall[method] = _aggregate(sel, METRIC_KEYS)

    # Paired summary: mean delta + fraction where PLRP > PRP (strict)
    delta_summary: Dict[str, Dict[str, float]] = {}
    for k in METRIC_KEYS:
        dk = f"delta_{k}"
        arr = np.array([float(r[dk]) for r in paired_rows], dtype=np.float64)
        delta_summary[k] = {
            "mean_delta_plrp_minus_prp": float(arr.mean()),
            "std_delta": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "fraction_strictly_positive": float((arr > 0).mean()) if arr.size else 0.0,
            "fraction_non_negative": float((arr >= 0).mean()) if arr.size else 0.0,
            "n_pairs": int(arr.size),
        }

    comparison = {
        "config": {
            "ckpt": str(args.ckpt),
            "param_jsonpath": str(args.param_jsonpath),
            "plrp_p_pos": args.plrp_p_pos,
            "plrp_p_neg": args.plrp_p_neg,
            "prototypes": proto_indices,
            "num_images_requested": args.num_images,
            "num_images_in_loop": len(items),
            "skipped_no_gt_mask": skipped,
            "include_prototype_activation": include_proto,
        },
        "metrics_overall_mean": overall,
        "paired_prp_vs_plrp": delta_summary,
        "interpretation": (
            "For all five metrics, higher is better. "
            "If mean_delta_plrp_minus_prp > 0 and fraction_strictly_positive is high, "
            "PLRP-PRP localizes the GT artifact region better than baseline PRP on average."
        ),
    }
    comp_path = out_dir / "comparison_paired.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    print(f"Wrote {comp_path}", flush=True)

    # Console summary
    print("\n=== Overall mean (all image × prototype pairs) ===", flush=True)
    hdr = f"{'metric':<28} {'prp_baseline':>14} {'plrp_prp':>14} {'delta':>10}"
    if include_proto:
        hdr = f"{'metric':<28} {'proto_act':>12} {'prp_baseline':>14} {'plrp_prp':>14} {'Δ(plrp-prp)':>12}"
    print(hdr, flush=True)
    for k in METRIC_KEYS:
        b = overall["prp_baseline"][k]["mean"]
        l = overall["plrp_prp"][k]["mean"]
        d = delta_summary[k]["mean_delta_plrp_minus_prp"]
        if include_proto:
            pa = overall["prototype_activation"][k]["mean"]
            print(f"  {k:<26} {pa:>12.4f} {b:>14.4f} {l:>14.4f} {d:>12.4f}", flush=True)
        else:
            print(f"  {k:<26} {b:>14.4f} {l:>14.4f} {d:>10.4f}", flush=True)

    print("\n=== Fraction of pairs where PLRP > PRP (strict) ===", flush=True)
    for k in METRIC_KEYS:
        fr = delta_summary[k]["fraction_strictly_positive"]
        print(f"  {k:<28} {fr:.3f}", flush=True)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
