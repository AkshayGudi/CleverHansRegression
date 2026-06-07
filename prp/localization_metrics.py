#!/usr/bin/env python3
"""
Ground-truth localization metrics for PRP vs InsightR-Net activation heatmaps.

Metrics
  - Pointing Game (PG)
  - Relevance Mass Accuracy
  - Relevance Rank Accuracy
  - Pixel-wise ROC AUC
  - Top-K IoU

Ground-truth artifact mask is derived per-image via DIFF between the
original (no-artifact) image and the artifact-added image. This handles
arbitrary artifact positions (random or fixed) and the feathered/Poisson
blending.

reusing the same heatmaps as in relevance-ordering test, so that things are comparable.

Example
-------
  cd /path/to/CleverHansRegression
  conda activate new_insight_env

  PYTHONUNBUFFERED=1 python3 -m prp.localization_metrics \\
      --ckpt .../saved_models/Epoch_50_after_protopushing.pth \\
      --param_jsonpath config/params_example_ordinal.json \\
      --original_dir /path/to/data_store/DR/test \\
      --artifact_dir /path/to/data_store/DR/bld_artifact/class3_v14/test \\
      --artifact_csv /path/to/data_store/DR/bld_artifact/class3_v14/data_details_class3/test_labeled_data.csv \\
      --target_class 3 \\
      --prototypes 24 25 27 28 \\
      --num_images 50 \\
      --output_dir .../img/localization_metrics_v14

Self-check (no data needed):
  python3 -m prp.localization_metrics --self_check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# CleverHansRegression root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json  # noqa: E402
from prp.insight_prp import PRPCanonizedModel  # noqa: E402
from prp.relevance_ordering_general import (  # noqa: E402
    get_prp_heatmap,
    get_prototype_heatmap,
    load_image,
    load_ppnet,
)


# ---------------------------------------------------------------------------
# Ground-truth mask derivation (image-diff method)
# ---------------------------------------------------------------------------


def derive_gt_mask_from_position(
    center_x: int,
    center_y: int,
    target_size: int,
    image_w: int,
    image_h: int,
    img_size: int,
    feather_amount: int = 15,
) -> np.ndarray:
    """
    Get the binary mask for the artifact region from the center_x, center_y, target_size.

    Args:
        center_x, center_y: artifact center in the *original* image (pixels).
        target_size: artifact patch side length used during overlay.
        image_w, image_h: original image dimensions stored in the position CSV.
        img_size: model input resolution (final mask size).
        feather_amount: same as ``create_feathered_mask`` default in the
            augmentation script.

    Returns:
        Binary mask (img_size, img_size), uint8 with values {0, 1}.
    """
    mask_full = np.zeros((image_h, image_w), dtype=np.uint8)
    radius = max(1, target_size // 2 - int(feather_amount))
    cv2.circle(mask_full, (int(center_x), int(center_y)), radius, 1, thickness=-1)
    if (image_w, image_h) != (img_size, img_size):
        mask_full = cv2.resize(mask_full, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask_full.astype(np.uint8)


def load_position_csv(path: Path) -> Dict[str, Dict[str, int]]:
    """
    Read artifact_pos_*.csv which has the center_x, center_y, target_size, image_w, image_h 
    information for each image with artifact label 1

    Returns dict: image_stem -> {center_x, center_y, target_size, image_w, image_h}
    """
    out: Dict[str, Dict[str, int]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "center_x", "center_y", "target_size", "image_w", "image_h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"position CSV missing columns: {sorted(missing)}")
        for row in reader:
            stem = row["image_name"].strip()
            if stem.lower().endswith(".jpeg"):
                stem = stem[:-5]
            out[stem] = {
                "center_x": int(row["center_x"]),
                "center_y": int(row["center_y"]),
                "target_size": int(row["target_size"]),
                "image_w": int(row["image_w"]),
                "image_h": int(row["image_h"]),
            }
    return out


def derive_gt_mask_from_diff(
    original_path: Path,
    artifact_path: Path,
    img_size: int,
    diff_threshold: int = 15,
    min_blob_area_frac: float = 0.001,
    morph_kernel: int = 5,
) -> Optional[np.ndarray]:
    """
    Compute a binary GT mask (H, W) by diffing the original and artifact images.

    Steps:
      1. Read both images, resize to ``img_size``.
      2. Per-pixel L1 difference summed over channels
      3. Threshold at ``diff_threshold`` to get a binary mask.
      4. Morphological closing then opening (kernel ``morph_kernel``) to remove speckle noise from JPEG re-encoding.
      5. Keep only the largest connected component if it is larger than ``min_blob_area_frac * img_size**2`` pixels (rejects pure-noise diffs).

    Returns:
        Binary mask (H, W) of dtype uint8 with values {0, 1}, or ``None`` if no
        artifact region is found.
    """
    orig = cv2.imread(str(original_path))
    art = cv2.imread(str(artifact_path))
    if orig is None or art is None:
        return None
    if orig.shape[:2] != (img_size, img_size):
        orig = cv2.resize(orig, (img_size, img_size))
    if art.shape[:2] != (img_size, img_size):
        art = cv2.resize(art, (img_size, img_size))

    diff = cv2.absdiff(orig.astype(np.int16), art.astype(np.int16)).astype(np.uint16)
    diff_gray = diff.sum(axis=2)
    bin_mask = (diff_gray > diff_threshold).astype(np.uint8)

    if bin_mask.sum() == 0:
        return None

    k = max(1, int(morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)

    # Largest connected component, if any of meaningful size.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    if areas[largest - 1] < int(min_blob_area_frac * img_size * img_size):
        return None
    out = (labels == largest).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Localization metrics (heatmap H, mask M, both shape (H, W))
# ---------------------------------------------------------------------------


def _validate_inputs(heatmap: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if heatmap.ndim != 2 or mask.ndim != 2:
        raise ValueError("heatmap and mask must be 2-D")
    if heatmap.shape != mask.shape:
        raise ValueError(f"shape mismatch: heatmap={heatmap.shape}, mask={mask.shape}")
    if mask.sum() == 0:
        raise ValueError("empty GT mask")
    return heatmap.astype(np.float64), mask.astype(np.uint8)


def pointing_game(heatmap: np.ndarray, mask: np.ndarray) -> int:
    """1 if argmax pixel of |heatmap| lies inside ``mask``, else 0."""
    h, m = _validate_inputs(heatmap, mask)
    abs_h = np.abs(h)
    flat_idx = int(np.argmax(abs_h))
    y, x = np.unravel_index(flat_idx, abs_h.shape)
    return int(m[y, x] == 1)


def relevance_mass_accuracy(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total |relevance| mass located inside ``mask``."""
    h, m = _validate_inputs(heatmap, mask)
    abs_h = np.abs(h)
    total = float(abs_h.sum())
    if total <= 0:
        return 0.0
    inside = float(abs_h[m == 1].sum())
    return inside / total


def relevance_rank_accuracy(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Top-K (K=|mask|) most-relevant pixels' overlap with ``mask`` / K."""
    h, m = _validate_inputs(heatmap, mask)
    abs_h = np.abs(h)
    K = int(m.sum())
    if K <= 0:
        return 0.0
    flat_idx = np.argpartition(abs_h.ravel(), -K)[-K:]
    pixels_in_mask = int(m.ravel()[flat_idx].sum())
    return pixels_in_mask / K


def localization_auc(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """
    Pixel-wise ROC AUC: |heatmap| values used as scores, ``mask`` as labels.

    Implemented without sklearn to avoid an extra dependency: rank-sum based.
    """
    h, m = _validate_inputs(heatmap, mask)
    abs_h = np.abs(h).ravel()
    labels = m.ravel().astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(abs_h, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = abs_h.size
    sorted_scores = abs_h[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def topk_iou(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """IoU of the top-K (K=|mask|) most relevant pixels with ``mask``."""
    h, m = _validate_inputs(heatmap, mask)
    abs_h = np.abs(h).ravel()
    K = int(m.sum())
    if K <= 0:
        return 0.0
    flat_idx = np.argpartition(abs_h, -K)[-K:]
    pred = np.zeros_like(m.ravel(), dtype=np.uint8)
    pred[flat_idx] = 1
    inter = int((pred & m.ravel()).sum())
    union = int((pred | m.ravel()).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def all_metrics(heatmap: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Convenience: return all five metrics as a dict."""
    return {
        "pointing_game": float(pointing_game(heatmap, mask)),
        "relevance_mass_accuracy": relevance_mass_accuracy(heatmap, mask),
        "relevance_rank_accuracy": relevance_rank_accuracy(heatmap, mask),
        "localization_auc": localization_auc(heatmap, mask),
        "topk_iou": topk_iou(heatmap, mask),
    }


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def discover_evaluation_images(
    artifact_csv: Path,
    original_dir: Path,
    artifact_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """
    Read the artifact-labeling CSV
    Keep only rows with ``artifact_label == 1`` whose JPEG exists in BOTH
    ``original_dir`` and ``artifact_dir`` so we can derive the GT mask via diff.

    Returns:
        List of (image_stem, original_path, artifact_path).
    """
    out: List[Tuple[str, Path, Path]] = []
    with artifact_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("artifact_label", "").strip() != "1":
                continue
            stem = row["image_name"].strip()
            if stem.lower().endswith(".jpeg"):
                stem = stem[:-5]
            orig = original_dir / (stem + ".jpeg")
            art = artifact_dir / (stem + ".jpeg")
            if not orig.is_file() or not art.is_file():
                continue
            out.append((stem, orig, art))
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check() -> None:
    """Synthetic sanity checks for the metric implementations."""
    rng = np.random.default_rng(0)
    H = W = 64
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[20:30, 20:30] = 1

    # Heatmap perfectly inside mask.
    hp = np.zeros((H, W), dtype=np.float32)
    hp[mask == 1] = 1.0
    res = all_metrics(hp, mask)
    assert res["pointing_game"] == 1, res
    assert abs(res["relevance_mass_accuracy"] - 1.0) < 1e-9, res
    assert abs(res["relevance_rank_accuracy"] - 1.0) < 1e-9, res
    assert abs(res["topk_iou"] - 1.0) < 1e-9, res
    assert res["localization_auc"] >= 0.99, res

    # Heatmap fully outside mask: (point-mass / rank metrics).
    ho = np.zeros((H, W), dtype=np.float32)
    ho[40:50, 40:50] = 1.0
    res = all_metrics(ho, mask)
    assert res["pointing_game"] == 0, res
    assert abs(res["relevance_mass_accuracy"] - 0.0) < 1e-9, res
    assert abs(res["relevance_rank_accuracy"] - 0.0) < 1e-9, res
    assert res["topk_iou"] == 0.0, res

    hi = np.full((H, W), 1.0, dtype=np.float32)
    hi[mask == 1] = 0.0
    res = all_metrics(hi, mask)
    assert res["localization_auc"] <= 0.01, res

    # Random heatmap → roughly chance.
    hr = rng.standard_normal((H, W)).astype(np.float32)
    res = all_metrics(hr, mask)
    assert 0.2 <= res["localization_auc"] <= 0.8, res

    # feathered circle should land at the requested center
    # and resize correctly to the model resolution.
    pmask = derive_gt_mask_from_position(
        center_x=200, center_y=300,
        target_size=100, image_w=540, image_h=540,
        img_size=540, feather_amount=15,
    )
    assert pmask.shape == (540, 540) and pmask.dtype == np.uint8, pmask.shape
    assert pmask[300, 200] == 1, "position mask should be 1 at requested center"
    assert pmask[10, 10] == 0, "position mask should be 0 far from center"
    pmask_rs = derive_gt_mask_from_position(
        center_x=200, center_y=300,
        target_size=100, image_w=540, image_h=540,
        img_size=270, feather_amount=15,
    )
    assert pmask_rs.shape == (270, 270), pmask_rs.shape
    assert pmask_rs[150, 100] == 1, "resized position mask should be 1 near scaled center"

    # Diff-mask sanity: synthesize two images that differ inside a square.
    img_orig = (rng.integers(0, 255, size=(H, W, 3))).astype(np.uint8)
    img_art = img_orig.copy()
    img_art[24:34, 24:34] = (255, 255, 255)
    cv2.imwrite("/tmp/_lm_orig.jpeg", img_orig)
    cv2.imwrite("/tmp/_lm_art.jpeg", img_art)
    derived = derive_gt_mask_from_diff(
        Path("/tmp/_lm_orig.jpeg"),
        Path("/tmp/_lm_art.jpeg"),
        img_size=H,
        diff_threshold=10,
        min_blob_area_frac=1e-4,
        morph_kernel=3,
    )
    assert derived is not None and int(derived.sum()) > 0, "diff-derived mask was empty"
    print("localization_metrics self_check: OK", flush=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Localization-metric evaluation for PRP vs InsightR-Net activation heatmaps.",
    )
    p.add_argument("--ckpt", type=str, default="", help="Path to Epoch_*_after_protopushing.pth")
    p.add_argument("--param_jsonpath", type=str, default="", help="JSON with network_params block")
    p.add_argument("--original_dir", type=str, default="", help="Folder with ORIGINAL (no-artifact) JPEGs (required for diff mode; not needed in position mode)")
    p.add_argument("--artifact_dir", type=str, default="", help="Folder with artifact-added JPEGs (same names as original_dir)")
    p.add_argument("--artifact_csv", type=str, default="", help="CSV with image_name, artifact_label")
    p.add_argument(
        "--position_csv",
        type=str,
        default="",
        help=(
            "Optional artifact_pos_*.csv which has artifact position information "
            "(image_name, center_x, center_y, target_size, image_w, image_h). "
            "If provided, GT masks are reconstructed from these positions instead "
            "of being derived by image diff."
        ),
    )
    p.add_argument(
        "--feather_amount",
        type=int,
        default=15,
        help="Feather amount used during artifact overlay (default 15).",
    )
    p.add_argument("--target_class", type=str, default="", help="Optional: keep only images whose label matches (uses dr_test_config.json if provided)")
    p.add_argument("--test_config", type=str, default="", help="Optional dr_test_config.json for label filtering")
    p.add_argument("--prototypes", type=int, nargs="+", default=[], help="Prototype indices to evaluate (defaults to ALL)")
    p.add_argument("--num_images", type=int, default=0, help="Optional cap on artifact images (0 = all)")
    p.add_argument("--seed", type=int, default=42, help="Subsample seed")
    p.add_argument("--diff_threshold", type=int, default=15, help="Per-pixel L1 sum threshold for GT mask")
    p.add_argument("--min_blob_area_frac", type=float, default=0.001, help="Reject diff masks smaller than this fraction of the image area")
    p.add_argument("--morph_kernel", type=int, default=5, help="Morphological closing/opening kernel size")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--output_dir", type=str, default="localization_metrics_output")
    p.add_argument("--save_per_image_csv", action="store_true", help="Also save per-image rows (large file)")
    p.add_argument("--self_check", action="store_true", help="Run unit-style self-checks and exit")
    return p.parse_args()


def _filter_by_target_class(
    items: List[Tuple[str, Path, Path]],
    target_class: str,
    test_config_path: Path,
) -> List[Tuple[str, Path, Path]]:
    """Keep only stems whose label in test_config equals ``target_class``."""
    cfg = load_json(str(test_config_path))
    files = cfg["files"]
    labels = cfg["labels"]
    if len(files) != len(labels):
        raise ValueError("test_config files/labels length mismatch")
    label_for_stem = {str(s).strip(): str(l).strip() for s, l in zip(files, labels)}
    target = str(target_class).strip()
    return [it for it in items if label_for_stem.get(it[0], None) == target]


def _aggregate(rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    """mean/std for each metric across rows."""
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


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
        return

    base_needed = [args.ckpt, args.param_jsonpath, args.artifact_dir, args.artifact_csv]
    if not all(base_needed):
        raise SystemExit("Need --ckpt --param_jsonpath --artifact_dir --artifact_csv (or use --self_check).")
    if not args.position_csv and not args.original_dir:
        raise SystemExit(
            "Provide either --position_csv (position-based GT mask) "
            "or --original_dir (diff-based GT mask)."
        )

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    img_size = int(network_params.img_size)

    ppnet = load_ppnet(args.ckpt, network_params, device).to(device).eval()
    ppnet.base_architecture = network_params.base_architecture
    ppnet_for_prp = load_ppnet(args.ckpt, network_params, device)
    ppnet_for_prp.base_architecture = network_params.base_architecture
    prp_model = PRPCanonizedModel(ppnet_for_prp).to(device).eval()

    discovery_original = Path(args.original_dir) if args.original_dir else Path(args.artifact_dir)
    items = discover_evaluation_images(
        artifact_csv=Path(args.artifact_csv),
        original_dir=discovery_original,
        artifact_dir=Path(args.artifact_dir),
    )

    position_index: Dict[str, Dict[str, int]] = {}
    if args.position_csv:
        position_index = load_position_csv(Path(args.position_csv))
        print(f"Loaded {len(position_index)} positions from {args.position_csv}")
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
    print(f"Evaluating on {len(items)} artifact image(s).", flush=True)

    if args.prototypes:
        proto_indices: List[int] = sorted(set(args.prototypes))
    else:
        proto_indices = list(range(int(ppnet.num_prototypes)))
    for j in proto_indices:
        if j < 0 or j >= ppnet.num_prototypes:
            raise SystemExit(f"prototype index {j} out of range [0, {ppnet.num_prototypes - 1}]")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "pointing_game",
        "relevance_mass_accuracy",
        "relevance_rank_accuracy",
        "localization_auc",
        "topk_iou",
    ]
    fieldnames = (
        ["image", "prototype", "method", "gt_mask_pixels"]
        + metric_keys
    )
    per_image_rows: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    skipped_no_mask = 0

    print("\n[1/2] Computing per-image metrics ...", flush=True)
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
            skipped_no_mask += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(items)}] {stem}: no GT mask, skipped", flush=True)
            continue

        img_t = load_image(art_path, img_size=img_size).to(device)

        for pno in proto_indices:
            proto_hm = get_prototype_heatmap(ppnet, img_t, pno, device)
            prp_hm = get_prp_heatmap(prp_model, img_t, pno, device)

            for method, hm in (("prototype_activation", proto_hm), ("prp", prp_hm)):
                metrics = all_metrics(hm, gt_mask)
                row = {
                    "image": stem,
                    "prototype": pno,
                    "method": method,
                    "gt_mask_pixels": int(gt_mask.sum()),
                    **metrics,
                }
                per_image_rows.append(row)

        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{i+1}/{len(items)}] {stem}: ok  ({elapsed:.1f}s elapsed; skipped_no_mask={skipped_no_mask})",
                flush=True,
            )

    if not per_image_rows:
        raise SystemExit("All images skipped (no GT mask derived). Check --diff_threshold / paths.")

    if args.save_per_image_csv:
        per_image_path = out_dir / "per_image_metrics.csv"
        with per_image_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)
        print(f"Wrote: {per_image_path}", flush=True)

    print("\n[2/2] Aggregating per-prototype + overall ...", flush=True)
    summary_rows: List[Dict[str, object]] = []
    for pno in proto_indices:
        for method in ("prototype_activation", "prp"):
            sel = [r for r in per_image_rows if r["prototype"] == pno and r["method"] == method]
            agg = _aggregate(sel, metric_keys)
            row = {"prototype": pno, "method": method, "n_images": len(sel)}
            for k in metric_keys:
                row[k + "_mean"] = agg[k]["mean"] if agg else None
                row[k + "_std"] = agg[k]["std"] if agg else None
            summary_rows.append(row)

    overall_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in ("prototype_activation", "prp"):
        sel = [r for r in per_image_rows if r["method"] == method]
        overall_summary[method] = _aggregate(sel, metric_keys)

    summary_csv = out_dir / "summary_per_prototype.csv"
    with summary_csv.open("w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    print(f"Wrote: {summary_csv}", flush=True)

    overall_json = out_dir / "summary_overall.json"
    overall_json.write_text(json.dumps({
        "num_images_evaluated": len(items) - skipped_no_mask,
        "num_images_skipped_no_mask": skipped_no_mask,
        "prototypes": proto_indices,
        "metrics_overall": overall_summary,
    }, indent=2))
    print(f"Wrote: {overall_json}", flush=True)

    # Console preview
    print("\nOverall (mean across all images and prototypes):", flush=True)
    print(f"  {'metric':<26} {'prototype_act':>14} {'prp':>10}", flush=True)
    for k in metric_keys:
        a = overall_summary["prototype_activation"][k]["mean"] if overall_summary["prototype_activation"] else float("nan")
        b = overall_summary["prp"][k]["mean"] if overall_summary["prp"] else float("nan")
        print(f"  {k:<26} {a:>14.4f} {b:>10.4f}", flush=True)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
