#!/usr/bin/env python3
"""
Prototype ablation (no retraining) for INSightR-Net.

Mirrors the *functional* idea of PRP paper Fig. 3: remove prototype contributions,
re-evaluate on a fixed test subset, report drop in rounded-grade accuracy and
change in MAE — **without** changing any weights.

Example (class 3, artifact-only, class-3 prototypes from your experiment):

  cd /path/to/CleverHansRegression
  python3 -m ablation.run_prototype_ablation \
    --ckpt /sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v14_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_28_Apr_1/saved_models/Epoch_50_after_protopushing.pth \
    --param_jsonpath /sc/home/akshay.gudi/code/CleverHansRegression/config/params_example_ordinal.json \
    --test_dir /sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14/test \
    --test_config /sc/home/akshay.gudi/code/CleverHansRegression/config/datasplit/dr_config/dr_test_config.json \
    --artifact_csv /sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14/data_details_class3/test_labeled_data.csv \
    --target_class 3 \
    --ablation_indices 19 21 24 25 26 27 28 29 30 31 32 33 35 37 39 3 \
    --output_dir /sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v14_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_28_Apr_1/img/ablation_epoch50 \
    --batch_size 16 \
    --device cuda

  # Numerical self-check (no data paths required):
  python3 ablation/run_prototype_ablation.py --self_check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# CleverHansRegression root (parent of ``ablation/``)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json  # noqa: E402
from insight_training.model import construct_PPNet  # noqa: E402

from ablation.forward_ablation import (  # noqa: E402
    assert_masked_forward_matches_standard,
    forward_logits_with_prototypes_masked,
)


# ---------------------------------------------------------------------------
# Image loading (must match training: BGR, HWC -> CHW with permute(2,0,1))
# ---------------------------------------------------------------------------


def load_image_tensor(path: Path, img_size: int) -> torch.Tensor:
    """Load one image as (1, 3, H, W) float in [0, 1], BGR — same as main_generate_prp."""
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if im.shape[0] != img_size or im.shape[1] != img_size:
        im = cv2.resize(im, (img_size, img_size))
    norm = im.astype(np.float32) / 255.0
    # Dimension fix: HWC -> CHW (see dataset / main_generate_prp).
    t = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
    return t


def discover_eval_image_paths(
    test_dir: Path,
    test_config: dict,
    target_class: str,
    artifact_only: bool,
    artifact_csv: Optional[Path],
) -> List[Tuple[Path, float]]:
    """
    Build list of (jpeg_path, label_float) for evaluation.

    Uses ``dr_test_config.json``-style structure: ``files`` (stem) + ``labels``.
    Keeps only rows whose true label equals ``target_class``.
    If ``artifact_only``, keeps only images listed with ``artifact_label == 1``
    in ``artifact_csv`` (column ``image_name`` without path).
    """
    files: List[str] = test_config["files"]
    labels: List[str] = test_config["labels"]
    if len(files) != len(labels):
        raise ValueError(
            f"test_config: len(files)={len(files)} != len(labels)={len(labels)}"
        )

    artifact_names: Optional[set] = None
    if artifact_only:
        if artifact_csv is None:
            raise ValueError("--artifact_only requires --artifact_csv")
        artifact_names = set()
        with artifact_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("artifact_label", "").strip() == "1":
                    name = row["image_name"].strip()
                    if not name.lower().endswith(".jpeg"):
                        name = name + ".jpeg"
                    artifact_names.add(name)

    out: List[Tuple[Path, float]] = []
    for stem, lab in zip(files, labels):
        if str(lab).strip() != str(target_class).strip():
            continue
        fname = stem + ".jpeg"
        if artifact_names is not None and fname not in artifact_names:
            continue
        p = test_dir / fname
        if not p.is_file():
            continue
        out.append((p, float(lab)))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rounded_grade_accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    """
    Primary metric: fraction of samples where rounded prediction equals
    rounded true grade (ordinal DR labels 1–5 as integers / strings parsed to float).
    """
    pr = torch.round(pred).clamp(1, 5)
    yr = torch.round(y).clamp(1, 5)
    return float((pr == yr).float().mean().item())


def mean_absolute_error(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Secondary metric: mean |pred - y| in grade units."""
    return float(torch.mean(torch.abs(pred - y)).item())


def run_predictions_on_batch(
    ppnet: torch.nn.Module,
    batch: torch.Tensor,
    mask: frozenset,
    device: torch.device,
) -> torch.Tensor:
    """Return predictions (N,) on device for one ablation mask."""
    x = batch.to(device)
    return forward_logits_with_prototypes_masked(ppnet, x, mask)


@torch.no_grad()
def evaluate_mask(
    ppnet: torch.nn.Module,
    paths_labels: Sequence[Tuple[Path, float]],
    mask: frozenset,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Run full eval subset for one mask.

    Returns:
        (rounded_accuracy, mae, n_images_used)
    """
    if not paths_labels:
        raise RuntimeError("No evaluation images after filtering; check paths and filters.")

    preds: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    batch_paths: List[Path] = []
    batch_y: List[float] = []

    def flush():
        nonlocal batch_paths, batch_y
        if not batch_paths:
            return
        ims = torch.cat([load_image_tensor(p, img_size) for p in batch_paths], dim=0)
        y_t = torch.tensor(batch_y, dtype=torch.float32, device=device)
        pr = run_predictions_on_batch(ppnet, ims, mask, device)
        preds.append(pr.detach())
        ys.append(y_t)
        batch_paths = []
        batch_y = []

    for p, y in paths_labels:
        batch_paths.append(p)
        batch_y.append(y)
        if len(batch_paths) >= batch_size:
            flush()
    flush()

    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return (
        rounded_grade_accuracy(pred, y),
        mean_absolute_error(pred, y),
        int(pred.shape[0]),
    )


# ---------------------------------------------------------------------------
# Matrix + plots (PRP-style drop figure)
# ---------------------------------------------------------------------------


def build_drop_matrix(
    index_order: List[int],
    baseline_acc: float,
    acc_by_mask: Dict[frozenset, float],
) -> np.ndarray:
    """
    ``index_order``: row/col prototype indices (sorted).
    Cell [i, j] = baseline_acc - acc after removing prototypes index_order[i] and
    index_order[j] (if i != j), or only index_order[i] if i == j.
    """
    n = len(index_order)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                key = frozenset([index_order[i]])
            else:
                key = frozenset([index_order[i], index_order[j]])
            acc = acc_by_mask[key]
            mat[i, j] = baseline_acc - acc
    return mat


def _write_matrix_csv_with_labels(path: Path, mat: np.ndarray, index_order: List[int]) -> None:
    """Write matrix with header row/column of prototype indices (for spreadsheets)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [str(k) for k in index_order])
        for i, row in enumerate(mat):
            w.writerow([str(index_order[i])] + [f"{v:.6f}" for v in row])


def save_drop_matrix_plot(
    mat: np.ndarray,
    index_order: List[int],
    out_path: Path,
    title: str,
) -> None:
    """Save heatmap of accuracy *drops* (higher = more important coalition)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(index_order)))
    ax.set_yticks(range(len(index_order)))
    ax.set_xticklabels([str(k) for k in index_order], rotation=45, ha="right")
    ax.set_yticklabels([str(k) for k in index_order])
    ax.set_xlabel("Prototype index")
    ax.set_ylabel("Prototype index")
    ax.set_title(title + "\n(diagonal: one prototype removed; off-diagonal: pair removed)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ rounded accuracy (baseline − ablated)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_mae_matrix_plot(
    mat: np.ndarray,
    index_order: List[int],
    out_path: Path,
    title: str,
) -> None:
    """MAE *increase* relative to baseline: mae_ablated − mae_baseline (positive = worse)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(index_order)))
    ax.set_yticks(range(len(index_order)))
    ax.set_xticklabels([str(k) for k in index_order], rotation=45, ha="right")
    ax.set_yticklabels([str(k) for k in index_order])
    ax.set_title(title + "\n(diagonal: one prototype removed; off-diagonal: pair removed)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ MAE (ablated − baseline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-check (no dataset)
# ---------------------------------------------------------------------------


def run_self_check() -> None:
    """Construct a random PPNet and verify empty-mask equivalence to forward."""
    np_net = NetworkParams()
    net = construct_PPNet(network_params=np_net)
    net.eval()
    device = torch.device("cpu")
    net.to(device)
    x = torch.rand(4, 3, np_net.img_size, np_net.img_size, device=device)
    assert_masked_forward_matches_standard(net, x)
    # Removing every prototype should break denominator / produce finite numbers
    all_idx = frozenset(range(net.num_prototypes))
    with torch.no_grad():
        out = forward_logits_with_prototypes_masked(net, x, all_idx)
    if not torch.isfinite(out).all():
        raise AssertionError("All-prototypes-masked predictions should still be finite")
    print("ablation self_check: OK (empty mask matches forward; all-masked finite).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="INSightR-Net prototype ablation (no retraining) — rounded accuracy + MAE.",
    )
    p.add_argument("--ckpt", type=str, default="", help="Path to Epoch_*_after_protopushing.pth")
    p.add_argument("--param_jsonpath", type=str, default="", help="JSON with network_params block")
    p.add_argument("--test_dir", type=str, default="", help="Folder containing test JPEGs")
    p.add_argument("--test_config", type=str, default="", help="dr_test_config.json")
    p.add_argument("--artifact_csv", type=str, default="", help="Optional CSV with image_name,artifact_label")
    p.add_argument("--target_class", type=str, default="3", help="True grade string in test_config, e.g. 3")
    p.add_argument(
        "--artifact_only",
        action="store_true",
        help="Keep only images with artifact_label==1 in artifact_csv",
    )
    p.add_argument(
        "--ablation_indices",
        type=int,
        nargs="+",
        required=False,
        help="Prototype indices for the n×n matrix (class-specific pool)",
    )
    p.add_argument("--output_dir", type=str, default="ablation_output")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument(
        "--self_check",
        action="store_true",
        help="Run numerical sanity checks and exit (no dataset).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return

    if not args.ckpt or not args.param_jsonpath or not args.test_dir or not args.test_config:
        raise SystemExit("Need --ckpt --param_jsonpath --test_dir --test_config (or use --self_check).")
    if not args.ablation_indices:
        raise SystemExit("Need --ablation_indices ...")

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    params_dict = load_json(args.param_jsonpath)
    net_dict = params_dict.get("network_params", {})
    network_params = NetworkParams.from_dict(net_dict)
    img_size = network_params.img_size

    ppnet = construct_PPNet(network_params=network_params)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        sub = {k[len("ppnet.") :]: v for k, v in sd.items() if k.startswith("ppnet.")}
        ppnet.load_state_dict(sub, strict=True)
    else:
        ppnet.load_state_dict(ckpt, strict=True)
    ppnet.to(device)
    ppnet.eval()

    test_cfg_path = Path(args.test_config)
    if not test_cfg_path.is_file():
        # allow relative to repo root
        test_cfg_path = _ROOT / args.test_config
    with test_cfg_path.open() as f:
        test_config = json.load(f)

    test_dir = Path(args.test_dir)
    art_path = Path(args.artifact_csv) if args.artifact_csv else None
    paths_labels = discover_eval_image_paths(
        test_dir=test_dir,
        test_config=test_config,
        target_class=args.target_class,
        artifact_only=args.artifact_only,
        artifact_csv=art_path,
    )
    if not paths_labels:
        raise SystemExit(
            "No images matched filters. Check --test_dir, --test_config, --target_class, "
            "and --artifact_csv / --artifact_only."
        )

    index_order = sorted(set(args.ablation_indices))
    for k in index_order:
        if k < 0 or k >= ppnet.num_prototypes:
            raise SystemExit(f"ablation index {k} out of range [0, {ppnet.num_prototypes - 1}]")

    # --- quick numerical check on first batch ---
    with torch.no_grad():
        first = torch.cat(
            [load_image_tensor(paths_labels[0][0], img_size)],
            dim=0,
        )
        assert_masked_forward_matches_standard(ppnet, first.to(device))

    # All masks: baseline + singletons + pairs within index_order
    masks: List[frozenset] = [frozenset()]
    for i in range(len(index_order)):
        masks.append(frozenset([index_order[i]]))
    for i in range(len(index_order)):
        for j in range(i + 1, len(index_order)):
            masks.append(frozenset([index_order[i], index_order[j]]))
    # de-duplicate
    masks = list(dict.fromkeys(masks))

    # Output directory early so cluster jobs show a folder immediately; CSV is
    # appended row-by-row so partial results survive preemption / crash.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_metrics.csv"
    _csv_fieldnames = [
        "removed_prototypes",
        "n_removed",
        "rounded_accuracy",
        "mae",
        "n_images",
        "drop_rounded_accuracy",
        "delta_mae",
    ]

    rows: List[Dict[str, object]] = []
    acc_by: Dict[frozenset, float] = {}
    mae_by: Dict[frozenset, float] = {}
    baseline_acc: Optional[float] = None
    baseline_mae: Optional[float] = None

    t0 = time.perf_counter()
    with csv_path.open("w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=_csv_fieldnames)
        writer.writeheader()
        csv_f.flush()

        for run_idx, m in enumerate(masks):
            label = "none" if not m else "+".join(str(x) for x in sorted(m))
            print(
                f"[ablation {run_idx + 1}/{len(masks)}] removed={label!r}  "
                f"(n_removed={len(m)})  … running full eval on {len(paths_labels)} images",
                flush=True,
            )
            t_mask = time.perf_counter()
            acc, mae, n_used = evaluate_mask(
                ppnet, paths_labels, m, img_size, args.batch_size, device
            )
            elapsed = time.perf_counter() - t_mask

            if baseline_acc is None:
                baseline_acc = acc
                baseline_mae = mae
                drop_ra = 0.0
                delta_m = 0.0
            else:
                drop_ra = baseline_acc - acc
                delta_m = mae - baseline_mae

            row = {
                "removed_prototypes": label,
                "n_removed": len(m),
                "rounded_accuracy": acc,
                "mae": mae,
                "n_images": n_used,
                "drop_rounded_accuracy": drop_ra,
                "delta_mae": delta_m,
            }
            rows.append(row)
            acc_by[m] = acc
            mae_by[m] = mae
            writer.writerow(row)
            csv_f.flush()

            print(
                f"    → done in {elapsed:.1f}s | rounded_acc={acc:.4f} | mae={mae:.4f} | "
                f"drop_acc={drop_ra:.4f} | delta_mae={delta_m:.4f}",
                flush=True,
            )

    assert baseline_acc is not None and baseline_mae is not None
    total_elapsed = time.perf_counter() - t0
    print(
        f"[ablation] finished {len(masks)} masks in {total_elapsed:.1f}s; baseline "
        f"rounded_acc={baseline_acc:.4f}, mae={baseline_mae:.4f}",
        flush=True,
    )

    mat_acc = build_drop_matrix(index_order, baseline_acc, acc_by)
    np.savetxt(out_dir / "drop_matrix_rounded_accuracy.csv", mat_acc, delimiter=",", fmt="%.6f")
    _write_matrix_csv_with_labels(
        out_dir / "drop_matrix_rounded_accuracy_labeled.csv",
        mat_acc,
        index_order,
    )

    # MAE increase matrix
    n = len(index_order)
    mat_mae = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                key = frozenset([index_order[i]])
            else:
                key = frozenset([index_order[i], index_order[j]])
            mat_mae[i, j] = mae_by[key] - baseline_mae
    np.savetxt(out_dir / "delta_mae_matrix.csv", mat_mae, delimiter=",", fmt="%.6f")
    _write_matrix_csv_with_labels(
        out_dir / "delta_mae_matrix_labeled.csv",
        mat_mae,
        index_order,
    )

    meta = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "n_eval_images": len(paths_labels),
        "target_class": args.target_class,
        "artifact_only": bool(args.artifact_only),
        "ablation_indices": index_order,
        "baseline_rounded_accuracy": baseline_acc,
        "baseline_mae": baseline_mae,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    save_drop_matrix_plot(
        mat_acc,
        index_order,
        out_dir / "drop_matrix_rounded_accuracy.png",
        "Prototype ablation: drop in rounded accuracy (no retraining)",
    )
    save_mae_matrix_plot(
        mat_mae,
        index_order,
        out_dir / "delta_mae_matrix.png",
        "Prototype ablation: MAE increase vs baseline (no retraining)",
    )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out_dir / 'drop_matrix_rounded_accuracy.png'}")
    print(f"Baseline rounded_accuracy={baseline_acc:.4f}, MAE={baseline_mae:.4f}, n={len(paths_labels)}")
    best = max(rows, key=lambda r: r["drop_rounded_accuracy"])
    print(
        f"Largest rounded-accuracy drop: {best['drop_rounded_accuracy']:.4f} "
        f"(removed: {best['removed_prototypes']})"
    )


if __name__ == "__main__":
    main()
