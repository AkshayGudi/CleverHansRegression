#!/usr/bin/env python3
"""
Prototype ablation (custom multiple prototypes, no retraining) for INSightR-Net.

This script applies the same logic as ``run_prototype_ablation.py`` but evaluates
exactly one user-provided multi-prototype mask (plus baseline/no-removal), instead
of automatically testing all singletons and pairs.

Use case:
  - You already identified a set of prototypes (e.g., from PRP / pairwise ablation)
  - You want the direct effect of removing *that exact set* on the target class
  - No model weights are changed; only forward-time prototype contributions are masked

source /sc/home/akshay.gudi/conda3/etc/profile.d/conda.sh
conda activate new_insight_env

python3 -m ablation.ablation_multiple \
  --ckpt /sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v14_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_28_Apr_1/saved_models/Epoch_50_after_protopushing.pth \
  --param_jsonpath /sc/home/akshay.gudi/code/CleverHansRegression/config/params_example_ordinal.json \
  --test_dir /sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14/test \
  --test_config /sc/home/akshay.gudi/code/CleverHansRegression/config/datasplit/dr_config/dr_test_config.json \
  --target_class 3 \
  --prototypes_to_remove 3 24 25 \
  --batch_size 16 \
  --device cuda \
  --output_dir /sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v14_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_28_Apr_1/img/ablation_multiple_3_24_25  

If you want artifact-only subset (only images with artifact label 1), add:
    --artifact_only
    --artifact_csv /sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14/data_details_class3/test_labeled_data.csv


"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# CleverHansRegression root (parent of ``ablation/``)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json  # noqa: E402
from insight_training.model import construct_PPNet  # noqa: E402

from ablation.run_prototype_ablation import (  # noqa: E402
    assert_masked_forward_matches_standard,
    discover_eval_image_paths,
    evaluate_mask,
    run_self_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "INSightR-Net ablation for one user-defined multi-prototype set "
            "(plus baseline), no retraining."
        ),
    )
    p.add_argument("--ckpt", type=str, default="", help="Path to Epoch_*_after_protopushing.pth")
    p.add_argument("--param_jsonpath", type=str, default="", help="JSON with network_params block")
    p.add_argument("--test_dir", type=str, default="", help="Folder containing test JPEGs")
    p.add_argument("--test_config", type=str, default="", help="dr_test_config.json")
    p.add_argument("--artifact_csv", type=str, default="", help="Optional CSV with image_name,artifact_label")
    p.add_argument(
        "--target_class",
        type=str,
        default="",
        help="Class/grade with artifact to evaluate (e.g., 3).",
    )
    p.add_argument(
        "--artifact_only",
        action="store_true",
        help="Keep only images with artifact_label==1 in artifact_csv",
    )
    p.add_argument(
        "--prototypes_to_remove",
        type=int,
        nargs="+",
        required=False,
        help="Prototype indices to remove together in one ablation mask.",
    )
    p.add_argument("--output_dir", type=str, default="ablation_multiple_output")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional: cap number of evaluation images for quick smoke tests (0 = all).",
    )
    p.add_argument(
        "--self_check",
        action="store_true",
        help="Run numerical sanity checks and exit (no dataset).",
    )
    return p.parse_args()


def _resolve_test_config(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    p2 = _ROOT / path_str
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Could not find test config: {path_str}")


def _load_ppnet(args: argparse.Namespace, device: torch.device):
    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    ppnet = construct_PPNet(network_params=network_params)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        sub = {k[len("ppnet."):]: v for k, v in sd.items() if k.startswith("ppnet.")}
        ppnet.load_state_dict(sub, strict=True)
    else:
        ppnet.load_state_dict(ckpt, strict=True)

    ppnet.to(device)
    ppnet.eval()
    return ppnet, network_params.img_size


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return

    if not args.ckpt or not args.param_jsonpath or not args.test_dir or not args.test_config:
        raise SystemExit("Need --ckpt --param_jsonpath --test_dir --test_config (or use --self_check).")
    if not str(args.target_class).strip():
        raise SystemExit("Need --target_class (class with artifact to evaluate).")
    if not args.prototypes_to_remove:
        raise SystemExit("Need --prototypes_to_remove ...")

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    ppnet, img_size = _load_ppnet(args, device)
    print(f"Loaded checkpoint: {args.ckpt}", flush=True)

    # Validate and canonicalize prototype mask.
    mask_list = sorted(set(args.prototypes_to_remove))
    if not mask_list:
        raise SystemExit("--prototypes_to_remove cannot be empty")
    for k in mask_list:
        if k < 0 or k >= ppnet.num_prototypes:
            raise SystemExit(f"prototype index {k} out of range [0, {ppnet.num_prototypes - 1}]")
    mask = frozenset(mask_list)
    print(f"Mask (removed together): {mask_list}", flush=True)

    # Load eval subset the same way as existing ablation script.
    test_cfg_path = _resolve_test_config(args.test_config)
    with test_cfg_path.open() as f:
        test_config = json.load(f)

    art_path = Path(args.artifact_csv) if args.artifact_csv else None
    paths_labels = discover_eval_image_paths(
        test_dir=Path(args.test_dir),
        test_config=test_config,
        target_class=args.target_class,
        artifact_only=args.artifact_only,
        artifact_csv=art_path,
    )
    if args.max_images > 0:
        paths_labels = paths_labels[:args.max_images]
    if not paths_labels:
        raise SystemExit(
            "No images matched filters. Check --test_dir, --test_config, "
            "--target_class, and --artifact_csv / --artifact_only."
        )
    print(f"Evaluation images: {len(paths_labels)}", flush=True)

    # Quick correctness check: empty mask path matches standard forward.
    with torch.no_grad():
        from ablation.run_prototype_ablation import load_image_tensor  # noqa: E402
        first = load_image_tensor(paths_labels[0][0], img_size).to(device)
        assert_masked_forward_matches_standard(ppnet, first)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline
    t0 = time.perf_counter()
    print("[1/2] Running baseline (no prototypes removed) ...", flush=True)
    base_acc, base_mae, n_used = evaluate_mask(
        ppnet=ppnet,
        paths_labels=paths_labels,
        mask=frozenset(),
        img_size=img_size,
        batch_size=args.batch_size,
        device=device,
    )
    print(
        f"      baseline: rounded_acc={base_acc:.4f}, mae={base_mae:.4f}, n={n_used}",
        flush=True,
    )

    # Multi-mask ablation
    print("[2/2] Running multi-prototype ablation ...", flush=True)
    abl_acc, abl_mae, n_used2 = evaluate_mask(
        ppnet=ppnet,
        paths_labels=paths_labels,
        mask=mask,
        img_size=img_size,
        batch_size=args.batch_size,
        device=device,
    )
    drop_acc = base_acc - abl_acc
    delta_mae = abl_mae - base_mae
    print(
        f"      ablated:  rounded_acc={abl_acc:.4f}, mae={abl_mae:.4f}, n={n_used2}",
        flush=True,
    )
    print(
        f"      deltas:   drop_acc={drop_acc:.4f}, delta_mae={delta_mae:.4f}",
        flush=True,
    )

    # Save compact outputs
    metrics_rows: List[Dict[str, object]] = [
        {
            "setting": "baseline",
            "removed_prototypes": "none",
            "n_removed": 0,
            "rounded_accuracy": base_acc,
            "mae": base_mae,
            "n_images": n_used,
            "drop_rounded_accuracy": 0.0,
            "delta_mae": 0.0,
        },
        {
            "setting": "multi_ablation",
            "removed_prototypes": "+".join(str(x) for x in mask_list),
            "n_removed": len(mask_list),
            "rounded_accuracy": abl_acc,
            "mae": abl_mae,
            "n_images": n_used2,
            "drop_rounded_accuracy": drop_acc,
            "delta_mae": delta_mae,
        },
    ]
    (out_dir / "ablation_multiple_metrics.json").write_text(json.dumps(metrics_rows, indent=2))

    meta = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "target_class": args.target_class,
        "artifact_only": bool(args.artifact_only),
        "removed_prototypes": mask_list,
        "n_eval_images": len(paths_labels),
        "elapsed_sec_total": time.perf_counter() - t0,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {out_dir / 'ablation_multiple_metrics.json'}", flush=True)
    print(f"Wrote: {out_dir / 'run_meta.json'}", flush=True)


if __name__ == "__main__":
    main()

