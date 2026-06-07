#!/usr/bin/env python3
"""
Ordinal regression test metrics for INSightR-Net (same setup as training / confusion matrix).

Computed metrics
-----------------
**Regression (continuous output vs ordinal true label 1..K)**
  - MAE, MSE, RMSE on raw predictions
  - R² (coefficient of determination)

**Classification-style (rounded predictions, clamped to [min_label, max_label])**
  - Accuracy (same as rounding pipeline in ``run_inference``)
  - Quadratic weighted Cohen's kappa (same weighting as ``torchmetrics.CohenKappa(..., weights='quadratic')`` in Lightning)

**Per true class** (true label c = 1..K, same grouping as ``last_layer_retrain``)
  - n, rounded_accuracy, MAE (continuous) for samples with that true label

**Focus class subset** (default: true label 3, Clever-Hans artifact class)
  - Same regression + discrete metrics computed only on test rows with ``true_label == focus_class``.
  - Use ``--focus_class 0`` to skip this block.

Outputs (in ``--output_dir``)
  - ``regression_test_summary.json`` — full test metrics (top level) + ``metrics_true_class_K_only``
  - ``predictions_test.csv`` — optional mirror of per-image preds (same columns as ``metrics.compute_confmatrix_from_model``)

In order to run from CleverHansRegression root::

    python -m metrics.evaluate_ordinal_regression_test \\
        --model_path path/to/Epoch_50_after_protopushing.pth \\
        --datapath /path/to/dataset_root_with_train_and_test \\
        --param_jsonpath config/params_example_ordinal.json \\
        --output_dir path/to/save_metrics

Optional: point test split at a different JSON::

    --test_config config/datasplit/dr_config/dr_test_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# CleverHansRegression root (parent of ``metrics/``)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from metrics.compute_confmatrix_from_model import load_ppnet, run_inference  # noqa: E402
from datamodule import MyDataModuleDiabRet  # noqa: E402
from define_parameters import Parameters  # noqa: E402
from helpers import load_json, plot_confmatrix, save_confmatrix_csv  # noqa: E402


def _json_sanitize(obj: Any) -> Any:
    """Replace NaN/Inf with None so JSON is RFC-compliant."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def _aggregate_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_raw: np.ndarray,
    min_label: int,
    max_label: int,
) -> Dict[str, Any]:
    """Build summary dict from parallel arrays (true int, pred int, pred float)."""
    y = true_labels.astype(np.float64)
    pr = pred_raw.astype(np.float64)
    pl = pred_labels.astype(np.int64)

    mae = float(mean_absolute_error(y, pr))
    mse = float(mean_squared_error(y, pr))
    rmse = float(np.sqrt(mse))
    # R² undefined if Var(y)==0
    if np.var(y) < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y, pr))

    acc = float(accuracy_score(true_labels, pl))
    qwk = float(
        cohen_kappa_score(true_labels, pl, labels=list(range(min_label, max_label + 1)), weights="quadratic")
    )

    n_classes = max_label - min_label + 1
    true_0 = true_labels - min_label
    pred_0 = pred_labels - min_label
    cm = confusion_matrix(true_0, pred_0, labels=np.arange(n_classes))

    per_class: Dict[str, Any] = {}
    for c in range(min_label, max_label + 1):
        mask = true_labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            per_class[str(c)] = {"n": 0, "rounded_accuracy": None, "mae": None}
            continue
        acc_c = float(np.mean(pl[mask] == true_labels[mask]))
        mae_c = float(np.mean(np.abs(pr[mask] - y[mask])))
        per_class[str(c)] = {"n": n_c, "rounded_accuracy": acc_c, "mae": mae_c}

    return {
        "n_samples": int(len(y)),
        "min_label": min_label,
        "max_label": max_label,
        "regression_on_raw": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        },
        "discrete_rounded": {
            "accuracy": acc,
            "quadratic_weighted_kappa": qwk,
        },
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "class_names_row_col": [str(i) for i in range(min_label, max_label + 1)],
        "per_class_true_label": per_class,
    }


def _aggregate_metrics_true_class_only(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_raw: np.ndarray,
    min_label: int,
    max_label: int,
    focus_class: int,
) -> Dict[str, Any]:
    """
    metrics for artifact grade, which is class 3
    """
    mask = true_labels == focus_class
    n = int(mask.sum())
    if n == 0:
        return {
            "true_label": focus_class,
            "n_samples": 0,
            "note": "No test images with this true label in the current split.",
        }
    return {
        "true_label": focus_class,
        **_aggregate_metrics(
            true_labels[mask],
            pred_labels[mask],
            pred_raw[mask],
            min_label,
            max_label,
        ),
    }


def _predictions_from_arrays(
    image_names: List[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_raw: List[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_name": [Path(n).name for n in image_names],
            "true_label": true_labels,
            "predicted_label": pred_labels,
            "prediction_raw": pred_raw,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ordinal regression + discrete metrics on test set (INSightR-Net .pth)."
    )
    p.add_argument("--model_path", default=None, help="Path to saved .pth (not needed with --from_predictions_csv).")
    p.add_argument(
        "--datapath",
        default=None,
        help="Dataset root with train/ and test/ (same as training). Not needed with --from_predictions_csv.",
    )
    p.add_argument("--param_jsonpath", default="config/params_example_ordinal.json")
    p.add_argument("--output_dir", default=None, help="Where to write JSON/CSV/plots. Default: next to model or CSV.")
    p.add_argument("--cv_fold", type=int, default=0)
    p.add_argument("--run_name", default="metrics_eval")
    p.add_argument("--savepath", default="metrics_tmp", help="Dummy base path for Parameters (unused for saving).")
    p.add_argument("--pretrained_path", default="config/pretrained_model.ckpt")
    p.add_argument(
        "--test_config",
        default=None,
        help="Optional path to dr_test_config.json (overrides JSON datasplittest_path).",
    )
    p.add_argument(
        "--from_predictions_csv",
        default=None,
        help="Skip model; load predictions_test.csv with columns image_name, true_label, prediction_raw.",
    )
    p.add_argument("--min_label", type=int, default=1)
    p.add_argument("--max_label", type=int, default=5)
    p.add_argument(
        "--focus_class",
        type=int,
        default=3,
        help="Also compute metrics on test images with this true label only (e.g. 3 = artifact class). "
        "Use 0 to disable.",
    )
    args = p.parse_args()

    if args.from_predictions_csv:
        csv_path = Path(args.from_predictions_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        need = {"true_label", "prediction_raw"}
        if not need.issubset(df.columns):
            raise ValueError(f"CSV must contain {need}, got {list(df.columns)}")
        y_true = df["true_label"].to_numpy().astype(int)
        pr = df["prediction_raw"].to_numpy(dtype=np.float64)
        min_l, max_l = args.min_label, args.max_label
        pl = np.clip(np.round(pr), min_l, max_l).astype(int)
        summary = _aggregate_metrics(y_true, pl, pr, min_l, max_l)
        if args.focus_class != 0:
            if args.focus_class < min_l or args.focus_class > max_l:
                p.error(f"--focus_class must be between {min_l} and {max_l}, or 0 to disable")
            key = f"metrics_true_class_{args.focus_class}_only"
            summary[key] = _aggregate_metrics_true_class_only(
                y_true, pl, pr, min_l, max_l, args.focus_class
            )
        out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not args.model_path or not args.datapath:
            p.error("--model_path and --datapath are required unless --from_predictions_csv is set")

        params_dict = load_json(args.param_jsonpath)
        params_fields = {f.name for f in Parameters.__dataclass_fields__.values()}
        args_for_params = {k: v for k, v in vars(args).items() if k in params_fields}
        params_exp = {**params_dict, **args_for_params}
        params = Parameters.from_dict(params_exp)
        params.cv_fold = args.cv_fold
        if args.test_config:
            params.datasplittest_path = Path(args.test_config)
        params.set_savepaths()

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dm = MyDataModuleDiabRet(params)
        loader = dm.test_dataloader()
        ppnet = load_ppnet(args.model_path, params.network_params, device=device)

        image_names, true_labels, pred_labels, pred_raw = run_inference(
            ppnet, loader, params.min_label, params.max_label, device=device
        )
        pr_arr = np.array(pred_raw, dtype=np.float64)
        summary = _aggregate_metrics(
            true_labels, pred_labels, pr_arr, params.min_label, params.max_label
        )
        if args.focus_class != 0:
            if args.focus_class < params.min_label or args.focus_class > params.max_label:
                p.error(
                    f"--focus_class must be between {params.min_label} and {params.max_label}, or 0 to disable"
                )
            key = f"metrics_true_class_{args.focus_class}_only"
            summary[key] = _aggregate_metrics_true_class_only(
                true_labels,
                pred_labels,
                pr_arr,
                params.min_label,
                params.max_label,
                args.focus_class,
            )

        out_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        df = _predictions_from_arrays(image_names, true_labels, pred_labels, pred_raw)
        pred_path = out_dir / "predictions_test.csv"
        df.to_csv(pred_path, index=False)
        print(f"Wrote: {pred_path}")

        min_l, max_l = params.min_label, params.max_label

    json_path = out_dir / "regression_test_summary.json"
    with json_path.open("w") as f:
        json.dump(_json_sanitize(summary), f, indent=2)
    print(f"Wrote: {json_path}")

    # Console summary — full test set (top-level keys)
    reg = summary["regression_on_raw"]
    dis = summary["discrete_rounded"]
    print("\n--- Full test set (all classes in split) ---")
    print(f"n_samples: {summary['n_samples']}")
    print(f"MAE (raw vs true class index):  {reg['mae']:.6f}")
    print(f"MSE:                             {reg['mse']:.6f}")
    print(f"RMSE:                            {reg['rmse']:.6f}")
    print(f"R²:                              {reg['r2']}")
    print(f"Accuracy (rounded):              {dis['accuracy']:.6f}")
    print(f"Quadratic weighted kappa:        {dis['quadratic_weighted_kappa']:.6f}")

    subset_keys = [k for k in summary if k.startswith("metrics_true_class_") and k.endswith("_only")]
    for sk in sorted(subset_keys):
        block = summary[sk]
        print(f"\n--- Subset: {sk} ---")
        if block.get("n_samples", 0) == 0 and "note" in block:
            print(block["note"])
            continue
        reg_s = block["regression_on_raw"]
        dis_s = block["discrete_rounded"]
        print(f"true_label: {block['true_label']}  n_samples: {block['n_samples']}")
        print(f"MAE:   {reg_s['mae']:.6f}")
        print(f"MSE:   {reg_s['mse']:.6f}")
        print(f"RMSE:  {reg_s['rmse']:.6f}")
        print(f"R²:    {reg_s['r2']}")
        print(f"Accuracy (rounded):       {dis_s['accuracy']:.6f}")
        print(f"Quadratic weighted kappa: {dis_s['quadratic_weighted_kappa']:.6f}")

    cm = np.array(summary["confusion_matrix_rows_true_cols_pred"])
    classes = summary["class_names_row_col"]
    save_confmatrix_csv(cm, classes=classes, savepath=out_dir / "testing_confmatrix.csv")
    plot_confmatrix(cm, classes=classes, savepath=out_dir / "testing_confmatrix.png")
    print(f"Wrote: {out_dir / 'testing_confmatrix.csv'}")
    print(f"Wrote: {out_dir / 'testing_confmatrix.png'}")


if __name__ == "__main__":
    main()
