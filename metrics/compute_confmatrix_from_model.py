"""
Compute confusion matrix CSV from a saved INSightR-Net model on the test set.

Usage:
    python -m metrics.compute_confmatrix_from_model \
        --model_path <path to .pth saved model> \
        --param_jsonpath config/params_example_ordinal.json \
        --datapath <root folder containing train/ and test/ subdirs> \
        --output_dir <directory where CSV will be saved>

Plots (saved in --output_dir, same folder as predictions_test.csv by default):
    Always: predictions_raw_by_true_label.png
    If --artifact_class is 1–5 and --artifact_labels_csv is set:
        predictions_raw_true_class{k}_by_artifact.png

Example with artifact overlay diagnostics:
    python -m metrics.compute_confmatrix_from_model \\
        --model_path ... --datapath ... --output_dir ... \\
        --artifact_class 3 \\
        --artifact_labels_csv /path/to/test_labeled_data.csv

Plots only from an existing CSV (no GPU):
    python -m metrics.compute_confmatrix_from_model \\
        --from_predictions_csv path/to/predictions_test.csv \\
        --artifact_class 3 \\
        --artifact_labels_csv path/to/test_labeled_data.csv
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from define_parameters import NetworkParams, Parameters
from helpers import load_json, plot_confmatrix, save_confmatrix_csv
from datamodule import MyDataModuleDiabRet
from insight_training import model


def image_name_stem(name: str) -> str:
    """Match predictions_test.csv names (e.g. foo.jpeg) to artifact CSV (e.g. foo)."""
    return Path(name).stem


def plot_predictions_by_true_label(
    df: pd.DataFrame,
    savepath: Path,
    min_label: int,
    max_label: int,
    seed: int = 0,
) -> None:
    """Scatter: y = continuous prediction_raw, x = true_label with jitter; color = true_label."""
    rng = np.random.default_rng(seed)
    labels = df["true_label"].to_numpy()
    preds = df["prediction_raw"].to_numpy()
    jitter = rng.uniform(-0.25, 0.25, size=len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    n_cls = max_label - min_label + 1
    base_colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_cls, 3)))[:n_cls]
    for t in range(min_label, max_label + 1):
        mask = labels == t
        if not np.any(mask):
            continue
        color = base_colors[t - min_label]
        ax.scatter(
            labels[mask] + jitter[mask],
            preds[mask],
            s=12,
            alpha=0.5,
            label=f"true {t}",
            color=color,
            edgecolors="none",
        )

    for boundary in np.arange(min_label - 0.5, max_label + 0.5):
        ax.axhline(boundary, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    ax.set_xlabel("True label (ordinal class)")
    ax.set_ylabel("Raw prediction (continuous)")
    ax.set_xticks(list(range(min_label, max_label + 1)))
    ax.set_title("Test predictions: raw output vs true label")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_artifact_class_subset(
    df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    artifact_class: int,
    savepath: Path,
    seed: int = 0,
) -> None:
    """
    For true_label == artifact_class only: color by artifact_label (0 = no, 1 = yes).
    artifact_df columns: image_name (no extension ok), artifact_label.
    """
    art = artifact_df.copy()
    art["image_stem"] = art["image_name"].astype(str).map(image_name_stem)
    flag_map = dict(zip(art["image_stem"], art["artifact_label"].astype(int)))

    sub = df[df["true_label"] == artifact_class].copy()
    sub["image_stem"] = sub["image_name"].astype(str).map(image_name_stem)
    sub["artifact_label"] = sub["image_stem"].map(flag_map)
    missing = sub["artifact_label"].isna().sum()
    if missing:
        print(
            f"Warning: {missing} test rows for true_label=={artifact_class} "
            f"have no row in artifact CSV; they are omitted from artifact plot."
        )
    sub = sub.dropna(subset=["artifact_label"])
    sub["artifact_label"] = sub["artifact_label"].astype(int)

    if sub.empty:
        print(f"No rows left for artifact-class plot (true_label={artifact_class}). Skipping.")
        return

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.12, 0.12, size=len(sub))
    preds = sub["prediction_raw"].to_numpy()
    flags = sub["artifact_label"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    labels_map = {0: "No artifact", 1: "With artifact"}
    for flag in (0, 1):
        mask = flags == flag
        if not np.any(mask):
            continue
        ax.scatter(
            flags[mask].astype(float) + jitter[mask],
            preds[mask],
            s=18,
            alpha=0.55,
            label=labels_map[flag],
            color=colors[flag],
            edgecolors="none",
        )

    ax.set_xlabel("Artifact (0 = no, 1 = yes, jittered for visibility)")
    ax.set_ylabel("Raw prediction (continuous)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (no)", "1 (yes)"])
    ax.set_title(f"True class {artifact_class} only: raw prediction vs artifact presence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def load_ppnet(model_path, network_params, device="cuda"):
    """Construct PPNet and load weights from a .pth state dict file."""
    ppnet = model.construct_PPNet(network_params=network_params)
    state_dict = torch.load(model_path, map_location=device)
    ppnet.load_state_dict(state_dict)
    ppnet.eval()
    ppnet = ppnet.to(device)
    return ppnet


def run_inference(ppnet, dataloader, min_label, max_label, device="cuda"):
    """Run the model on a dataloader and return per-image predictions."""
    image_names = []
    true_labels = []
    pred_labels = []
    pred_raw_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference on test set"):
            images, labels, filenames = batch
            images = images.to(device)
            output, _, _ = ppnet(images, return_convs=False)

            pred_reg = output.squeeze().cpu()
            pred_raw_values.extend(pred_reg.numpy().tolist())

            pred_cls = torch.clamp(
                torch.round(pred_reg).long(), min=min_label, max=max_label
            )
            true_cls = labels.round().squeeze().long()

            image_names.extend(filenames)
            true_labels.extend(true_cls.numpy().tolist())
            pred_labels.extend(pred_cls.numpy().tolist())

    return (
        image_names,
        np.array(true_labels, dtype=int),
        np.array(pred_labels, dtype=int),
        pred_raw_values,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute confusion matrix CSV from a saved INSightR-Net model"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to the saved .pth model (not needed if --from_predictions_csv is set).",
    )
    parser.add_argument(
        "--param_jsonpath", default="config/params_example_ordinal.json",
        help="Path to the parameter JSON used during training",
    )
    parser.add_argument(
        "--datapath",
        default=None,
        help="Root data directory (train/ and test/); not needed if --from_predictions_csv is set.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save the CSV output. Defaults to same directory as model_path.",
    )
    parser.add_argument(
        "--cv_fold", type=int, default=0,
        help="Cross-validation fold index (default: 0)",
    )
    parser.add_argument(
        "--run_name", default="confmatrix_run",
        help="Run name (used internally for path setup)",
    )
    parser.add_argument(
        "--savepath", default="confmatrix_tmp",
        help="Base save path (used internally for parameter initialization)",
    )
    parser.add_argument(
        "--pretrained_path", default="config/pretrained_model.ckpt",
        help="Path to pretrained model (not used, but required by Parameters)",
    )
    parser.add_argument(
        "--artifact_class",
        type=int,
        default=-1,
        help="1–5: true DR class that may carry artifact (enables second plot). -1: skip artifact plot (default).",
    )
    parser.add_argument(
        "--artifact_labels_csv",
        type=str,
        default=None,
        help="CSV with columns image_name, artifact_label (0=no, 1=yes). Required if --artifact_class is 1–5.",
    )
    parser.add_argument(
        "--from_predictions_csv",
        default=None,
        help="Load an existing predictions_test.csv and only save plots (skip model and confusion matrix).",
    )
    parser.add_argument(
        "--min_label",
        type=int,
        default=1,
        help="Minimum ordinal class label (used for plot axes with --from_predictions_csv).",
    )
    parser.add_argument(
        "--max_label",
        type=int,
        default=5,
        help="Maximum ordinal class label (used for plot axes with --from_predictions_csv).",
    )
    args = parser.parse_args()

    if args.artifact_class != -1 and (
        args.artifact_class < 1 or args.artifact_class > 5
    ):
        parser.error("--artifact_class must be -1 or an integer from 1 to 5")
    if args.artifact_class != -1 and not args.artifact_labels_csv:
        parser.error(
            "--artifact_labels_csv is required when --artifact_class is between 1 and 5"
        )

    if args.from_predictions_csv:
        preds_in = Path(args.from_predictions_csv)
        if not preds_in.is_file():
            raise FileNotFoundError(f"predictions CSV not found: {preds_in}")
        df = pd.read_csv(preds_in)
        need = {"image_name", "true_label", "prediction_raw"}
        if not need.issubset(df.columns):
            raise ValueError(
                f"--from_predictions_csv must contain columns {need}, got {list(df.columns)}"
            )
        output_dir = Path(args.output_dir) if args.output_dir else preds_in.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_by_label_path = output_dir / "predictions_raw_by_true_label.png"
        plot_predictions_by_true_label(
            df,
            plot_by_label_path,
            min_label=args.min_label,
            max_label=args.max_label,
        )
        print(f"Saved raw prediction plot (by true label) to: {plot_by_label_path}")
        if args.artifact_class != -1:
            artifact_path = Path(args.artifact_labels_csv)
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Artifact labels CSV not found: {artifact_path}")
            artifact_df = pd.read_csv(artifact_path)
            required = {"image_name", "artifact_label"}
            if not required.issubset(set(artifact_df.columns)):
                raise ValueError(
                    f"Artifact CSV must have columns {required}, got {list(artifact_df.columns)}"
                )
            plot_art_path = (
                output_dir
                / f"predictions_raw_true_class{args.artifact_class}_by_artifact.png"
            )
            plot_artifact_class_subset(
                df,
                artifact_df,
                args.artifact_class,
                plot_art_path,
            )
            print(f"Saved artifact-class subset plot to: {plot_art_path}")
        return

    if not args.model_path or not args.datapath:
        parser.error("--model_path and --datapath are required unless --from_predictions_csv is set")

    params_dict = load_json(args.param_jsonpath)
    params_fields = {f.name for f in Parameters.__dataclass_fields__.values()}
    args_for_params = {
        k: v for k, v in vars(args).items() if k in params_fields
    }
    params_exp = {**params_dict, **args_for_params}
    params = Parameters.from_dict(params_exp)
    params.cv_fold = args.cv_fold
    params.set_savepaths()

    datamodule = MyDataModuleDiabRet(params)
    test_loader = datamodule.test_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppnet = load_ppnet(args.model_path, params.network_params, device=device)

    image_names, true_labels, pred_labels, pred_raw = run_inference(
        ppnet, test_loader, params.min_label, params.max_label, device=device
    )

    n_classes = params.max_label - params.min_label + 1
    class_names = [str(i) for i in range(params.min_label, params.max_label + 1)]

    true_0based = true_labels - params.min_label
    pred_0based = pred_labels - params.min_label

    cm = confusion_matrix(true_0based, pred_0based, labels=np.arange(n_classes))

    acc = accuracy_score(true_0based, pred_0based)
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Confusion matrix (rows=true, cols=predicted), classes {class_names}:")
    print(cm)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "testing_confmatrix.csv"
    save_confmatrix_csv(cm, classes=class_names, savepath=csv_path)
    print(f"\nSaved confusion matrix CSV to: {csv_path}")

    png_path = output_dir / "testing_confmatrix.png"
    plot_confmatrix(cm, classes=class_names, savepath=png_path)
    print(f"Saved confusion matrix PNG to: {png_path}")

    predictions_path = output_dir / "predictions_test.csv"
    df = pd.DataFrame({
        "image_name": [Path(n).name for n in image_names],
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "prediction_raw": pred_raw,
    })
    df.to_csv(predictions_path, index=False)
    print(f"Saved per-image predictions to: {predictions_path}")

    plot_by_label_path = output_dir / "predictions_raw_by_true_label.png"
    plot_predictions_by_true_label(
        df,
        plot_by_label_path,
        min_label=params.min_label,
        max_label=params.max_label,
    )
    print(f"Saved raw prediction plot (by true label) to: {plot_by_label_path}")

    if args.artifact_class != -1:
        artifact_path = Path(args.artifact_labels_csv)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact labels CSV not found: {artifact_path}")
        artifact_df = pd.read_csv(artifact_path)
        required = {"image_name", "artifact_label"}
        if not required.issubset(set(artifact_df.columns)):
            raise ValueError(
                f"Artifact CSV must have columns {required}, got {list(artifact_df.columns)}"
            )
        plot_art_path = (
            output_dir
            / f"predictions_raw_true_class{args.artifact_class}_by_artifact.png"
        )
        plot_artifact_class_subset(
            df,
            artifact_df,
            args.artifact_class,
            plot_art_path,
        )
        print(f"Saved artifact-class subset plot to: {plot_art_path}")


if __name__ == "__main__":
    main()
