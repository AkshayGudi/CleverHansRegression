"""
Relevance ordering test split by artifact presence (artifact_label in CSV).

Same insertion protocol as relevance_ordering_general.py, but:
  - Input CSV must have columns: image_name, artifact_label
    (image_name without .jpeg suffix; artifact_label 1 = artifact, 0 = clean)
  - Runs the full test separately for artifact_label==1 and artifact_label==0
  - Writes results under subfolders:
        <output_dir>/with_artifact/
        <output_dir>/without_artifact/

Usage:
    python relevance_ordering_artifact_split.py \\
        --model_path saved_models/Epoch_50_after_protopushing.pth \\
        --param_jsonpath config/params_example_ordinal.json \\
        --image_dir /path/to/test/jpeg/folder \\
        --artifact_labels_csv /path/to/test_labeled_data.csv \\
        --output_dir relevance_ordering_split_results \\
        --topk_prototypes 3
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from define_parameters import NetworkParams
from helpers import load_json
from prp.insight_prp import PRPCanonizedModel
from insight_training.model import construct_PPNet

# Reuse all core logic from relevance_ordering_general.py (single source of truth)
from prp.relevance_ordering_general import (
    load_ppnet,
    load_image,
    process_single_image,
    get_topk_prototypes,
    plot_mean_curves,
)


def _normalize_artifact_label(val):
    """Accept 0/1, bool, or string '0'/'1'."""
    if isinstance(val, (bool, np.bool_)):
        return 1 if val else 0
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "artifact"):
        return 1
    if s in ("0", "false", "no", "clean"):
        return 0
    raise ValueError(f"Cannot interpret artifact_label: {val!r}")


def load_image_paths_from_artifact_csv(csv_path, image_dir):
    """
    Read CSV with image_name + artifact_label, return two lists of Path:
      paths_with_artifact, paths_without_artifact
    Only includes rows whose file exists under image_dir as <name>.jpeg
    """
    df = pd.read_csv(csv_path)
    needed = {"image_name", "artifact_label"}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV must contain columns {sorted(needed)}. Missing: {sorted(missing_cols)}. "
            f"Found: {list(df.columns)}"
        )

    image_dir = Path(image_dir)
    with_art = []
    without_art = []
    missing_files = []

    for _, row in df.iterrows():
        name = str(row["image_name"]).strip()
        # strip .jpeg if user accidentally included it
        if name.lower().endswith(".jpeg"):
            name = name[:-5]
        label = _normalize_artifact_label(row["artifact_label"])
        p = image_dir / f"{name}.jpeg"
        if not p.exists():
            missing_files.append(str(p))
            continue
        if label == 1:
            with_art.append(p)
        else:
            without_art.append(p)

    if missing_files:
        print(
            f"WARNING: {len(missing_files)} path(s) from CSV not found on disk "
            f"(first few: {missing_files[:5]})"
        )

    return with_art, without_art


def run_relevance_batch(
    name,
    image_paths,
    out_dir,
    ppnet,
    prp_model,
    network_params,
    device,
    args,
    fractions,
    seed_offset,
    artifact_label_value,
):
    """
    Run the full relevance-ordering loop for one group of images.

    name: short label for logs, e.g. 'with_artifact'
    seed_offset: added to args.seed for per-image seeds (keeps groups independent)
    artifact_label_value: 1 or 0 written in CSV rows for this group
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print(f"\n[{name}] No images — skipping.")
        return

    img_size = network_params.img_size
    detail_csv = out_dir / "relevance_ordering_per_image.csv"

    fraction_cols_prp = [f"prp_sim_{f:.2f}" for f in fractions]
    fraction_cols_proto = [f"proto_sim_{f:.2f}" for f in fractions]
    fraction_cols_rand = [f"rand_sim_{f:.2f}" for f in fractions]
    all_columns = (
        ["image", "prototype", "artifact_label", "auc_prp", "auc_proto", "auc_rand",
         "auc_delta_prp_minus_proto", "prp_better"]
        + fraction_cols_prp + fraction_cols_proto + fraction_cols_rand
    )

    if not detail_csv.exists():
        pd.DataFrame(columns=all_columns).to_csv(detail_csv, index=False)
        print(f"[{name}] Created {detail_csv}")
    else:
        print(f"[{name}] Appending to {detail_csv}")

    all_prp_sim = []
    all_proto_sim = []
    all_rand_sim = []
    all_prp_pred = []
    all_proto_pred = []
    all_rand_pred = []

    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[{name}] [{img_idx+1}/{len(image_paths)}] {img_path.name}")
        image_tensor = load_image(str(img_path), img_size=img_size)

        if args.prototype_index is not None:
            proto_indices = [args.prototype_index]
        else:
            proto_indices = get_topk_prototypes(
                ppnet, image_tensor, args.topk_prototypes, device
            )
            print(f"  Top-{args.topk_prototypes} prototypes: {proto_indices}")

        rows_this_image = []
        for pno in proto_indices:
            print(f"  Prototype {pno}: computing orderings + insertion curves ...")
            result = process_single_image(
                ppnet, prp_model, image_tensor, pno, fractions, device,
                seed=args.seed + seed_offset + img_idx,
            )

            all_prp_sim.append(result["prp_sim"])
            all_proto_sim.append(result["proto_sim"])
            all_rand_sim.append(result["rand_sim"])
            all_prp_pred.append(result["prp_pred"])
            all_proto_pred.append(result["proto_pred"])
            all_rand_pred.append(result["rand_pred"])

            auc_prp = np.trapz(result["prp_sim"], fractions)
            auc_proto = np.trapz(result["proto_sim"], fractions)
            auc_rand = np.trapz(result["rand_sim"], fractions)
            auc_delta = auc_prp - auc_proto

            rows_this_image.append({
                "image": img_path.name,
                "prototype": pno,
                "artifact_label": artifact_label_value,
                "auc_prp": round(auc_prp, 6),
                "auc_proto": round(auc_proto, 6),
                "auc_rand": round(auc_rand, 6),
                "auc_delta_prp_minus_proto": round(auc_delta, 6),
                "prp_better": auc_delta > 0,
                **{f"prp_sim_{f:.2f}": s for f, s in zip(fractions, result["prp_sim"])},
                **{f"proto_sim_{f:.2f}": s for f, s in zip(fractions, result["proto_sim"])},
                **{f"rand_sim_{f:.2f}": s for f, s in zip(fractions, result["rand_sim"])},
            })

        pd.DataFrame(rows_this_image).to_csv(detail_csv, mode="a", header=False, index=False)
        print(f"  Saved to CSV ({img_idx+1}/{len(image_paths)} done)")

    # Aggregate
    prp_sim_arr = np.array(all_prp_sim)
    proto_sim_arr = np.array(all_proto_sim)
    rand_sim_arr = np.array(all_rand_sim)
    prp_pred_arr = np.array(all_prp_pred)
    proto_pred_arr = np.array(all_proto_pred)
    rand_pred_arr = np.array(all_rand_pred)

    mean_prp_sim = prp_sim_arr.mean(axis=0)
    mean_proto_sim = proto_sim_arr.mean(axis=0)
    mean_rand_sim = rand_sim_arr.mean(axis=0)
    std_prp_sim = prp_sim_arr.std(axis=0)
    std_proto_sim = proto_sim_arr.std(axis=0)
    std_rand_sim = rand_sim_arr.std(axis=0)

    mean_prp_pred = prp_pred_arr.mean(axis=0)
    mean_proto_pred = proto_pred_arr.mean(axis=0)
    mean_rand_pred = rand_pred_arr.mean(axis=0)
    std_prp_pred = prp_pred_arr.std(axis=0)
    std_proto_pred = proto_pred_arr.std(axis=0)
    std_rand_pred = rand_pred_arr.std(axis=0)

    detail_df = pd.read_csv(detail_csv)
    n_prp_better = detail_df["prp_better"].sum()
    n_total = len(detail_df)
    print(f"\n[{name}] PRP better than Prototype in {n_prp_better}/{n_total} "
          f"({100*n_prp_better/max(n_total,1):.1f}%) of (image, prototype) pairs.")
    print(f"[{name}] Mean AUC — PRP: {detail_df['auc_prp'].mean():.4f}  |  "
          f"Proto: {detail_df['auc_proto'].mean():.4f}  |  "
          f"Random: {detail_df['auc_rand'].mean():.4f}")

    summary_df = pd.DataFrame({
        "fraction": fractions,
        "mean_prp_sim": mean_prp_sim,
        "mean_proto_sim": mean_proto_sim,
        "mean_rand_sim": mean_rand_sim,
        "std_prp_sim": std_prp_sim,
        "std_proto_sim": std_proto_sim,
        "std_rand_sim": std_rand_sim,
        "mean_prp_pred": mean_prp_pred,
        "mean_proto_pred": mean_proto_pred,
        "mean_rand_pred": mean_rand_pred,
    })
    summary_csv = out_dir / "relevance_ordering_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[{name}] Saved summary: {summary_csv}")

    title_suffix = " (with artifact)" if artifact_label_value == 1 else " (without artifact)"
    plot_mean_curves(
        fractions,
        mean_prp_sim, mean_proto_sim, mean_rand_sim,
        std_prp_sim, std_proto_sim, std_rand_sim,
        ylabel="Prototype Similarity (higher = better)",
        title=f"Relevance Ordering: Prototype Similarity{title_suffix}",
        savepath=str(out_dir / "relevance_ordering_similarity.png"),
    )
    plot_mean_curves(
        fractions,
        mean_prp_pred, mean_proto_pred, mean_rand_pred,
        std_prp_pred, std_proto_pred, std_rand_pred,
        ylabel="Regression Prediction (continuous)",
        title=f"Relevance Ordering: Regression Output{title_suffix}",
        savepath=str(out_dir / "relevance_ordering_prediction.png"),
    )

    sim_p = out_dir / "relevance_ordering_similarity.png"
    pred_p = out_dir / "relevance_ordering_prediction.png"
    plot_mean_curves(
        fractions,
        mean_prp_sim, mean_proto_sim, mean_rand_sim,
        None,
        None,
        None,
        ylabel="Prototype Similarity (higher = better)",
        title=f"Relevance Ordering: Prototype Similarity{title_suffix}",
        savepath=str(sim_p.with_name(sim_p.stem + "_mean_only" + sim_p.suffix)),
    )
    plot_mean_curves(
        fractions,
        mean_prp_pred, mean_proto_pred, mean_rand_pred,
        None,
        None,
        None,
        ylabel="Regression Prediction (continuous)",
        title=f"Relevance Ordering: Regression Output{title_suffix}",
        savepath=str(pred_p.with_name(pred_p.stem + "_mean_only" + pred_p.suffix)),
    )

    print(f"[{name}] Done. {n_total} rows in per-image CSV. Results in: {out_dir}")


def maybe_subsample(paths, num_images, rng):
    if num_images > 0 and len(paths) > num_images:
        chosen = rng.choice(len(paths), size=num_images, replace=False)
        return [paths[i] for i in sorted(chosen)]
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Relevance ordering split by artifact_label (1 vs 0) from CSV"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--param_jsonpath", required=True)
    parser.add_argument("--image_dir", required=True,
                        help="Folder containing <image_name>.jpeg files")
    parser.add_argument("--artifact_labels_csv", required=True,
                        help="CSV with columns: image_name, artifact_label (1=artifact, 0=no)")
    parser.add_argument("--output_dir", default="relevance_ordering_split_results",
                        help="Base folder; creates with_artifact/ and without_artifact/ inside")
    parser.add_argument("--prototype_index", type=int, default=None)
    parser.add_argument("--topk_prototypes", type=int, default=3)
    parser.add_argument("--num_images", type=int, default=-1,
                        help="Per group: max images (-1 = all in that group)")
    parser.add_argument("--num_fractions", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    print(f"Architecture: {network_params.base_architecture}, img_size={network_params.img_size}")

    ppnet = load_ppnet(args.model_path, network_params, device)
    ppnet_for_prp = load_ppnet(args.model_path, network_params, device)
    ppnet_for_prp.base_architecture = network_params.base_architecture
    prp_model = PRPCanonizedModel(ppnet_for_prp).to(device)
    print("Canonized model ready for PRP.")

    with_paths, without_paths = load_image_paths_from_artifact_csv(
        args.artifact_labels_csv, args.image_dir
    )
    print(f"\nFrom CSV: {len(with_paths)} image(s) with artifact, "
          f"{len(without_paths)} without artifact (files found on disk).")

    rng = np.random.default_rng(args.seed)
    with_paths = maybe_subsample(with_paths, args.num_images, rng)
    without_paths = maybe_subsample(without_paths, args.num_images, rng)
    if args.num_images > 0:
        print(f"After --num_images={args.num_images} per group: "
              f"{len(with_paths)} with, {len(without_paths)} without.")

    fractions = np.linspace(0.0, 1.0, args.num_fractions).tolist()
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Optional one-line summary file at base output
    summary_lines = [
        f"artifact_labels_csv: {args.artifact_labels_csv}",
        f"image_dir: {args.image_dir}",
        f"with_artifact_count: {len(with_paths)}",
        f"without_artifact_count: {len(without_paths)}",
    ]
    (base_out / "split_counts.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {base_out / 'split_counts.txt'}")

    # Different seed offsets so baseline noise differs between groups (optional clarity)
    run_relevance_batch(
        "with_artifact",
        with_paths,
        base_out / "with_artifact",
        ppnet,
        prp_model,
        network_params,
        device,
        args,
        fractions,
        seed_offset=0,
        artifact_label_value=1,
    )
    run_relevance_batch(
        "without_artifact",
        without_paths,
        base_out / "without_artifact",
        ppnet,
        prp_model,
        network_params,
        device,
        args,
        fractions,
        seed_offset=1_000_000,
        artifact_label_value=0,
    )

    print(f"\nAll done. Base output directory: {base_out}")


if __name__ == "__main__":
    main()
