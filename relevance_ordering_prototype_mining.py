"""
Relevance ordering over ALL prototypes on stratified random samples.

Label convention (CSV columns: image_name, artifact_label):
  - artifact_label == 0  →  no artifact (clean)
  - artifact_label == 1  →  image HAS the synthetic artifact

For each of two groups (with artifact / without), draws N random images (default 50),
runs the insertion relevance-ordering test for EVERY prototype index.

PRP maps are computed via the same path as relevance_ordering_general.py:
  insight_prp.generate_prp_image  (LRP backprop from prototype similarity)

"Bad" prototypes (paper Fig.~9 style): high similarity even when pixels are revealed
in random order — we rank by high mean AUC of the random curve relative to PRP.

"Good" prototypes: PRP ordering recovers similarity much faster than upsampled
prototype heatmaps — we rank by high mean (AUC_PRP − AUC_proto).

Outputs per group:
  - per_image_all_prototypes.csv — one row per (image, prototype)
  - per_prototype_summary.csv    — mean AUCs per prototype id
  - bad_prototypes_relevance_ordering.png  — 3 subplots, mean curves for 3 worst prototypes
  - good_prototypes_relevance_ordering.png — 3 subplots for 3 best prototypes

Usage:
  python relevance_ordering_prototype_mining.py \\
      --model_path saved_models/Epoch_50_after_protopushing.pth \\
      --param_jsonpath config/params_example_ordinal.json \\
      --image_dir /path/to/test \\
      --artifact_labels_csv /path/to/test_labeled_data.csv \\
      --output_dir prototype_mining_results \\
      --num_images 50 \\
      --seed 42
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from define_parameters import NetworkParams
from helpers import load_json
from insight_prp import PRPCanonizedModel
from relevance_ordering_general import (
    load_ppnet,
    load_image,
    process_single_image,
    plot_mean_curves,
)

# PRP heatmaps: relevance_ordering_general.get_prp_heatmap → insight_prp.generate_prp_image


def load_paths_by_artifact_labels(csv_path, image_dir):
    """
    artifact_label == 0 → without artifact (clean)
    artifact_label == 1 → with artifact
    """
    df = pd.read_csv(csv_path)
    if "image_name" not in df.columns or "artifact_label" not in df.columns:
        raise ValueError("CSV must have columns: image_name, artifact_label")

    image_dir = Path(image_dir)
    with_art = []
    without_art = []
    missing = []

    for _, row in df.iterrows():
        name = str(row["image_name"]).strip()
        if name.lower().endswith(".jpeg"):
            name = name[:-5]
        lab = int(row["artifact_label"])
        p = image_dir / f"{name}.jpeg"
        if not p.exists():
            missing.append(str(p))
            continue
        if lab == 0:
            without_art.append(p)
        elif lab == 1:
            with_art.append(p)
        else:
            raise ValueError(f"artifact_label must be 0 or 1, got {lab}")

    if missing:
        print(f"WARNING: {len(missing)} CSV rows missing on disk (first 5): {missing[:5]}")
    return with_art, without_art


def sample_random(paths, n, rng):
    if len(paths) <= n:
        return list(paths)
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def run_all_prototypes_for_images(
    image_paths,
    ppnet,
    prp_model,
    img_size,
    fractions,
    device,
    base_seed,
    group_name,
    out_dir,
    num_prototypes,
):
    """
    For each image, run process_single_image for prototype 0 .. num_prototypes-1.
    Append rows to CSV after each image.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "per_image_all_prototypes.csv"

    frac_prp = [f"prp_sim_{f:.2f}" for f in fractions]
    frac_pt = [f"proto_sim_{f:.2f}" for f in fractions]
    frac_rn = [f"rand_sim_{f:.2f}" for f in fractions]
    cols = (
        ["image", "prototype", "auc_prp", "auc_proto", "auc_rand",
         "auc_delta_prp_minus_proto", "prp_better"]
        + frac_prp + frac_pt + frac_rn
    )
    pd.DataFrame(columns=cols).to_csv(detail_csv, index=False)

    rows_buffer = []

    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[{group_name}] image {img_idx+1}/{len(image_paths)}: {img_path.name}")
        tensor = load_image(str(img_path), img_size=img_size)

        for pno in range(num_prototypes):
            if pno % 10 == 0:
                print(f"  prototype {pno}/{num_prototypes} ...")
            result = process_single_image(
                ppnet, prp_model, tensor, pno, fractions, device,
                seed=base_seed + img_idx * 100_000 + pno,
            )
            auc_prp = float(np.trapz(result["prp_sim"], fractions))
            auc_proto = float(np.trapz(result["proto_sim"], fractions))
            auc_rand = float(np.trapz(result["rand_sim"], fractions))
            row = {
                "image": img_path.name,
                "prototype": pno,
                "auc_prp": round(auc_prp, 6),
                "auc_proto": round(auc_proto, 6),
                "auc_rand": round(auc_rand, 6),
                "auc_delta_prp_minus_proto": round(auc_prp - auc_proto, 6),
                "prp_better": (auc_prp - auc_proto) > 0,
            }
            for f, s in zip(fractions, result["prp_sim"]):
                row[f"prp_sim_{f:.2f}"] = s
            for f, s in zip(fractions, result["proto_sim"]):
                row[f"proto_sim_{f:.2f}"] = s
            for f, s in zip(fractions, result["rand_sim"]):
                row[f"rand_sim_{f:.2f}"] = s
            rows_buffer.append(row)

        pd.DataFrame(rows_buffer).to_csv(detail_csv, mode="a", header=False, index=False)
        rows_buffer.clear()
        print(f"  appended rows for image {img_path.name}")

    return pd.read_csv(detail_csv)


def summarize_per_prototype(detail_df):
    """One row per prototype: mean AUCs and mean curves."""
    g = detail_df.groupby("prototype", as_index=False).agg(
        mean_auc_prp=("auc_prp", "mean"),
        mean_auc_proto=("auc_proto", "mean"),
        mean_auc_rand=("auc_rand", "mean"),
        mean_delta_prp_proto=("auc_delta_prp_minus_proto", "mean"),
        n=("image", "count"),
    )
    # Badness: random curve nearly as good as PRP → high rand, small gap prp-rand
    g["bad_score"] = g["mean_auc_rand"] / (g["mean_auc_prp"].abs() + 1e-6)
    g["prp_minus_rand"] = g["mean_auc_prp"] - g["mean_auc_rand"]
    # Goodness: PRP much better than coarse prototype map
    g["good_score"] = g["mean_delta_prp_proto"]
    return g.sort_values("prototype")


def mean_curves_for_prototype(detail_df, pno, fractions):
    """Mean PRP / proto / random similarity curves for one prototype across images."""
    sub = detail_df[detail_df["prototype"] == pno]
    if sub.empty:
        return None, None, None
    pr_cols = [f"prp_sim_{f:.2f}" for f in fractions]
    pt_cols = [f"proto_sim_{f:.2f}" for f in fractions]
    rn_cols = [f"rand_sim_{f:.2f}" for f in fractions]
    return (
        sub[pr_cols].mean().values,
        sub[pt_cols].mean().values,
        sub[rn_cols].mean().values,
    )


def plot_three_prototypes_panel(
    detail_df,
    prototype_ids,
    fractions,
    title_prefix,
    savepath,
):
    """Three vertically stacked subplots; each shows mean PRP vs Proto vs Random for one prototype."""
    pcts = [f * 100 for f in fractions]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, pno in zip(axes, prototype_ids):
        m_prp, m_pt, m_rn = mean_curves_for_prototype(detail_df, pno, fractions)
        if m_prp is None:
            continue
        ax.plot(pcts, m_prp, "o-", color="#d62728", label="PRP", linewidth=2, markersize=3)
        ax.plot(pcts, m_pt, "s-", color="#1f77b4", label="Prototype upsample", linewidth=2, markersize=3)
        ax.plot(pcts, m_rn, "^--", color="#7f7f7f", label="Random", linewidth=1.5, markersize=3)
        ax.set_ylabel("Similarity")
        ax.set_title(f"Prototype {pno}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Pixels revealed (%)")
    fig.suptitle(title_prefix, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savepath}")


def pick_bad_prototypes(summary_df, k=3):
    """
    Prototypes where random ordering achieves AUC close to PRP (not learning a specific spatial pattern).
    Highest bad_score = worst.
    """
    s = summary_df.sort_values("bad_score", ascending=False)
    return s.head(k)["prototype"].astype(int).tolist()


def pick_good_prototypes(summary_df, k=3):
    """Largest gap PRP vs coarse prototype map on average."""
    s = summary_df.sort_values("good_score", ascending=False)
    return s.head(k)["prototype"].astype(int).tolist()


def run_group(
    group_name,
    image_paths,
    ppnet,
    prp_model,
    network_params,
    device,
    args,
    fractions,
    seed_offset,
):
    num_p = ppnet.num_prototypes
    out_dir = Path(args.output_dir) / group_name
    detail_df = run_all_prototypes_for_images(
        image_paths,
        ppnet,
        prp_model,
        network_params.img_size,
        fractions,
        device,
        args.seed + seed_offset,
        group_name,
        out_dir,
        num_p,
    )

    summary_df = summarize_per_prototype(detail_df)
    summary_path = out_dir / "per_prototype_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    bad_ids = pick_bad_prototypes(summary_df, k=3)
    good_ids = pick_good_prototypes(summary_df, k=3)

    (out_dir / "selected_bad_prototypes.txt").write_text(
        " ".join(map(str, bad_ids)) + "\n", encoding="utf-8"
    )
    (out_dir / "selected_good_prototypes.txt").write_text(
        " ".join(map(str, good_ids)) + "\n", encoding="utf-8"
    )

    plot_three_prototypes_panel(
        detail_df,
        bad_ids,
        fractions,
        f"{group_name}: 3 prototypes with weakest PRP-vs-random separation (high random AUC)",
        str(out_dir / "bad_prototypes_relevance_ordering.png"),
    )
    plot_three_prototypes_panel(
        detail_df,
        good_ids,
        fractions,
        f"{group_name}: 3 prototypes with strongest PRP vs InsightR prototype map",
        str(out_dir / "good_prototypes_relevance_ordering.png"),
    )

    # Overall mean plot (like original script) for reference
    prp_cols = [f"prp_sim_{f:.2f}" for f in fractions]
    pt_cols = [f"proto_sim_{f:.2f}" for f in fractions]
    rn_cols = [f"rand_sim_{f:.2f}" for f in fractions]
    plot_mean_curves(
        fractions,
        detail_df[prp_cols].mean().values,
        detail_df[pt_cols].mean().values,
        detail_df[rn_cols].mean().values,
        detail_df[prp_cols].std().values,
        detail_df[pt_cols].std().values,
        detail_df[rn_cols].std().values,
        ylabel="Prototype Similarity (higher = better)",
        title=f"Relevance ordering — all prototypes pooled — {group_name}",
        savepath=str(out_dir / "relevance_ordering_all_prototypes_pooled.png"),
    )
    pooled = out_dir / "relevance_ordering_all_prototypes_pooled.png"
    plot_mean_curves(
        fractions,
        detail_df[prp_cols].mean().values,
        detail_df[pt_cols].mean().values,
        detail_df[rn_cols].mean().values,
        None,
        None,
        None,
        ylabel="Prototype Similarity (higher = better)",
        title=f"Relevance ordering — all prototypes pooled — {group_name}",
        savepath=str(pooled.with_name(pooled.stem + "_mean_only" + pooled.suffix)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="All-prototype relevance ordering; mine good vs bad prototypes"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--param_jsonpath", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument(
        "--artifact_labels_csv",
        required=True,
        help="Columns: image_name, artifact_label. 0=clean (no artifact), 1=has artifact",
    )
    parser.add_argument("--output_dir", default="prototype_mining_results")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Random sample size per group (with / without artifact)")
    parser.add_argument("--num_fractions", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))

    ppnet = load_ppnet(args.model_path, network_params, device)
    ppnet_for_prp = load_ppnet(args.model_path, network_params, device)
    ppnet_for_prp.base_architecture = network_params.base_architecture
    prp_model = PRPCanonizedModel(ppnet_for_prp).to(device)

    print(f"num_prototypes={ppnet.num_prototypes}")
    print("PRP maps: insight_prp.generate_prp_image (via relevance_ordering_general.process_single_image)")

    with_paths, clean_paths = load_paths_by_artifact_labels(
        args.artifact_labels_csv, args.image_dir
    )
    rng = np.random.default_rng(args.seed)
    with_sample = sample_random(with_paths, args.num_images, rng)
    clean_sample = sample_random(clean_paths, args.num_images, rng)

    print(f"Sampled {len(with_sample)} images WITH artifact (label 1), "
          f"{len(clean_sample)} WITHOUT artifact (label 0).")

    fractions = np.linspace(0.0, 1.0, args.num_fractions).tolist()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    (Path(args.output_dir) / "README_labels.txt").write_text(
        "artifact_label: 0 = clean (no artifact), 1 = image has artifact\n",
        encoding="utf-8",
    )

    if with_sample:
        run_group(
            "with_artifact",
            with_sample,
            ppnet,
            prp_model,
            network_params,
            device,
            args,
            fractions,
            0,
        )
    else:
        print("No images with artifact — skip with_artifact group.")

    if clean_sample:
        run_group(
            "without_artifact",
            clean_sample,
            ppnet,
            prp_model,
            network_params,
            device,
            args,
            fractions,
            500_000,
        )
    else:
        print("No clean images — skip without_artifact group.")

    print(f"\nDone. Output under: {args.output_dir}")


if __name__ == "__main__":
    main()
