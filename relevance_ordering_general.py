"""
Relevance Ordering Test (Insertion variant) — PRP paper adaptation for INSightR-Net.

Implements the "MacDonald-style" relevance ordering test from:
  "This looks more like that" (Gautam et al., Pattern Recognition 2022, Figs 8-10)

The test measures how faithfully an explanation map (PRP or prototype heatmap)
identifies the pixels that matter most for prototype similarity.

Protocol:
  1. Start from a random baseline image (uniform noise in [0, 1]).
  2. Gradually REPLACE pixels in the baseline with real image pixels,
     in the order defined by the explanation map (most important first).
  3. After each step, run the model and record:
       - prototype similarity for the chosen prototype
       - regression output (continuous prediction)
  4. Compare curves: PRP ordering vs prototype-heatmap ordering vs random ordering.
     If PRP ordering makes similarity rise faster, PRP is more faithful.

Usage:
    # Process only images from a class CSV (e.g. all class-3 test images):
    python relevance_ordering_general.py \
        --model_path saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --image_dir /path/to/test/images \
        --image_list_csv /path/to/label_3.csv \
        --output_dir relevance_ordering_results/class3 \
        --topk_prototypes 3

    # Process ALL test images in a folder (default):
    python relevance_ordering_general.py \
        --model_path saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --image_dir /path/to/test/images \
        --output_dir relevance_ordering_results \
        --topk_prototypes 3

    # Process only a random subset of N images:
    python relevance_ordering_general.py \
        --model_path saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --image_dir /path/to/test/images \
        --output_dir relevance_ordering_results \
        --num_images 50

    # Single image with a specific prototype:
    python relevance_ordering_general.py \
        --model_path saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --image_path /path/to/single_image.jpeg \
        --output_dir relevance_ordering_results \
        --prototype_index 42

    # No test folder: use only the pushed image stored in the checkpoint (prototype_images):
    python relevance_ordering_general.py \
        --model_path saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --output_dir relevance_ordering_stored \
        --from_stored_prototypes \
        --prototype_index 42
"""

import argparse
import copy
import os
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from define_parameters import NetworkParams
from helpers import load_json
from insight_training.model import construct_PPNet
from insight_prp import PRPCanonizedModel, generate_prp_image


# ═══════════════════════════════════════════════════════════
#  Model loading (reused from main_generate_prp.py pattern)
# ═══════════════════════════════════════════════════════════

def load_ppnet(model_path, network_params, device):
    """Load PPNet weights from .pth or Lightning .ckpt."""
    checkpoint = torch.load(model_path, map_location=device)
    ppnet = construct_PPNet(network_params=network_params)
    ppnet = ppnet.to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        ppnet_sd = {
            k[len("ppnet."):]: v
            for k, v in state_dict.items()
            if k.startswith("ppnet.")
        }
        ppnet.load_state_dict(ppnet_sd, strict=True)
    else:
        ppnet.load_state_dict(checkpoint, strict=True)

    ppnet.eval()
    return ppnet


# ═══════════════════════════════════════════════════════════
#  Image loading (same pipeline as dataset_DiabeticRet.py)
# ═══════════════════════════════════════════════════════════

def load_image(image_path, img_size=540):
    """Load a fundus image exactly like the training dataloader does.

    Returns:
        tensor of shape (1, 3, H, W) in [0, 1], BGR channel order.
    """
    jpeg_im = cv2.imread(str(image_path))
    if jpeg_im is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    if jpeg_im.shape[0] != img_size or jpeg_im.shape[1] != img_size:
        jpeg_im = cv2.resize(jpeg_im, (img_size, img_size))
    norm = jpeg_im / 255.0
    # Dimension fix: cv2.imread gives (H, W, C); PyTorch expects (C, H, W).
    # The previous permute(2, 1, 0) yielded (C, W, H) and silently
    # transposed the spatial axes. Use (2, 0, 1) for HWC -> CHW.
    # .contiguous() ensures the memory layout is contiguous after permute,
    # which is required for .view() calls later in the insertion loop.
    return torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().contiguous()


def tensor_from_stored_prototype(ppnet, pno: int, img_size: int, device: torch.device):
    """
    Build input tensor from the pushed prototype image in the checkpoint (prototype_images[pno]).

    Same layout as main_generate_prp.py / prototype pushing. Raises if that slot is empty (zeros).
    """
    proto_img = ppnet.prototype_images[pno]
    if proto_img.max() == 0:
        raise ValueError(
            f"Prototype {pno} has no stored image in the checkpoint (prototype_images all zeros). "
            "Run prototype pushing so prototype_images are populated, or use external test images "
            "without --from_stored_prototypes."
        )
    arr = proto_img.detach().cpu().numpy()
    if arr.shape[0] != img_size or arr.shape[1] != img_size:
        arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    norm = arr.astype(np.float32) / 255.0
    # Dimension fix: prototype_images stored as (H, W, C); PyTorch expects
    # (C, H, W). Previous permute(2, 1, 0) gave (C, W, H) and silently
    # transposed spatial axes. Use (2, 0, 1) for HWC -> CHW.
    t = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().contiguous()
    return t.to(device)


# ═══════════════════════════════════════════════════════════
#  Explanation maps (the two maps we compare)
# ═══════════════════════════════════════════════════════════

def get_prototype_heatmap(ppnet, image_tensor, pno, device):
    """
    Build the ProtoPNet-style explanation: bilinear-upsample the activation
    map for prototype `pno` to full image size.

    This is what ProtoPNet uses for visualization — coarse, model-agnostic.

    Returns:
        2D numpy array (H, W) with per-pixel importance (higher = more similar).
    """
    ppnet.eval()
    img = image_tensor.to(device)
    H, W = img.shape[2], img.shape[3]

    with torch.no_grad():
        # distances shape: (1, num_prototypes, H_lat, W_lat)
        distances, _ = ppnet.prototype_distances(img)
        # Convert distances to similarity — higher = more similar
        sim_maps = ppnet.distance_2_similarity(distances)   # (1, P, H_lat, W_lat)
        # Pick the map for prototype pno
        proto_map = sim_maps[0, pno, :, :]                  # (H_lat, W_lat)

    # Bilinear upsample to input resolution — this is the "naive" ProtoPNet method
    proto_map_4d = proto_map.unsqueeze(0).unsqueeze(0)       # (1, 1, H_lat, W_lat)
    upsampled = F.interpolate(proto_map_4d, size=(H, W), mode="bilinear", align_corners=False)
    return upsampled.squeeze().cpu().numpy()


def get_prp_heatmap(prp_model, image_tensor, pno, device):
    """
    Build the PRP explanation: backpropagate prototype similarity through
    the canonized model to get per-pixel relevance.

    This is the fine-grained, model-aware method from the PRP paper.

    Returns:
        2D numpy array (H, W) with per-pixel relevance (signed).
    """
    return generate_prp_image(image_tensor, pno, prp_model, device)


# ═══════════════════════════════════════════════════════════
#  Pixel ordering from an importance map
# ═══════════════════════════════════════════════════════════

def importance_order(heatmap):
    """
    Sort pixel positions by importance (most important first).

    For prototype heatmap (all positive): sort descending by value.
    For PRP heatmap (signed): sort descending by absolute value,
      so both strong positive and strong negative relevance rank high.

    Args:
        heatmap: 2D numpy array (H, W).

    Returns:
        1D array of flat indices sorted most-important-first.
    """
    flat = np.abs(heatmap).ravel()
    return np.argsort(flat)[::-1].copy()


# ═══════════════════════════════════════════════════════════
#  Model forward: extract similarity + regression output
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def get_similarity_and_prediction(ppnet, image_tensor, pno, device):
    """
    Run a standard forward pass (no LRP) and return:
      - similarity: pooled prototype similarity s[pno]  (scalar)
      - prediction: continuous regression output         (scalar)

    These are the two quantities we track as pixels are inserted.
    """
    ppnet.eval()
    img = image_tensor.to(device)

    # forward returns: (logits, min_distances, prototype_activations)
    logits, min_distances, prototype_activations = ppnet(img, return_convs=False)

    similarity = prototype_activations[0, pno].item()
    prediction = logits[0, 0].item()
    return similarity, prediction


# ═══════════════════════════════════════════════════════════
#  Core: insertion curve for one image + one prototype
# ═══════════════════════════════════════════════════════════

def run_insertion_curve(
    ppnet,
    real_image,       # (1, 3, H, W) tensor
    pno,              # prototype index
    pixel_order,      # 1D array of flat indices (most important first)
    fractions,        # list of floats in [0, 1], e.g. [0, 0.05, 0.10, ...]
    device,
    seed=0,
):
    """
    Run the insertion test for one ordering.

    Steps:
      1. Create a random baseline image (same shape as real_image).
      2. For each fraction p in `fractions`:
           - Copy the top-p% of pixels (by `pixel_order`) from the real image
             into the baseline.
           - Run the model and record similarity + prediction.

    Args:
        ppnet:       Non-canonized PPNet model (for standard forward).
        real_image:  (1, 3, H, W) tensor in [0, 1].
        pno:         Prototype index.
        pixel_order: 1D flat indices sorted most-important-first.
        fractions:   List of fractions [0.0, 0.05, ..., 1.0].
        device:      Torch device.
        seed:        Random seed for the baseline image.

    Returns:
        similarities: list of floats, one per fraction.
        predictions:  list of floats, one per fraction.
    """
    rng = np.random.default_rng(seed)

    # Baseline: uniform random in [0, 1], same shape as real image
    C, H, W = real_image.shape[1], real_image.shape[2], real_image.shape[3]
    base_np = rng.uniform(0, 1, size=(C, H, W)).astype(np.float32)
    # .contiguous() ensures the baseline tensor supports .view() operations
    base = torch.from_numpy(base_np).unsqueeze(0).contiguous()  # (1, C, H, W)

    # Flatten spatial dims for easy indexing: (1, C, H*W)
    real_flat = real_image.view(1, C, -1)
    n_pixels = H * W

    similarities = []
    predictions = []

    for frac in fractions:
        # How many pixels to reveal at this fraction
        k = int(round(frac * n_pixels))

        # Start from the random baseline each time
        current = base.clone()
        current_flat = current.view(1, C, -1)

        if k > 0:
            # Pick the top-k pixel positions from the ordering
            indices = pixel_order[:k]
            # Replace ALL 3 channels at those positions with the real image
            current_flat[:, :, indices] = real_flat[:, :, indices]

        # Reshape back to (1, C, H, W)
        current = current_flat.view(1, C, H, W)

        # Run model forward and record metrics
        sim, pred = get_similarity_and_prediction(ppnet, current, pno, device)
        similarities.append(sim)
        predictions.append(pred)

    return similarities, predictions


# ═══════════════════════════════════════════════════════════
#  Process one image: run all three orderings
# ═══════════════════════════════════════════════════════════

def process_single_image(
    ppnet,
    prp_model,
    image_tensor,
    pno,
    fractions,
    device,
    seed=0,
):
    """
    For one image and one prototype, run the insertion test with three orderings:
      1. PRP ordering       (fine-grained, model-aware)
      2. Prototype ordering  (coarse, bilinear-upsampled activation)
      3. Random ordering     (baseline control)

    Returns:
        dict with keys 'prp_sim', 'proto_sim', 'rand_sim',
                        'prp_pred', 'proto_pred', 'rand_pred'
        each being a list of floats (one per fraction).
    """
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    n_pixels = H * W

    # --- Build the three pixel orderings ---

    # 1) PRP: fine-grained relevance → sort by |relevance|
    prp_hm = get_prp_heatmap(prp_model, image_tensor, pno, device)
    order_prp = importance_order(prp_hm)

    # 2) Prototype heatmap: upsampled activation → sort by activation value
    proto_hm = get_prototype_heatmap(ppnet, image_tensor, pno, device)
    order_proto = importance_order(proto_hm)

    # 3) Random: shuffled indices (control baseline)
    rng = np.random.default_rng(seed + 999)
    order_rand = rng.permutation(n_pixels)

    # --- Run insertion curves ---
    # (use the NON-canonized ppnet for standard forward)
    sim_prp, pred_prp = run_insertion_curve(
        ppnet, image_tensor, pno, order_prp, fractions, device, seed=seed
    )
    sim_proto, pred_proto = run_insertion_curve(
        ppnet, image_tensor, pno, order_proto, fractions, device, seed=seed
    )
    sim_rand, pred_rand = run_insertion_curve(
        ppnet, image_tensor, pno, order_rand, fractions, device, seed=seed
    )

    return {
        "prp_sim": sim_prp,
        "proto_sim": sim_proto,
        "rand_sim": sim_rand,
        "prp_pred": pred_prp,
        "proto_pred": pred_proto,
        "rand_pred": pred_rand,
    }


# ═══════════════════════════════════════════════════════════
#  Find the top-k most activated prototypes for an image
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def get_topk_prototypes(ppnet, image_tensor, k, device):
    """Return indices of the k prototypes with highest similarity to this image."""
    ppnet.eval()
    img = image_tensor.to(device)
    _, _, prototype_activations = ppnet(img, return_convs=False)
    sims = prototype_activations[0]  # (num_prototypes,)
    _, topk_idx = torch.topk(sims, k=min(k, sims.shape[0]))
    return topk_idx.cpu().tolist()


# ═══════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════

def plot_mean_curves(
    fractions,
    mean_prp,
    mean_proto,
    mean_rand,
    std_prp=None,
    std_proto=None,
    std_rand=None,
    ylabel="Prototype Similarity",
    title="Relevance Ordering Test (Insertion)",
    savepath="relevance_ordering.png",
    prototype_plot_label=None,
):
    """Plot mean insertion curves for the three orderings, optionally with ±1 std shading.

    Pass ``std_*=None`` (default) for mean lines only — no shaded uncertainty.

    If prototype_plot_label is an int, the figure title is prefixed with
    \"Prototype {label} —\" so curves are easy to match to an index.
    """
    if prototype_plot_label is not None:
        title = f"Prototype {prototype_plot_label} — {title}"
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [f * 100 for f in fractions]

    ax.plot(pcts, mean_prp, "o-", color="#d62728", label="PRP ordering", linewidth=2, markersize=4)
    ax.plot(pcts, mean_proto, "s-", color="#1f77b4", label="Prototype ordering", linewidth=2, markersize=4)
    ax.plot(pcts, mean_rand, "^--", color="#7f7f7f", label="Random ordering", linewidth=1.5, markersize=4)

    if std_prp is not None:
        ax.fill_between(pcts, mean_prp - std_prp, mean_prp + std_prp, color="#d62728", alpha=0.15)
    if std_proto is not None:
        ax.fill_between(pcts, mean_proto - std_proto, mean_proto + std_proto, color="#1f77b4", alpha=0.15)
    if std_rand is not None:
        ax.fill_between(pcts, mean_rand - std_rand, mean_rand + std_rand, color="#7f7f7f", alpha=0.10)

    ax.set_xlabel("Pixels revealed (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # Save PNG (as requested by savepath) plus an SVG sibling in the same folder.
    sp = Path(savepath)
    fig.savefig(str(sp), dpi=150)
    saved = [str(sp)]
    if sp.suffix.lower() != ".svg":
        svg_path = sp.with_suffix(".svg")
        fig.savefig(str(svg_path))
        saved.append(str(svg_path))
    plt.close(fig)
    print(f"Saved plot: {', '.join(saved)}")


def run_relevance_ordering_core(
    ppnet,
    prp_model,
    image_paths,
    out_dir: Path,
    fractions,
    img_size: int,
    device: torch.device,
    *,
    seed: int,
    fixed_prototype_index,
    topk_prototypes: int,
    prototype_plot_label,
    preloaded_tensors=None,
    preloaded_names=None,
    output_filename_suffix: str = "",
):
    """
    Run one relevance-ordering batch: all images, prototype selection either
    fixed (one index for every image) or top-k per image.

    prototype_plot_label: int or None — passed to plot titles (None = no prefix).

    If preloaded_tensors is set: list of (1,3,H,W) tensors already on/near device;
    preloaded_names parallel list for CSV rows. In that case image_paths is ignored,
    fixed_prototype_index must be set, and only that prototype is tested per tensor.

    output_filename_suffix: e.g. \"_with_artifact\" → files like
    relevance_ordering_per_image_with_artifact.csv and ..._similarity_with_artifact.png

    Always writes both shaded (±std) and ``*_mean_only.png`` figures for similarity
    and prediction (mean curves only).
    """
    use_preloaded = preloaded_tensors is not None
    if use_preloaded:
        if fixed_prototype_index is None:
            raise ValueError("preloaded_tensors requires fixed_prototype_index")
        if preloaded_names is None or len(preloaded_tensors) != len(preloaded_names):
            raise ValueError("preloaded_tensors and preloaded_names must be same length")
    else:
        if image_paths is None or len(image_paths) == 0:
            raise ValueError("image_paths required unless preloaded_tensors is set")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sfx = output_filename_suffix or ""
    detail_csv = out_dir / f"relevance_ordering_per_image{sfx}.csv"

    fraction_cols_prp = [f"prp_sim_{f:.2f}" for f in fractions]
    fraction_cols_proto = [f"proto_sim_{f:.2f}" for f in fractions]
    fraction_cols_rand = [f"rand_sim_{f:.2f}" for f in fractions]
    all_columns = (
        ["image", "prototype",
         "auc_prp", "auc_proto", "auc_rand",
         "auc_delta_prp_minus_proto", "prp_better"]
        + fraction_cols_prp + fraction_cols_proto + fraction_cols_rand
    )

    if not detail_csv.exists():
        pd.DataFrame(columns=all_columns).to_csv(detail_csv, index=False)
        print(f"Created per-image CSV: {detail_csv}")
    else:
        print(f"Appending to existing per-image CSV: {detail_csv}")

    all_prp_sim = []
    all_proto_sim = []
    all_rand_sim = []
    all_prp_pred = []
    all_proto_pred = []
    all_rand_pred = []

    n_iter = len(preloaded_tensors) if use_preloaded else len(image_paths)

    for img_idx in range(n_iter):
        if use_preloaded:
            name = preloaded_names[img_idx]
            print(f"\n[{img_idx+1}/{n_iter}] {name} (checkpoint prototype_images)")
            image_tensor = preloaded_tensors[img_idx]
            if image_tensor.device != device:
                image_tensor = image_tensor.to(device)
            proto_indices = [fixed_prototype_index]
        else:
            img_path = image_paths[img_idx]
            print(f"\n[{img_idx+1}/{len(image_paths)}] {img_path.name}")
            image_tensor = load_image(str(img_path), img_size=img_size)

            if fixed_prototype_index is not None:
                proto_indices = [fixed_prototype_index]
            else:
                proto_indices = get_topk_prototypes(
                    ppnet, image_tensor, topk_prototypes, device
                )
                print(f"  Top-{topk_prototypes} prototypes: {proto_indices}")

        rows_this_image = []

        for pno in proto_indices:
            print(f"  Prototype {pno}: computing orderings + insertion curves ...")
            result = process_single_image(
                ppnet, prp_model, image_tensor, pno, fractions, device,
                seed=seed + img_idx,
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

            row_image_name = preloaded_names[img_idx] if use_preloaded else img_path.name
            rows_this_image.append({
                "image": row_image_name,
                "prototype": pno,
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
        print(f"  Saved to CSV ({img_idx+1}/{n_iter} done)")

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
    print(f"\nPRP better than Prototype heatmap in {n_prp_better}/{n_total} "
          f"({100*n_prp_better/max(n_total,1):.1f}%) of (image, prototype) pairs.")
    print(f"Mean AUC  —  PRP: {detail_df['auc_prp'].mean():.4f}  |  "
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
    summary_csv = out_dir / f"relevance_ordering_summary{sfx}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    plot_mean_curves(
        fractions,
        mean_prp_sim, mean_proto_sim, mean_rand_sim,
        std_prp_sim, std_proto_sim, std_rand_sim,
        ylabel="Prototype Similarity (higher = better)",
        title="Relevance Ordering: Prototype Similarity",
        savepath=str(out_dir / f"relevance_ordering_similarity{sfx}.png"),
        prototype_plot_label=prototype_plot_label,
    )

    plot_mean_curves(
        fractions,
        mean_prp_pred, mean_proto_pred, mean_rand_pred,
        std_prp_pred, std_proto_pred, std_rand_pred,
        ylabel="Regression Prediction (continuous)",
        title="Relevance Ordering: Regression Output",
        savepath=str(out_dir / f"relevance_ordering_prediction{sfx}.png"),
        prototype_plot_label=prototype_plot_label,
    )

    sim_png = out_dir / f"relevance_ordering_similarity{sfx}.png"
    pred_png = out_dir / f"relevance_ordering_prediction{sfx}.png"
    plot_mean_curves(
        fractions,
        mean_prp_sim, mean_proto_sim, mean_rand_sim,
        None,
        None,
        None,
        ylabel="Prototype Similarity (higher = better)",
        title="Relevance Ordering: Prototype Similarity",
        savepath=str(sim_png.with_name(sim_png.stem + "_mean_only" + sim_png.suffix)),
        prototype_plot_label=prototype_plot_label,
    )
    plot_mean_curves(
        fractions,
        mean_prp_pred, mean_proto_pred, mean_rand_pred,
        None,
        None,
        None,
        ylabel="Regression Prediction (continuous)",
        title="Relevance Ordering: Regression Output",
        savepath=str(pred_png.with_name(pred_png.stem + "_mean_only" + pred_png.suffix)),
        prototype_plot_label=prototype_plot_label,
    )

    n_runs = len(all_prp_sim)
    print(f"\nDone. Averaged over {n_runs} (image, prototype) pairs.")
    print(f"Results in: {out_dir}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Relevance Ordering Test (Insertion) — PRP vs Prototype vs Random"
    )

    # Model / config
    parser.add_argument("--model_path", required=True,
                        help="Path to trained .pth or .ckpt model file")
    parser.add_argument("--param_jsonpath", required=True,
                        help="Path to parameter JSON (e.g. config/params_example_ordinal.json)")

    # Image input — provide one of these three
    parser.add_argument("--image_dir", default=None,
                        help="Folder with .jpeg test images (processes ALL files by default)")
    parser.add_argument("--image_path", default=None,
                        help="Path to a single test image")
    parser.add_argument("--image_list_csv", default=None,
                        help=(
                            "CSV file with an 'image_name' column listing image names WITHOUT "
                            "the .jpeg suffix (e.g. label_3.csv). Must be used together with "
                            "--image_dir so the script knows which folder to look in."
                        ))

    # Prototype selection (use one mode: --prototypes list, or --prototype_index, or top-k default)
    parser.add_argument("--prototype_index", type=int, default=None,
                        help="Use this single prototype index for all images (writes to --output_dir as-is)")
    parser.add_argument("--prototypes", type=int, nargs="+", default=None,
                        help=(
                            "Run the test for each listed prototype index. Results go under "
                            "<output_dir>/prototype_<idx>/ for each index. Mutually exclusive with "
                            "--prototype_index and incompatible with default top-k mode."
                        ))
    parser.add_argument("--topk_prototypes", type=int, default=3,
                        help="If neither --prototype_index nor --prototypes is set, use top-k per image (default: 3)")

    # Experiment settings
    parser.add_argument("--num_images", type=int, default=-1,
                        help="Number of images to sample from --image_dir (-1 = ALL images, default: -1)")
    parser.add_argument("--num_fractions", type=int, default=21,
                        help="Number of evenly spaced fraction steps from 0 to 1 (default: 21, i.e. 0%%, 5%%, 10%%, ...)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Output
    parser.add_argument("--output_dir", default="relevance_ordering_results",
                        help="Directory to save CSV and plots")

    parser.add_argument(
        "--from_stored_prototypes",
        action="store_true",
        help=(
            "Use each prototype's pushed image from the checkpoint (prototype_images) as the "
            "target for pixel insertion — no --image_dir, --image_path, or CSV. Requires "
            "--prototype_index or --prototypes. Run prototype pushing first so prototype_images "
            "are non-zero."
        ),
    )

    args = parser.parse_args()

    if args.prototype_index is not None and args.prototypes is not None:
        parser.error("Use only one of --prototype_index and --prototypes")

    has_external_images = (
        args.image_dir is not None
        or args.image_path is not None
        or args.image_list_csv is not None
    )

    if args.from_stored_prototypes:
        if has_external_images:
            parser.error(
                "Do not combine --from_stored_prototypes with --image_dir, --image_path, or --image_list_csv"
            )
        if args.prototype_index is None and args.prototypes is None:
            parser.error("--from_stored_prototypes requires --prototype_index or --prototypes")
    else:
        if args.image_dir is None and args.image_path is None:
            parser.error(
                "Provide --image_dir, --image_path, or --image_dir + --image_list_csv, "
                "or use --from_stored_prototypes"
            )
        if args.image_list_csv is not None and args.image_dir is None:
            parser.error("--image_list_csv requires --image_dir to know which folder to look in")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load parameters ──
    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))
    img_size = network_params.img_size
    print(f"Architecture: {network_params.base_architecture}, img_size={img_size}")

    # ── Load the PLAIN model (for forward pass / similarity measurement) ──
    ppnet = load_ppnet(args.model_path, network_params, device)
    ppnet = ppnet.to(device)

    # ── Build the CANONIZED model (for PRP heatmap generation) ──
    # We need a separate copy because canonization modifies the model in place
    ppnet_for_prp = load_ppnet(args.model_path, network_params, device)
    ppnet_for_prp.base_architecture = network_params.base_architecture
    prp_model = PRPCanonizedModel(ppnet_for_prp)
    prp_model = prp_model.to(device)
    print("Canonized model ready for PRP.")

    fractions = np.linspace(0.0, 1.0, args.num_fractions).tolist()
    num_proto = int(network_params.proto_shape[0])

    # ── No external test images: use pushed prototype_images[p] from checkpoint ──
    if args.from_stored_prototypes:

        def _run_stored_for_pno(pno: int, out: Path):
            tensor = tensor_from_stored_prototype(ppnet, pno, img_size, device)
            lab = f"stored_prototype_{pno}"
            print(f"\n========== {lab} → {out} ==========")
            run_relevance_ordering_core(
                ppnet,
                prp_model,
                None,
                out,
                fractions,
                img_size,
                device,
                seed=args.seed,
                fixed_prototype_index=pno,
                topk_prototypes=args.topk_prototypes,
                prototype_plot_label=pno,
                preloaded_tensors=[tensor],
                preloaded_names=[lab],
            )

        if args.prototypes is not None:
            plist = list(dict.fromkeys(args.prototypes))
            for p in plist:
                if p < 0 or p >= num_proto:
                    raise ValueError(
                        f"prototype index {p} out of range; valid 0..{num_proto - 1}"
                    )
            base_out = Path(args.output_dir)
            base_out.mkdir(parents=True, exist_ok=True)
            print(
                f"Using checkpoint prototype_images only (no test folder). "
                f"{len(plist)} run(s) under {base_out}/prototype_<idx>/"
            )
            for pno in plist:
                _run_stored_for_pno(pno, base_out / f"prototype_{pno}")
            return

        pno = args.prototype_index
        if pno < 0 or pno >= num_proto:
            raise ValueError(f"prototype_index must be in [0, {num_proto - 1}]; got {pno}")
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        print("Using checkpoint prototype_images only (no test folder).")
        _run_stored_for_pno(pno, out)
        return

    # ── Collect image paths (external test data) ──
    rng = np.random.default_rng(args.seed)

    if args.image_path:
        # Single image mode
        image_paths = [Path(args.image_path)]

    elif args.image_list_csv:
        # CSV-filtered mode: only process images listed in the CSV file.
        # The CSV must have an 'image_name' column with names WITHOUT .jpeg suffix.
        list_df = pd.read_csv(args.image_list_csv)
        if "image_name" not in list_df.columns:
            raise ValueError(
                f"--image_list_csv must have an 'image_name' column. "
                f"Found columns: {list(list_df.columns)}"
            )
        # Build full paths by appending .jpeg and joining with --image_dir
        image_dir = Path(args.image_dir)
        image_paths = []
        missing = []
        for name in list_df["image_name"]:
            p = image_dir / f"{name}.jpeg"
            if p.exists():
                image_paths.append(p)
            else:
                missing.append(str(p))
        if missing:
            print(f"WARNING: {len(missing)} image(s) listed in CSV not found on disk "
                  f"(first few: {missing[:5]})")
        if not image_paths:
            raise FileNotFoundError(
                f"None of the images from {args.image_list_csv} were found in {args.image_dir}"
            )
        print(f"Loaded {len(image_paths)} image path(s) from {args.image_list_csv}")
        # Optionally subsample further if --num_images is set
        if args.num_images > 0 and len(image_paths) > args.num_images:
            chosen = rng.choice(len(image_paths), size=args.num_images, replace=False)
            image_paths = [image_paths[i] for i in sorted(chosen)]

    else:
        # Full-folder mode: all .jpeg files in --image_dir
        all_images = sorted(glob(os.path.join(args.image_dir, "*.jpeg")))
        if not all_images:
            raise FileNotFoundError(f"No .jpeg files found in {args.image_dir}")
        # --num_images -1 (default) means use ALL images
        if args.num_images > 0 and len(all_images) > args.num_images:
            chosen = rng.choice(len(all_images), size=args.num_images, replace=False)
            image_paths = [Path(all_images[i]) for i in sorted(chosen)]
        else:
            image_paths = [Path(p) for p in all_images]

    print(f"Will process {len(image_paths)} image(s).")

    # ── Multiple explicit prototypes: one subfolder per index ──
    if args.prototypes is not None:
        plist = list(dict.fromkeys(args.prototypes))
        for p in plist:
            if p < 0 or p >= num_proto:
                raise ValueError(
                    f"prototype index {p} out of range; valid 0..{num_proto - 1} "
                    f"(proto_shape[0]={num_proto})"
                )
        base_out = Path(args.output_dir)
        base_out.mkdir(parents=True, exist_ok=True)
        print(f"Running {len(plist)} prototype(s); outputs under {base_out}/prototype_<idx>/")
        for pno in plist:
            sub = base_out / f"prototype_{pno}"
            print(f"\n========== Prototype {pno} → {sub} ==========")
            run_relevance_ordering_core(
                ppnet,
                prp_model,
                image_paths,
                sub,
                fractions,
                img_size,
                device,
                seed=args.seed,
                fixed_prototype_index=pno,
                topk_prototypes=args.topk_prototypes,
                prototype_plot_label=pno,
            )
        return

    # ── Single index, top-k, or legacy: one output_dir ──
    fixed_pi = args.prototype_index
    plot_lab = args.prototype_index if args.prototype_index is not None else None
    if fixed_pi is not None:
        if fixed_pi < 0 or fixed_pi >= num_proto:
            raise ValueError(
                f"prototype_index must be in [0, {num_proto - 1}]; got {fixed_pi}"
            )

    run_relevance_ordering_core(
        ppnet,
        prp_model,
        image_paths,
        Path(args.output_dir),
        fractions,
        img_size,
        device,
        seed=args.seed,
        fixed_prototype_index=fixed_pi,
        topk_prototypes=args.topk_prototypes,
        prototype_plot_label=plot_lab,
    )


if __name__ == "__main__":
    main()
