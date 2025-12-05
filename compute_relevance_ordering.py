# ---------------------------------------------------------
# relevance_ordering.py
#
# Relevance Ordering tests for INSightR-Net:
#   1) Prototype-wise (concept-level): remove top-K prototypes (zero s_j)
#   2) Region-wise (spatial): Gaussian-noise top-relevance tiles
#
# Outputs:
#   - CSVs with per-image deletion curves
#   - Plots of mean deletion curves (per split)
#
# All key inputs are in CONFIG below.
# ---------------------------------------------------------

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===============================
# CONFIG — EDIT THESE IN ONE PLACE
# ===============================

training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/all_yellow_18_Nov/class3/exp1/gpupro/Fold0_DR_ay_18_Nov_1/'
relevance_ordering_output_folder = 'tsne_pca_with_artifact1/test/relevance_order_outputs2'

CONFIG = {
    # Parent folder used by your datamodule (contains subfolders like train/, val/, test/)
    "DATA_ROOT": "/sc/home/akshay.gudi/data_store/DR/all_yellow/class3",          # <-- EDIT

    # Training params JSON (the same you use for training/PCA)
    "PARAMS_JSON": "config/params_example_ordinal.json", # <-- EDIT

    # Checkpoint to evaluate (as in your PCA workflow)
    "CHECKPOINT_PATH": training_root_folder + 'saved_models/Epoch_50_after_protopushing.pth',  # <-- EDIT

    # Splits to evaluate (must be supported by your datamodule)
    "SPLITS": ["test"],

    # Output directory
    "OUT_DIR": training_root_folder + relevance_ordering_output_folder,

    # -------- Prototype-wise RO options --------
    # Remove up to this many prototypes (e.g., 50 if you want full curve)
    "PROTOTYPE_MAX_REMOVE": 20,

    # -------- Region-wise RO options --------
    "IMAGE_SIZE": (540, 540),      # (H_img, W_img) — used for upsampling relevance map
    "TILE_SIZE": 54,               # tile size for spatial relevance (30 -> 18x18 tiles on 540x540)
    # Fractions of area to perturb in deletion curve
    "AREA_FRACTIONS": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],

    # Gaussian noise options (region-wise)
    # If True, estimate mean/std from the image itself; else use fixed values below
    "NOISE_FROM_IMAGE_STATS": True,
    "GAUSS_MEAN": 0.5,             # used only if NOISE_FROM_IMAGE_STATS=False
    "GAUSS_STD": 0.2,              # used only if NOISE_FROM_IMAGE_STATS=False

    # Number of example images per split to save qualitative plots
    "N_EXAMPLES_TO_PLOT": 6,

    "CLASS_WITH_ARTIFACT": 3
}

# ===============================
# Project imports (your codebase)
# ===============================
from define_parameters import Parameters, NetworkParams
from helpers import load_json
from datamodule import MyDataModuleDiabRet
from insight_training import model as insight_model  # construct_PPNet

# -----------------------------
# Small I/O helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

# -----------------------------
# Core extraction from your PPNet
# -----------------------------
@torch.no_grad()
def extract_sims_s_r_w(ppnet, images: torch.Tensor) -> Dict[str, Any]:
    """
    From YOUR PPNet, extract (batch-wise):

    images: torch.Tensor, shape (B, 3, H_img, W_img)

    z     = ppnet.conv_features(images)            -> (B, C, H_lat, W_lat)
    dists = ppnet._l2_convolution(z)               -> (B, P, H_lat, W_lat)
    sims  = ppnet.activation_function(dists)       -> (B, P, H_lat, W_lat) (higher=more similar)

    s[b,j] = max_{u,v} sims[b,j,u,v]               -> (B, P)   (pooled similarity per prototype)
    r[j]   = last_layer.weight.squeeze(0)          -> (P,)     (SIGNED regression weights)
    w[b,j] = s[b,j] * |r[j]|                       -> (B, P)   (non-neg contribution magnitude)

    returns dict:
      sims:  (B, P, H_lat, W_lat)
      s:     (B, P)
      r:     (P,)
      w:     (B, P)
      H_lat, W_lat, H_img, W_img
    """
    ppnet.eval()

    # Backbone + add_on_layers outputs latent features
    z = ppnet.conv_features(images)             # (B, C, H_lat, W_lat)

    # Per-prototype L2 distances
    dists = ppnet._l2_convolution(z)            # (B, P, H_lat, W_lat)

    # Similarity maps after activation (paper-defined choice)
    sims = ppnet.distance_2_similarity(dists)     # (B, P, H_lat, W_lat)

    B, P, H_lat, W_lat = sims.shape
    _, _, H_img, W_img = images.shape

    # Max pool per prototype to get image-level similarity
    s = sims.amax(dim=(-2, -1))                 # (B, P)

    # Signed regression weights from final linear layer
    last = ppnet.last_layer                     # nn.Linear(P, 1)
    r = last.weight.squeeze(0)                  # (P,) signed
    # Contribution magnitudes (used for importance ranking)
    w = s * r.abs()                             # (B, P)

    return {
        "sims": sims, "s": s, "r": r, "w": w,
        "H_lat": H_lat, "W_lat": W_lat,
        "H_img": H_img, "W_img": W_img,
    }

# -----------------------------
# Prototype-wise Relevance Ordering
# -----------------------------
def proto_ro_deletion_curve_for_batch(
    s: torch.Tensor,      # (B, P)
    r: torch.Tensor,      # (P,)
    w: torch.Tensor,      # (B, P) = s * |r|
    bias: float,
    k_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build prototype-wise deletion curves for a batch.

    For each image b:
      - Rank prototypes by w[b] (descending)
      - Iteratively set s[b, top_idx] = 0 for k = 1..k_max
      - Recompute y_k = (s_masked[b] @ r) + bias

    Returns:
      y0:  (B,) baseline predictions
      Y:   (B, k_max+1) predictions after removing 0..k_max prototypes
    """
    B, P = s.shape
    k_max = min(k_max, P)

    # Baseline predictions y0 = (s @ r) + b
    with torch.no_grad():
        y0_t = (s @ r) + bias          # (B,)
    y0 = y0_t.detach().cpu().numpy()   # (B,)

    # Prepare output: include k=0 baseline
    Y = np.zeros((B, k_max + 1), dtype=np.float32)
    Y[:, 0] = y0

    for b in range(B):
        # Sort prototype indices by contribution magnitude w[b]
        order = torch.argsort(w[b], descending=True)  # (P,)
        s_masked = s[b].clone()                       # (P,)

        for k in range(1, k_max + 1):
            s_masked[order[k-1]] = 0.0
            yk = float((s_masked @ r) + bias)
            Y[b, k] = yk

    return y0, Y

# -----------------------------
# Region-wise Relevance Ordering (Gaussian noise)
# -----------------------------
def build_relevance_map(
    sims_b: torch.Tensor,   # (P, H_lat, W_lat)
    w_b: torch.Tensor,      # (P,)
    out_h: int, out_w: int
) -> torch.Tensor:
    """
    Build a single-image relevance map R_img (H_img, W_img) as a
    weighted sum of prototype similarity maps (upsampled to image size).

      R_9x9  = sum_j w_j * sims_b[j]        -> (H_lat, W_lat)
      R_img  = bilinear_upsample(R_9x9)     -> (H_img, W_img)

    Returns R_img (float tensor).
    """
    # (P, H_lat, W_lat) * (P,) -> sum over P -> (H_lat, W_lat)
    R_9x9 = (w_b[:, None, None] * sims_b).sum(dim=0)  # (H_lat, W_lat)

    # Upsample to image size for pixel-wise relevance
    R_img = F.interpolate(
        R_9x9.unsqueeze(0).unsqueeze(0),             # (1,1,H_lat,W_lat)
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)                           # (H_img, W_img)
    return R_img

def tile_ranking_from_relevance(
    R_img: torch.Tensor,   # (H_img, W_img)
    tile_size: int
) -> Tuple[np.ndarray, int, int]:
    """
    Split R_img into tiles and rank tiles by mean relevance.

    Returns:
      order: np.ndarray of tile indices (descending by relevance)
      n_ty:  number of tiles vertically
      n_tx:  number of tiles horizontally
    """
    H, W = R_img.shape
    n_ty = H // tile_size
    n_tx = W // tile_size

    tile_vals = []
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, x0 = ty * tile_size, tx * tile_size
            block = R_img[y0:y0+tile_size, x0:x0+tile_size]
            tile_vals.append(float(block.mean().item()))

    tile_vals = np.array(tile_vals, dtype=np.float32)        # (n_ty * n_tx,)
    order = np.argsort(tile_vals)[::-1]                      # highest relevance first
    return order, n_ty, n_tx

def gaussian_noise_like(tile_shape, mean, std, device):
    return torch.normal(mean=mean, std=std, size=tile_shape, device=device)

@torch.no_grad()
def region_ro_deletion_curve_for_image(
    ppnet,
    image_b: torch.Tensor,       # (3, H_img, W_img)
    sims_b: torch.Tensor,        # (P, H_lat, W_lat)
    w_b: torch.Tensor,           # (P,)
    s_b: torch.Tensor,           # (P,) (used to compute baseline via s_b @ r + b)
    r: torch.Tensor,             # (P,) signed weights
    bias: float,
    tile_size: int,
    area_fractions: List[float],
    noise_from_image_stats: bool = True,
    gauss_mean: float = 0.5,
    gauss_std: float = 0.2
) -> Tuple[float, List[float]]:
    """
    Region-wise deletion curve for ONE image.

    Steps:
      1) Build R_img (H_img,W_img) from sims_b (P,9,9) and w_b (P,)
      2) Rank tiles by R_img mean
      3) For each area fraction f:
           - Copy image, noise top-k tiles (Gaussian)
           - Compute new prediction via y = (s' @ r) + b (we use ppnet conv to be exact or reuse s_b)
         Here to be fast & faithful to s/r, we recompute via ppnet forward components:
           a) Re-extract s' from sims' (but that would require a forward)
           b) Simpler & consistent: just call model forward on perturbed image.
              If forward returns tuple/dict, we compute via s/r path.
      4) Return baseline y0 and list y_f for each f.
    """
    device = image_b.device
    H_img, W_img = image_b.shape[-2:]

    # ----- Baseline prediction (consistent with prototype pathway)
    # Use signed r and s_b from this image:
    y0 = float((s_b @ r) + bias)

    # ----- Build pixel-level relevance map
    R_img = build_relevance_map(sims_b, w_b, out_h=H_img, out_w=W_img)   # (H_img, W_img)

    # ----- Rank tiles by mean relevance
    order, n_ty, n_tx = tile_ranking_from_relevance(R_img, tile_size)
    n_tiles_total = n_ty * n_tx

    # ----- Prepare Gaussian noise stats
    if noise_from_image_stats:
        mu = float(image_b.mean().item())
        sd = float(image_b.std().item() + 1e-6)
    else:
        mu = gauss_mean
        sd = gauss_std

    # ----- Deletion: noise top tiles progressively
    y_vals = []
    for f in area_fractions:
        k_tiles = max(1, int(round(f * n_tiles_total)))
        idxs = order[:k_tiles]

        img_noisy = image_b.clone()  # (3, H, W)
        for idx in idxs:
            ty = idx // n_tx
            tx = idx %  n_tx
            y0_t, x0_t = ty * tile_size, tx * tile_size
            img_noisy[:, y0_t:y0_t+tile_size, x0_t:x0_t+tile_size] = gaussian_noise_like(
                (image_b.shape[0], tile_size, tile_size), mean=mu, std=sd, device=device
            )

        # Compute prediction on perturbed image (use s/r path for consistency and speed)
        # Recompute s' from sims' requires a forward; we'll do a small forward path:
        z = ppnet.conv_features(img_noisy.unsqueeze(0))               # (1, C, H_lat, W_lat)
        dists = ppnet._l2_convolution(z)                              # (1, P, H_lat, W_lat)
        sims_p = ppnet.distance_2_similarity(dists)                     # (1, P, H_lat, W_lat)
        s_p = sims_p.amax(dim=(-2, -1)).squeeze(0)                    # (P,)
        y_f = float((s_p @ r) + bias)                                 # scalar
        y_vals.append(y_f)

    return y0, y_vals

# -----------------------------
# Run over a loader and save results
# -----------------------------
def run_over_loader(ppnet, loader, device, cfg: dict, split_name: str):
    out_dir = os.path.join(cfg["OUT_DIR"], split_name)
    ensure_dir(out_dir)

    # Signed weights & bias for prediction via s/r path
    last = ppnet.last_layer
    r = last.weight.squeeze(0)                      # (P,) signed
    bias = float(last.bias.squeeze().item()) if last.bias is not None else 0.0

    # To collect per-image curves
    proto_rows = []
    region_rows = []

    example_count = 0
    examples_to_plot = []

    for batch in loader:
        # Your dataloader like PCA returns: images, labels, filenames
        images, labels, filenames = batch
        images = images.to(device, non_blocking=True)
        labels_np = labels.detach().cpu().numpy()

        with torch.no_grad():
            ex = extract_sims_s_r_w(ppnet, images)
            sims = ex["sims"]      # (B,P,H_lat,W_lat)
            s    = ex["s"]         # (B,P)
            w    = ex["w"]         # (B,P)
            H_img, W_img = ex["H_img"], ex["W_img"]

        B, P = s.shape
        k_max = min(cfg["PROTOTYPE_MAX_REMOVE"], P)

        # ---------- Prototype-wise RO (batch)
        y0_proto, Y_proto = proto_ro_deletion_curve_for_batch(
            s=s, r=r, w=w, bias=bias, k_max=k_max
        )  # y0_proto: (B,), Y_proto: (B,k_max+1)

        # Save rows
        for i in range(B):
            row = {
                "image_name": str(filenames[i]),
                "severity": int(labels_np[i]),
                "y0": float(y0_proto[i]),
            }
            # Store the whole curve as JSON list for compact CSV
            row.update({
                "curve_proto": str(list(map(float, Y_proto[i].tolist()))),
            })
            proto_rows.append(row)

        # ---------- Region-wise RO (per image)
        for i in range(B):
            image_b = images[i]                    # (3,H_img,W_img)
            sims_b  = sims[i]                      # (P,H_lat,W_lat)
            w_b     = w[i]                         # (P,)
            s_b     = s[i]                         # (P,)
            y0_reg, y_vals_reg = region_ro_deletion_curve_for_image(
                ppnet=ppnet,
                image_b=image_b,
                sims_b=sims_b,
                w_b=w_b,
                s_b=s_b,
                r=r,
                bias=bias,
                tile_size=cfg["TILE_SIZE"],
                area_fractions=cfg["AREA_FRACTIONS"],
                noise_from_image_stats=cfg["NOISE_FROM_IMAGE_STATS"],
                gauss_mean=cfg["GAUSS_MEAN"],
                gauss_std=cfg["GAUSS_STD"],
            )
            region_rows.append({
                "image_name": str(filenames[i]),
                "severity": int(labels_np[i]),
                "y0": float(y0_reg),
                "area_fracs": str([float(x) for x in cfg["AREA_FRACTIONS"]]),
                "curve_region": str(list(map(float, y_vals_reg))),
            })

            # Collect a few examples to plot
            if example_count < cfg["N_EXAMPLES_TO_PLOT"]:
                examples_to_plot.append((image_b.detach().cpu(), sims_b.detach().cpu(), w_b.detach().cpu(),
                                          y0_reg, y_vals_reg, str(filenames[i])))
                example_count += 1

    # Save per-image curves to CSV
    proto_df  = pd.DataFrame(proto_rows)
    region_df = pd.DataFrame(region_rows)
    save_csv(proto_df,  os.path.join(out_dir, f"prototype_ro_curves.csv"))
    save_csv(region_df, os.path.join(out_dir, f"region_ro_curves.csv"))

    # --------------------------------------------------------------------
    # GROUP RESULTS: separate artifact class vs clean classes
    # --------------------------------------------------------------------
    artifact_class = cfg.get("CLASS_WITH_ARTIFACT", None)
    if artifact_class is not None:
        # For prototype-wise deletion curves
        proto_df["is_artifact_class"] = proto_df["severity"] == artifact_class
        region_df["is_artifact_class"] = region_df["severity"] == artifact_class

        # Split curves
        art_proto = np.stack([np.array(eval(s)) for s in proto_df.loc[proto_df["is_artifact_class"], "curve_proto"].tolist()])
        clean_proto = np.stack([np.array(eval(s)) for s in proto_df.loc[~proto_df["is_artifact_class"], "curve_proto"].tolist()])
        art_y0 = art_proto[:, 0:1]; clean_y0 = clean_proto[:, 0:1]
        art_drop = art_y0 - art_proto
        clean_drop = clean_y0 - clean_proto
        mean_drop_art = art_drop.mean(axis=0)
        mean_drop_clean = clean_drop.mean(axis=0)
        x_proto = np.linspace(0, (art_proto.shape[1]-1)/art_proto.shape[1], art_proto.shape[1])

        # Plot mean deletion curves per group
        plt.figure(figsize=(6,4))
        plt.plot(x_proto, mean_drop_art, marker='o', label=f"Class {artifact_class} (artifact)")
        plt.plot(x_proto, mean_drop_clean, marker='o', label="Other classes (clean)")
        plt.xlabel("Fraction of prototypes removed")
        plt.ylabel("Mean drop in prediction")
        plt.title(f"Prototype-wise Deletion — {split_name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_proto_deletion_artifact_vs_clean_{split_name}.png"), dpi=200)
        plt.close()

        # Similarly for region-wise curves
        area_fracs = np.array(eval(region_df["area_fracs"].iloc[0]))
        art_reg = np.stack([np.array(eval(s)) for s in region_df.loc[region_df["is_artifact_class"], "curve_region"].tolist()])
        clean_reg = np.stack([np.array(eval(s)) for s in region_df.loc[~region_df["is_artifact_class"], "curve_region"].tolist()])
        art_y0 = region_df.loc[region_df["is_artifact_class"], "y0"].to_numpy().reshape(-1,1)
        clean_y0 = region_df.loc[~region_df["is_artifact_class"], "y0"].to_numpy().reshape(-1,1)
        art_drop = art_y0 - art_reg
        clean_drop = clean_y0 - clean_reg
        mean_drop_art = art_drop.mean(axis=0)
        mean_drop_clean = clean_drop.mean(axis=0)

        plt.figure(figsize=(6,4))
        plt.plot(area_fracs, mean_drop_art, marker='o', label=f"Class {artifact_class} (artifact)")
        plt.plot(area_fracs, mean_drop_clean, marker='o', label="Other classes (clean)")
        plt.xlabel("Fraction of area perturbed")
        plt.ylabel("Mean drop in prediction")
        plt.title(f"Region-wise Deletion (Gaussian noise) — {split_name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_region_deletion_artifact_vs_clean_{split_name}.png"), dpi=200)
        plt.close()

    # ---------- Aggregate & plot mean curves ----------
    # Prototype-wise mean drop curve
    curves_proto = np.stack([np.array(eval(s)) for s in proto_df["curve_proto"].tolist()], axis=0)  # (N, k_max+1)
    y0p = curves_proto[:, 0:1]              # (N,1)
    drops_proto = y0p - curves_proto        # (N, k_max+1)
    mean_drop_proto = drops_proto.mean(axis=0)        # (k_max+1,)
    x_proto = np.linspace(0, (curves_proto.shape[1]-1)/curves_proto.shape[1], curves_proto.shape[1])

    plt.figure(figsize=(6,4))
    plt.plot(x_proto, mean_drop_proto, marker='o')
    plt.xlabel("Fraction of prototypes removed (0..~1)")
    plt.ylabel("Mean drop in prediction (ŷ₀ − ŷ_k)")
    plt.title(f"Prototype-wise Deletion Curve — {split_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"mean_proto_deletion_{split_name}.png"), dpi=200)
    plt.close()

    # Region-wise mean drop curve
    area_fracs = np.array(eval(region_df["area_fracs"].iloc[0])) if len(region_df) > 0 else np.array([])
    curves_reg = np.stack([np.array(eval(s)) for s in region_df["curve_region"].tolist()], axis=0) if len(region_df)>0 else np.zeros((0, len(area_fracs)))
    y0r = region_df["y0"].to_numpy().reshape(-1,1) if len(region_df)>0 else np.zeros((0,1))
    drops_reg = y0r - curves_reg
    mean_drop_reg = drops_reg.mean(axis=0) if len(drops_reg)>0 else np.zeros_like(area_fracs)

    plt.figure(figsize=(6,4))
    plt.plot(area_fracs, mean_drop_reg, marker='o')
    plt.xlabel("Fraction of image area perturbed")
    plt.ylabel("Mean drop in prediction (ŷ₀ − ŷ_f)")
    plt.title(f"Region-wise Deletion Curve (Gaussian noise) — {split_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"mean_region_deletion_{split_name}.png"), dpi=200)
    plt.close()

    print(f"[{split_name}] Done. Curves saved to: {out_dir}")

# -----------------------------
# Loader + model constructors
# -----------------------------
def load_ppnet_and_loaders_from_config(cfg: dict):
    # Load params JSON
    params_dict = load_json(cfg["PARAMS_JSON"])
    # Inject/override required fields for your datamodule
    args_dict = {
        "run_name": "RO_eval",
        "param_jsonpath": cfg["PARAMS_JSON"],
        "datapath": cfg["DATA_ROOT"],   # parent folder with train/val/test inside
        "savepath": params_dict.get("savepath", "savedmodel"),
        "pretrained_path": params_dict.get("pretrained_path", ""),
    }
    params_exp = {**params_dict, **args_dict}
    params = Parameters.from_dict(params_exp)
    params.set_savepaths()

    # Build datamodule
    dm = MyDataModuleDiabRet(params)
    loaders = {}
    for split in cfg["SPLITS"]:
        if split == "train":
            loaders["train"] = dm.train_dataloader()
        elif split == "val":
            loaders["val"] = dm.val_dataloader()
        elif split == "test":
            loaders["test"] = dm.test_dataloader()

    # Construct PPNet
    network_params = NetworkParams()
    ppnet = insight_model.construct_PPNet(network_params=network_params)

    # Load checkpoint
    ckpt = torch.load(cfg["CHECKPOINT_PATH"], map_location="cpu")
    if "state_dict" in ckpt:
        ppnet.load_state_dict(ckpt["state_dict"])
    else:
        ppnet.load_state_dict(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppnet = ppnet.to(device)
    ppnet.eval()
    return ppnet, loaders, device

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cfg = CONFIG
    ensure_dir(cfg["OUT_DIR"])

    ppnet, loaders, device = load_ppnet_and_loaders_from_config(cfg)

    for split in cfg["SPLITS"]:
        if split not in loaders:
            print(f"[WARN] No loader for split '{split}'. Skipping.")
            continue
        run_over_loader(ppnet, loaders[split], device, cfg, split_name=split)