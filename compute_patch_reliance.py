# ---------------------------------------------------------
# compute_patch_reliance.py
#
# End-to-end Patch Reliance for INSightR-Net using your codebase:
# - Loads PPNet checkpoint and a dataloader from a parent data folder
#   that contains `train/` and `test/` subfolders.
# - Extracts prototype similarity maps (B, P, H_lat, W_lat) ~ (B, P, 9, 9)
# - Computes Patch Reliance per image (IoU + Soft variants)
# - Saves a CSV and plots for patched vs clean subsets
#
# IMPORTANT: All inputs live in the CONFIG block below.
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

training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/each_class_30_Oct/class2/exp1/gpupro/Fold0_DR_ec_30_Oct_2/'
patch_reliance_output_folder = 'tsne_pca_with_artifact1/train/patch_reliance_outputs4'

CONFIG = {
    # Parent folder containing subfolders such as 'train/' and 'test/' with images.
    # This should match what your datamodule expects (datamodule.py / dataset_DiabeticRet.py).
    "DATA_ROOT": "/sc/home/akshay.gudi/data_store/DR/each_class_yellow/class2",  # <-- EDIT

    # Training params JSON (same one you used to train).
    "PARAMS_JSON": "config/params_example_ordinal.json",  # <-- EDIT if needed

    # Checkpoint to evaluate (same one as your PCA workflow).
    "CHECKPOINT_PATH": '/sc/home/akshay.gudi/code/CleverHansRegression/each_class_30_Oct/class2/exp1/gpupro/Fold0_DR_ec_30_Oct_2/' + 'saved_models/Epoch_50_after_protopushing.pth',

    # Where to save outputs (CSV + plots)
    "OUT_DIR": training_root_folder + patch_reliance_output_folder,

    # Which split(s) to evaluate: choose any subset of {"train","val","test"}
    "SPLITS": ["test"],

    # Fixed patch rectangle for patched images (input size = 540x540).
    # You said: top-left at (x=380, y=380), size = 150x150.
    "PATCH_RECT": (380, 380, 150, 150),  # (x0, y0, w, h)

    # Which severity class has the patch? (0..4).
    # If you also provide PATCH_FILENAMES_CSV (below), that will take precedence per image.
    "PATCHED_CLASS_ID": 2,   # e.g., patch added to 80% of class 2

    # (Optional) CSV listing which images are patched (strongly recommended if only 80% are patched).
    # CSV must contain a column 'image_name'. If None, we will treat ALL images in PATCHED_CLASS_ID as patched.
    "PATCH_FILENAMES_CSV": "/sc/home/akshay.gudi/data_store/DR/each_class_yellow/class2/data_details_class2/train_yellow_patch.csv",  # e.g., "/path/to/train_yellow_patch.csv" or None

    # IoU mode thresholds
    "IOU_BINARIZE_THR": 0.90,  # normalize per-prototype to [0,1], then A>thr => active region
    "IOU_MIN_OVERLAP": 0.30,   # IoU >= this -> prototype is "patch-relevant"
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

def make_patch_mask(h: int, w: int, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Build a binary patch mask (h,w) with 1s inside the rectangle.
    rect = (x0, y0, rw, rh) in input-image coordinates (e.g., 540x540).
    """
    x0, y0, rw, rh = rect
    M = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
    M[y0:y1, x0:x1] = 1
    return M

def load_patched_names_from_csv(csv_path: Optional[str]) -> Optional[set]:
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)

    num_rows = df.shape[0]
    print("============================= patched image names ==============================")
    print("Number of rows:", num_rows)

    if "image_name" not in df.columns:
        raise ValueError("PATCH_FILENAMES_CSV must have a column 'image_name'.")
    return set(df["image_name"].astype(str).tolist())

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_violin_hist(pr_clean, pr_patched, title, outfile_prefix):
    # Violin
    plt.figure(figsize=(6,4))
    data = [pr_clean, pr_patched]
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks([1,2], ["Clean", "Patched"])
    plt.ylabel("Patch Reliance")
    plt.title(title + " (Violin)")
    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}_violin.png", dpi=200)
    plt.close()

    # Overlaid hist
    plt.figure(figsize=(6,4))
    bins = np.linspace(0,1,25)
    plt.hist(pr_clean,   bins=bins, alpha=0.6, label="Clean")
    plt.hist(pr_patched, bins=bins, alpha=0.6, label="Patched")
    plt.xlabel("Patch Reliance")
    plt.ylabel("Count")
    plt.title(title + " (Histogram)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}_hist.png", dpi=200)
    plt.close()

def scatter_pr_vs_pred(pr_vals, y_pred, is_patched, outfile_prefix, title):
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred[~is_patched], pr_vals[~is_patched], s=16, alpha=0.6, label="Clean")
    plt.scatter(y_pred[ is_patched], pr_vals[ is_patched], s=16, alpha=0.6, label="Patched")
    plt.xlabel("Prediction (ŷ)")
    plt.ylabel("Patch Reliance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}_scatter.png", dpi=200)
    plt.close()

# -----------------------------
# Core extraction from your PPNet
# -----------------------------
@torch.no_grad()
def extract_similarity_and_weights(ppnet, images: torch.Tensor) -> Dict[str, Any]:
    """
    Extract from YOUR PPNet:
      z     = ppnet.conv_features(images)           -> (B, C, H_lat, W_lat)
      dists = ppnet._l2_convolution(z)              -> (B, P, H_lat, W_lat)
      sims  = ppnet.activation_function(dists)      -> (B, P, H_lat, W_lat)

      s[b,j] = max_{u,v} sims[b,j,u,v]              -> (B, P)   (image-level similarity per prototype)
      r[j]   = |last_layer.weight[0,j]|             -> (P,)     (prototype→output weight magnitude)
      w[b,j] = s[b,j] * r[j]                        -> (B, P)   (per-image prototype contribution magnitude)
    """
    ppnet.eval()

    # 1) Latent features after backbone+add_on_layers
    z = ppnet.conv_features(images)                  # (B, C, H_lat, W_lat)

    # 2) Per-prototype L2 distance maps
    dists = ppnet._l2_convolution(z)                 # (B, P, H_lat, W_lat)

    # 3) Convert distance to similarity (paper-defined activation)
    sims = ppnet.distance_2_similarity(dists)          # (B, P, H_lat, W_lat)

    B, P, H_lat, W_lat = sims.shape
    _, _, H_img, W_img = images.shape

    # 4) Pool to image-level similarity per prototype
    s = sims.amax(dim=(-2, -1))                      # (B, P)

    # 5) Prototype→output regression weights (Linear(P,1))
    r = ppnet.last_layer.weight.squeeze(0).abs()     # (P,)

    # 6) Contribution magnitudes
    w = s * r                                        # (B, P) via broadcasting

    return {
        "sims": sims,   # (B,P,H_lat,W_lat)
        "s": s,         # (B,P)
        "r": r,         # (P,)
        "w": w,         # (B,P)
        "H_lat": H_lat, "W_lat": W_lat,
        "H_img": H_img, "W_img": W_img,
    }

# -----------------------------
# Patch Reliance (IoU & Soft)
# -----------------------------
def _upsample_similarity_maps(sim_maps_9x9: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    t = sim_maps_9x9.unsqueeze(1).float()                # (P,1,H_lat,W_lat)
    t_up = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t_up.squeeze(1)                                # (P,H_img,W_img)

def _normalize_per_prototype(A: torch.Tensor, eps=1e-8) -> torch.Tensor:
    P = A.shape[0]
    A_flat = A.view(P, -1)
    mins = A_flat.min(dim=1, keepdim=True).values
    maxs = A_flat.max(dim=1, keepdim=True).values
    norm = (A_flat - mins) / (maxs - mins + eps)
    return norm.view_as(A)

def patch_reliance_iou_for_one(sim_maps_9x9: torch.Tensor,
                               w_one: torch.Tensor,
                               patch_mask: torch.Tensor,
                               thr: float,
                               iou_tau: float) -> float:
    """
    IoU PR:
      1) Upsample (P,9,9)->(P,H,W)
      2) Normalize per-prototype to [0,1]
      3) Binarize > thr
      4) IoU with patch mask per prototype
      5) PR = sum w_j over IoU_j>=tau / sum w_j
    """
    P, H_lat, W_lat = sim_maps_9x9.shape
    H_img, W_img    = patch_mask.shape

    A_up   = _upsample_similarity_maps(sim_maps_9x9, H_img, W_img)  # (P,H,W)
    A_norm = _normalize_per_prototype(A_up)
    A_bin  = (A_norm > thr).to(torch.uint8)                         # (P,H,W)
    M      = (patch_mask > 0).to(torch.uint8)                       # (H,W)

    ious = []
    for j in range(P):
        inter = torch.logical_and(A_bin[j]==1, M==1).sum().item()
        union = torch.logical_or (A_bin[j]==1, M==1).sum().item()
        iou = (inter/union) if union>0 else 0.0
        ious.append(iou)
    ious = np.array(ious, dtype=np.float32)

    w_np = w_one.detach().cpu().numpy()                              # (P,)
    pr   = float(w_np[ious >= iou_tau].sum() / (w_np.sum() + 1e-12))
    return pr

def patch_reliance_soft_for_one(sim_maps_9x9: torch.Tensor,
                                w_one: torch.Tensor,
                                patch_mask: torch.Tensor) -> float:
    """
    Soft PR:
      For each prototype j:
        inside_frac_j = sum(A_up[j]*M) / sum(A_up[j])
      PR_soft = sum_j w_j * inside_frac_j / sum_j w_j
    """
    P, H_lat, W_lat = sim_maps_9x9.shape
    H_img, W_img    = patch_mask.shape

    A_up = _upsample_similarity_maps(sim_maps_9x9, H_img, W_img)     # (P,H,W)
    A_up = torch.clamp(A_up, min=0.0)
    M    = (patch_mask > 0).float()                                  # (H,W)

    A_in  = (A_up * M).flatten(1).sum(dim=1)                         # (P,)
    A_tot =  A_up      .flatten(1).sum(dim=1)                         # (P,)
    inside_frac = A_in / (A_tot + 1e-12)                              # (P,)

    w_nonneg = w_one.abs()
    pr_soft  = float((w_nonneg * inside_frac).sum() / (w_nonneg.sum() + 1e-12))
    return pr_soft

# -----------------------------
# Mask builder using CONFIG
# -----------------------------
def build_masks_and_flags(filenames: List[str],
                          labels_np: np.ndarray,
                          H_img: int, W_img: int,
                          patched_names_set: Optional[set],
                          rect: Tuple[int,int,int,int],
                          patched_class_id: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      masks:  (B,H,W) uint8 0/1
      flags:  (B,) bool  (True if image is considered 'patched')

    Priority:
      - If patched_names_set provided and filename is listed -> use rect, mark patched=True.
      - Else if patched_class_id provided and label==that class -> use rect, patched=True.
      - Else -> zero mask, patched=False.
    """
    masks, flags = [], []
    for nm, lbl in zip(filenames, labels_np):
        is_pat = False
        if patched_names_set is not None and str(nm) in patched_names_set:
            is_pat = True
        # elif (patched_class_id is not None) and (int(lbl) == int(patched_class_id)):
        #     # NOTE: this assumes *all* images of that class are patched.
        #     # If only 80% are patched, pass PATCH_FILENAMES_CSV to be precise.
        #     is_pat = True

        if is_pat:
            masks.append(make_patch_mask(H_img, W_img, rect))
        else:
            masks.append(np.zeros((H_img, W_img), dtype=np.uint8))
        flags.append(is_pat)

    return np.stack(masks, axis=0), np.array(flags, dtype=bool)

# -----------------------------
# Run over a loader and save CSV + plots
# -----------------------------
def run_over_loader(ppnet, loader, device, out_dir: str, split_name: str,
                    iou_thr: float, iou_tau: float,
                    rect: Tuple[int,int,int,int],
                    patched_names_set: Optional[set],
                    patched_class_id: Optional[int]):

    rows = []

    ppnet.eval()
    for batch in loader:
        # Your datamodule returns: images, labels, filenames
        images, labels, filenames = batch
        images = images.to(device, non_blocking=True)
        labels_np = labels.detach().cpu().numpy()

        # Extract similarities & contributions from PPNet
        with torch.no_grad():
            ex = extract_similarity_and_weights(ppnet, images)
            sims = ex["sims"]        # (B,P,H_lat,W_lat)
            w    = ex["w"]           # (B,P)
            Himg, Wimg = ex["H_img"], ex["W_img"]

            # Predictions (if your forward returns dict, adapt as needed)
            # y_pred = ppnet(images)
            raw = ppnet(images)
            y_pred_t = coerce_pred_to_tensor(raw)
            y_pred = y_pred_t.detach().cpu().numpy().reshape(-1)

            if isinstance(y_pred, dict) and "y_hat" in y_pred:
                y_pred = y_pred["y_hat"]
                y_pred = y_pred.detach().cpu().numpy().reshape(-1)  # (B,)

        # Build masks + is_patched flags per image
        masks_np, is_patched = build_masks_and_flags(
            filenames=filenames,
            labels_np=labels_np,
            H_img=Himg, W_img=Wimg,
            patched_names_set=patched_names_set,
            rect=rect,
            patched_class_id=patched_class_id
        )

        # Compute PR for each image in the batch
        PR_iou_list, PR_soft_list = [], []
        for b in range(images.shape[0]):
            sim_9x9 = sims[b]                              # (P,H_lat,W_lat), torch.Tensor
            w_one   = w[b]                                 # (P,), torch.Tensor
            mask_b  = torch.from_numpy(masks_np[b]).to(sim_9x9.device)  # (H,W)

            pr_iou  = patch_reliance_iou_for_one(sim_9x9, w_one, mask_b,
                                                 thr=iou_thr, iou_tau=iou_tau)
            pr_soft = patch_reliance_soft_for_one(sim_9x9, w_one, mask_b)

            PR_iou_list.append(pr_iou)
            PR_soft_list.append(pr_soft)

        # Append to CSV rows
        for i in range(images.shape[0]):
            rows.append({
                "image_name": str(filenames[i]),
                "severity": int(labels_np[i]),
                "is_patched": bool(is_patched[i]),
                "PR_iou": float(PR_iou_list[i]),
                "PR_soft": float(PR_soft_list[i]),
                "y_pred": float(y_pred[i]),
            })

    # Save CSV for this split
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"patch_reliance_{split_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{split_name}] Saved CSV -> {csv_path}")

    # Plots for this split
    pr_iou_clean   = df.loc[~df["is_patched"], "PR_iou"].to_numpy()
    pr_iou_patched = df.loc[ df["is_patched"], "PR_iou"].to_numpy()
    pr_soft_clean  = df.loc[~df["is_patched"], "PR_soft"].to_numpy()
    pr_soft_patched= df.loc[ df["is_patched"], "PR_soft"].to_numpy()

    plot_violin_hist(pr_iou_clean, pr_iou_patched,
                     title=f"Patch Reliance (IoU) — {split_name}",
                     outfile_prefix=os.path.join(out_dir, f"pr_iou_{split_name}"))
    plot_violin_hist(pr_soft_clean, pr_soft_patched,
                     title=f"Patch Reliance (Soft) — {split_name}",
                     outfile_prefix=os.path.join(out_dir, f"pr_soft_{split_name}"))

    # Scatter PR vs prediction
    y_pred = df["y_pred"].to_numpy()
    is_pat = df["is_patched"].to_numpy()
    scatter_pr_vs_pred(df["PR_iou"].to_numpy(),  y_pred, is_pat,
                       outfile_prefix=os.path.join(out_dir, f"pr_iou_vs_pred_{split_name}"),
                       title=f"PR (IoU) vs Prediction — {split_name}")
    scatter_pr_vs_pred(df["PR_soft"].to_numpy(), y_pred, is_pat,
                       outfile_prefix=os.path.join(out_dir, f"pr_soft_vs_pred_{split_name}"),
                       title=f"PR (Soft) vs Prediction — {split_name}")

    # Console summary
    def s(a):
        return f"mean={np.nanmean(a):.3f}, std={np.nanstd(a):.3f}, n={a.size}"
    print(f"\n[{split_name}] Summary:")
    print(" IoU  clean  :", s(pr_iou_clean))
    print(" IoU  patched:", s(pr_iou_patched))
    print(" SOFT clean  :", s(pr_soft_clean))
    print(" SOFT patched:", s(pr_soft_patched))
    print(" ΔPR_iou     :", float(np.nanmean(pr_iou_patched) - np.nanmean(pr_iou_clean)))
    print(" ΔPR_soft    :", float(np.nanmean(pr_soft_patched) - np.nanmean(pr_soft_clean)))


def coerce_pred_to_tensor(y):
    # Handles torch.Tensor, dict, tuple/list
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, dict):
        for k in ("y_hat", "pred", "logits", "output", "out"):
            if k in y and isinstance(y[k], torch.Tensor):
                return y[k]
        # fallback: first tensor value in dict
        for v in y.values():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("Dict output has no tensor-like value")
    if isinstance(y, (tuple, list)):
        for v in y:
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("Tuple/List output has no tensor-like element")
    raise TypeError(f"Unsupported output type: {type(y)}")

# -----------------------------
# Loader + model constructors
# -----------------------------
def load_ppnet_and_loaders_from_config(cfg: dict):
    # Load params JSON
    params_dict = load_json(cfg["PARAMS_JSON"])
    # Inject/override required fields for your datamodule
    args_dict = {
        "run_name": "PR_eval",
        "param_jsonpath": cfg["PARAMS_JSON"],
        "datapath": cfg["DATA_ROOT"],   # <-- parent folder with train/test subfolders
        "savepath": params_dict.get("savepath", "savedmodel"),
        "pretrained_path": params_dict.get("pretrained_path", ""),
    }
    params_exp = {**params_dict, **args_dict}
    params = Parameters.from_dict(params_exp)
    params.set_savepaths()

    # Build datamodule (same as your training/eval)
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

    # Load patched filenames if provided
    patched_names_set = load_patched_names_from_csv(cfg["PATCH_FILENAMES_CSV"])

    ppnet, loaders, device = load_ppnet_and_loaders_from_config(cfg)

    for split in cfg["SPLITS"]:
        if split not in loaders:
            print(f"[WARN] No loader for split '{split}'. Skipping.")
            continue

        run_over_loader(
            ppnet=ppnet,
            loader=loaders[split],
            device=device,
            out_dir=cfg["OUT_DIR"],
            split_name=split,
            iou_thr=cfg["IOU_BINARIZE_THR"],
            iou_tau=cfg["IOU_MIN_OVERLAP"],
            rect=cfg["PATCH_RECT"],
            patched_names_set=patched_names_set,
            patched_class_id=cfg["PATCHED_CLASS_ID"],
        )