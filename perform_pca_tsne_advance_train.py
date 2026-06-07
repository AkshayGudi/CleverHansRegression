# pca_tsne_utils.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Optional, Tuple

import torch
from tqdm import tqdm
from pathlib import Path
from insight_training import model
from define_parameters import NetworkParams
from define_parameters import Parameters
from helpers import load_json
from datamodule import  MyDataModuleDiabRet
import argparse
import logging
from pytorch_lightning.utilities import rank_zero_info
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys

from scipy.stats import pearsonr
import seaborn as sns

def _ensure_dir(path: str) -> None:
 os.makedirs(path, exist_ok=True)

def _extract_features_and_prototypes(model_path, dataloader):
    print("Extracting features.........")
    # 1. Initialize model
    network_params = NetworkParams()   # Your network parameters
    ppnet_model = model.construct_PPNet(network_params=network_params)

    # 2. Load your trained model
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        ppnet_model.load_state_dict(checkpoint['state_dict'])
    else:
        ppnet_model.load_state_dict(checkpoint)

    ppnet_model.eval()
    ppnet_model = ppnet_model.cuda()  # If using GPU

    # 3. Extract features
    feature_maps, image_labels, image_names = [], [], []

    with torch.no_grad():
        i = 0
        for i, batch in enumerate(tqdm(dataloader)):
            # Expecting dataloader to return (images, labels, filenames)
            images, labels, filenames = batch
            images = images.cuda()            

            # Get feature maps using conv_features method
            features = ppnet_model.conv_features(images)   # Shape: [batch_size, 128, 9, 9]
            feature_maps.append(features.cpu().numpy())
            image_labels.extend(labels.cpu().numpy()) # Store labels
            image_names.extend(filenames)
            
            print(f"extracting for batch - {i+1}")                

    # 4. Combine
    all_feature_maps = np.concatenate(feature_maps, axis=0)
    prototype_vectors = ppnet_model.prototype_vectors.detach().cpu().numpy()
    prototype_labels = ppnet_model.proto_classes.cpu().numpy()

    image_labels = [x - 1 for x in image_labels]
    # 1. Clip values at 5.0
    clipped = np.clip(prototype_labels, None, 5.0)

    # 2. Convert to integers with ceiling (round up)
    ceil_vals = np.ceil(clipped).astype(int)

    # 3. Subtract 1 (final values range from 0 to 4)
    prototype_labels = ceil_vals - 1

    return (
        all_feature_maps,
        prototype_vectors,
        np.array(image_labels),
        prototype_labels,
        np.array(image_names),
    )


# ====================== Mark Special Images ======================
def _mark_special_images(labels, filenames, images_with_artifact, class_with_artifact):
    new_labels = []
    for lbl, fname in zip(labels, filenames):
        if fname in images_with_artifact and lbl == class_with_artifact:  # only override label=1 subset
            new_labels.append(5)  # special new class            
        else:
            new_labels.append(lbl)
    return np.array(new_labels)


def _reshape_features(feature_maps: np.ndarray) -> np.ndarray:
     """
     feature_maps: (N, C, H, W) -> (N, C*H*W)
     """
     if feature_maps.ndim != 4:
          raise ValueError(f"Expected (N,C,H,W); got {feature_maps.shape}")
     return feature_maps.reshape(feature_maps.shape[0], -1)

def _reshape_prototypes(proto_vecs: np.ndarray) -> np.ndarray:
     """
     proto_vecs: (P, C, 1, 1) -> (P, C)
     """
     if proto_vecs.ndim != 4 or proto_vecs.shape[2:] != (1, 1):
          raise ValueError(f"Expected (P,C,1,1); got {proto_vecs.shape}")
     return proto_vecs.reshape(proto_vecs.shape[0], -1)

def _choose_perplexity(n_samples: int, default: int = 30) -> int:
     """
     t-SNE requires 5 <= perplexity < (n_samples - 1) / 3
     """
     upper = max(5, (n_samples - 1) // 3)
     return max(5, min(default, upper))

# =============================================================================================================

def plot_pca_per_class(pca_result, labels, save_path, special_label):
    """
    Plot PCA separately for each class.
    For special_label (e.g., class 2), split into normal vs special subset.
    """

    unique_classes = np.unique(labels)

    for class_id in unique_classes:
        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(1, 2, 1)
        if class_id == special_label or class_id == 5:  # handle patched class separately
            mask_normal = labels == special_label
            mask_special = labels == 5  # patched subset
            ax1.scatter(pca_result[mask_normal, 0], pca_result[mask_normal, 1],
                        c='green', label=f'Class {special_label+1} (normal)')
            ax1.scatter(pca_result[mask_special, 0], pca_result[mask_special, 1],
                        c='yellow', label=f'Class {special_label+1} (yellow patch)')
        else:
            mask = labels == class_id
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1],
                        label=f'Class {class_id+1}')

        ax1.set_title("PCA 2D")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.legend()

        # 3D Scatter
        ax4 = fig.add_subplot(1, 2, 2, projection="3d")
        if class_id == special_label or class_id == 5:
            ax4.scatter(pca_result[mask_normal, 0], pca_result[mask_normal, 1],
                        pca_result[mask_normal, 2], c='green', label="Normal")
            ax4.scatter(pca_result[mask_special, 0], pca_result[mask_special, 1],
                        pca_result[mask_special, 2], c='yellow', label="Yellow patch")
        else:
            ax4.scatter(pca_result[mask, 0], pca_result[mask, 1],
                        pca_result[mask, 2], label=f'Class {class_id+1}')

        ax4.set_title("PCA 3D")
        ax4.set_xlabel("PC1")
        ax4.set_ylabel("PC2")
        ax4.set_zlabel("PC3")
        ax4.legend()

        plt.tight_layout()
        save_file = save_path / f"class_{class_id+1}_pca.png"
        plt.savefig(save_file)
        plt.close()
        print(f"Saved PCA for class {class_id+1} to {save_file}")

def plot_3D_tsne_per_class(tsne_result, labels, save_path, special_label):
    """
    Plot t-SNE separately for each class.
    For special_label (e.g., class 2), split into normal vs special subset.
    """
    unique_classes = np.unique(labels)
    for class_id in unique_classes:
        fig = plt.figure(figsize=(7, 6))    
        ax2 = fig.add_subplot(111, projection="3d")
        if class_id == special_label or class_id == 5:
            mask_normal = labels == special_label
            mask_special = labels == 5
            ax2.scatter(tsne_result[mask_normal, 0], tsne_result[mask_normal, 1],
                         tsne_result[mask_normal, 2], c='green', label="Normal")
            ax2.scatter(tsne_result[mask_special, 0], tsne_result[mask_special, 1],
                         tsne_result[mask_special, 2], c='yellow', label="Yellow patch")
        else:
            mask = labels == class_id
            ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                         tsne_result[mask, 2], label=f'Class {class_id+1}')

        ax2.set_title("t-SNE 3D")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.set_zlabel("t-SNE 3")
        ax2.legend()
        plt.tight_layout()
        save_file = save_path / f"class_{class_id+1}_3D_tsne.png"
        plt.savefig(save_file)
        plt.close()
        print(f"Saved t-SNE for class {class_id+1} to {save_file}")    

def plot_2D_tsne_per_class(tsne_result, labels, save_path, special_label):
    """
    Plot t-SNE separately for each class.
    For special_label (e.g., class 2), split into normal vs special subset.
    """
    unique_classes = np.unique(labels)

    print("Plotting TSNE..")
    for class_id in unique_classes:
        fig, ax1 = plt.subplots(figsize=(7, 6))
        if class_id == special_label or class_id == 5:
            mask_normal = labels == special_label
            mask_special = labels == 5
            ax1.scatter(tsne_result[mask_normal, 0], tsne_result[mask_normal, 1],
                        c='green', label=f'Class {special_label+1} (normal)')
            ax1.scatter(tsne_result[mask_special, 0], tsne_result[mask_special, 1],
                        c='yellow', label=f'Class {special_label+1} (yellow patch)')
        else:
            mask = labels == class_id
            ax1.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                        label=f'Class {class_id+1}')

        ax1.set_title("t-SNE 2D")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.legend()

        plt.tight_layout()
        save_file = save_path / f"class_{class_id+1}_2D_tsne.png"
        plt.savefig(save_file)
        plt.close()
        print(f"Saved t-SNE for class {class_id+1} to {save_file}")

# =============================================================================================================

def _scatter2d(ax, XY, labels, title: str, alpha: float = 0.8,
               class_names: dict | None = None,
               class_colors: dict | None = None):
    """
    Scatter with FIXED per-class colors + legend.
    If class_colors is None, falls back to Matplotlib default coloring.
    """
    labels = np.asarray(labels).astype(int)
    uniq = sorted(np.unique(labels))

    handles = []
    last_scatter = None
    if class_colors is not None:
        for l in uniq:
            idx = (labels == l)
            color = class_colors.get(l, 'gray')
            last_scatter = ax.scatter(XY[idx, 0], XY[idx, 1], s=8, alpha=alpha, color=color)
            name = class_names.get(l, str(l)) if class_names else str(l)
            handles.append(Line2D([0], [0], marker='o', linestyle='', color=color,
                                  label=name, markersize=6))
        ax.legend(handles=handles, title="Class", loc="best")
    else:
        # fallback: no fixed colors provided
        last_scatter = ax.scatter(XY[:, 0], XY[:, 1], c=labels, s=8, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    return last_scatter  # you can ignore this return if you don't use it


def _overlay_prototypes2d(ax, XYp, proto_labels=None):
     if XYp is None:
          return
     ax.scatter(XYp[:, 0], XYp[:, 1], s=80, marker="x", linewidths=2, 
                c=proto_labels if proto_labels is not None else "k", 
                alpha=1.0, label="prototypes")
     ax.legend(loc="best")

def _scatter3d(ax, XYZ, labels, title: str, alpha: float = 0.8,
               class_names: dict | None = None,
               class_colors: dict | None = None):
    """
    3D scatter with FIXED per-class colors + legend.
    """
    labels = np.asarray(labels).astype(int)
    uniq = sorted(np.unique(labels))

    handles = []
    last_scatter = None
    if class_colors is not None:
        for l in uniq:
            idx = (labels == l)
            color = class_colors.get(l, 'gray')
            last_scatter = ax.scatter(XYZ[idx, 0], XYZ[idx, 1], XYZ[idx, 2],
                                      s=8, alpha=alpha, color=color)
            name = class_names.get(l, str(l)) if class_names else str(l)
            handles.append(Line2D([0], [0], marker='o', linestyle='', color=color,
                                  label=name, markersize=6))
        ax.legend(handles=handles, title="Class", loc="best")
    else:
        last_scatter = ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2],
                                  c=labels, s=8, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    return last_scatter

def _overlay_prototypes3d(ax, XYZp, proto_labels=None):
     if XYZp is None:
          return
     ax.scatter(XYZp[:, 0], XYZp[:, 1], XYZp[:, 2], s=80, marker="x", linewidths=2, 
                c=proto_labels if proto_labels is not None else "k", 
                alpha=1.0)
     ax.legend(["data", "prototypes"], loc="best")

def pca_fit_transform( X: np.ndarray, scale: bool = True, n_components: int = 50, random_state: int = 42) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]:
     """
     Standardize X and reduce with PCA to n_components (or to min(n_components, rank)).
     Returns: (X_pca, pca, scaler)
     """
     scaler = None
     X_proc = X
     if scale:
          scaler = StandardScaler(with_mean=True, with_std=True)
          X_proc = scaler.fit_transform(X)

     # bound n_components by rank/samples/features to avoid warnings
     max_nc = min(X_proc.shape[0], X_proc.shape[1])
     nc = min(n_components, max_nc) if n_components is not None else max_nc

     pca = PCA(n_components=nc, random_state=random_state)
     X_pca = pca.fit_transform(X_proc)
     return X_pca, pca, scaler

def tsne_fit_transform(X_lowdim: np.ndarray, n_components: int = 3, default_perplexity: int = 30, n_iter: int = 1000, random_state: int = 42) -> np.ndarray:
     """
     Run t-SNE on a low-dimensional representation (e.g., PCA outputs).
     """
     perp = _choose_perplexity(X_lowdim.shape[0], default_perplexity)
     tsne = TSNE(n_components=n_components,  init="pca",   # good default; stable with PCA input
                 learning_rate="auto", perplexity=perp,  n_iter=n_iter,  random_state=random_state,)
     return tsne.fit_transform(X_lowdim)

def pca_tsne_visualizations(feature_maps: np.ndarray,   # (N,C,H,W)
                            labels: np.ndarray,     # (N,)
                            out_dir: str = "pca_tsne_outputs", tag: str = "train",
                            prototypes: Optional[np.ndarray] = None,  # (P,C,1,1)
                            proto_labels: Optional[np.ndarray] = None, # (P,)
                            pca_components: int = 50, tsne_points_cap: int = 5000000,  # subsample for t-SNE to keep it fast/clear
                            random_state: int = 42, special_label=4):
     """
     Produces and saves:
     - PCA 2D & 3D scatter (data + optional prototypes)
     - PCA explained variance plot
     - t-SNE 2D & 3D scatter (data + optional prototypes, fitted jointly)
     """
     _ensure_dir(out_dir)

     CLASS_NAMES = {
     0: "No DR",
     1: "Mild",
     2: "Moderate",
     3: "Severe",
     4: "Proliferative",
     5: "Artifact",  # special
     }

     # fixed colors for labels 0..5
     class_colors = {
     0: 'red',
     1: 'green',
     2: 'blue',
     3: 'purple',
     4: 'orange',
     5: 'yellow',  # special artifact class (lowercase name is safer)
     }

     # ---- Flatten features (N, C*H*W) ----
     X = _reshape_features(feature_maps)
     y = np.asarray(labels)

     # Optional prototype flattening for plotting
     P = None
     if prototypes is not None:
          P = _reshape_prototypes(np.asarray(prototypes))
          if proto_labels is not None:
               proto_labels = np.asarray(proto_labels)

     # ---- PCA (with standardization) ----
     X_pca, pca, scaler = pca_fit_transform(X, scale=True, n_components=pca_components, random_state=random_state)

    #  results = compute_group_means_matrix(X_pca=X_pca, labels=y, n_components=3, out_dir=out_dir, tag="train", class_with_artifact=special_label)

     # Transform prototypes into the same PCA space (if provided)
     P_pca = None
     if P is not None:
          P_proc = scaler.transform(P) if scaler is not None else P
          P_pca = pca.transform(P_proc)

     # ---- PCA plots ----
     # 2D
     fig, ax = plt.subplots(figsize=(7, 6))
     _scatter2d(ax, X_pca[:, :2], y, title=f"PCA (PC1 vs PC2) – {tag}", class_names=CLASS_NAMES, class_colors=class_colors)
     _overlay_prototypes2d(ax, P_pca[:, :2] if P_pca is not None else None, proto_labels)
     fig.tight_layout()
     fig.savefig(os.path.join(out_dir, f"{tag}_pca_2d.png"), dpi=200)
     plt.close(fig)

     plot_pca_per_class(X_pca, y, out_dir, special_label=special_label)

     # 3D
     if X_pca.shape[1] >= 3:
          from mpl_toolkits.mplot3d import Axes3D # noqa: F401
          fig = plt.figure(figsize=(7, 6))
          ax = fig.add_subplot(111, projection='3d')
          _scatter3d(ax, X_pca[:, :3], y, title=f"PCA (PC1–PC3) – {tag}", class_names=CLASS_NAMES, class_colors=class_colors)
          _overlay_prototypes3d(ax, P_pca[:, :3] if P_pca is not None else None, proto_labels)
          fig.tight_layout()
          fig.savefig(os.path.join(out_dir, f"{tag}_pca_3d.png"), dpi=200)
          plt.close(fig)

     # Explained variance
     evr = pca.explained_variance_ratio_
     fig, ax = plt.subplots(figsize=(7, 4))
     ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), marker='o', linewidth=1)
     ax.set_xlabel("Number of components")
     ax.set_ylabel("Cumulative explained variance")
     ax.set_title(f"PCA explained variance – {tag}")
     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
     fig.tight_layout()
     fig.savefig(os.path.join(out_dir, f"{tag}_pca_explained_variance.png"), dpi=200)
     plt.close(fig)

     # ---- t-SNE on PCA space (joint fit with prototypes if provided) ----
     # Prepare input for t-SNE and remember indices to split later.
     X_tsne_input = X_pca
     P_tsne_input = P_pca

     # Subsample for t-SNE if too many points (keeps plots legible and runtime reasonable)
     idx_data = np.arange(X_tsne_input.shape[0])
     if X_tsne_input.shape[0] > tsne_points_cap:
          rng = np.random.default_rng(random_state)
          idx_data = rng.choice(idx_data, size=tsne_points_cap, replace=False)
     X_tsne_sub = X_tsne_input[idx_data]
     y_sub = y[idx_data]

     # Concatenate prototypes (if any) so t-SNE positions are comparable
     if P_tsne_input is not None:
          X_joint = np.vstack([X_tsne_sub, P_tsne_input])
          n_data = X_tsne_sub.shape[0]
          n_protos = P_tsne_input.shape[0]
     else:
          X_joint = X_tsne_sub
          n_data = X_tsne_sub.shape[0]
          n_protos = 0

     # 2D t-SNE
     Z2 = tsne_fit_transform(X_joint, n_components=2, random_state=random_state)
     Z2_data = Z2[:n_data]
     Z2_proto = Z2[n_data:] if n_protos > 0 else None

     fig, ax = plt.subplots(figsize=(7, 6))
     _scatter2d(ax, Z2_data, y_sub, title=f"t-SNE (2D) – {tag}", class_names=CLASS_NAMES, class_colors=class_colors)
     _overlay_prototypes2d(ax, Z2_proto, proto_labels)
     fig.tight_layout()
     fig.savefig(os.path.join(out_dir, f"{tag}_tsne_2d.png"), dpi=200)
     plt.close(fig)

     plot_2D_tsne_per_class(Z2_data, y_sub, out_dir, special_label=special_label)

     # 3D t-SNE
     Z3 = tsne_fit_transform(X_joint, n_components=3, random_state=random_state)
     Z3_data = Z3[:n_data]
     Z3_proto = Z3[n_data:] if n_protos > 0 else None

     from mpl_toolkits.mplot3d import Axes3D # noqa: F401
     fig = plt.figure(figsize=(7, 6))
     ax = fig.add_subplot(111, projection='3d')
     _scatter3d(ax, Z3_data, y_sub, title=f"t-SNE (3D) – {tag}", class_names=CLASS_NAMES, class_colors=class_colors)
     _overlay_prototypes3d(ax, Z3_proto, proto_labels)
     fig.tight_layout()
     fig.savefig(os.path.join(out_dir, f"{tag}_tsne_3d.png"), dpi=200)
     plt.close(fig)

     plot_3D_tsne_per_class(Z3_data, y_sub, out_dir, special_label=special_label)

     # Optional: return embeddings for further analysis
     return {
     "X_pca": X_pca,   # full PCA data (no subsampling)
     "P_pca": P_pca,   # prototype PCA (if any)
     "tsne2_data": Z2_data,  # subsampled t-SNE
     "tsne2_proto": Z2_proto,
     "tsne3_data": Z3_data,
     "tsne3_proto": Z3_proto,
     "pca": pca,
     "scaler": scaler,
     "data_idx_used_for_tsne": idx_data
     }
# ===================================================================================================

def main(params, use_saved=True):

     # Path of csv file which has names of all images with yellow patch
     images_with_artifact_csv_path = "/sc/home/akshay.gudi/data_store/DR/bld_artifact/class5_v4/data_details_class5/train_yellow_patch.csv"

     # path where all training results including trained model is stored
     training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/bld_art_23_Feb/class5_v3/exp1/DR_25_Jan_2026_1/Fold0_DR_23_Feb_2026_3/'

     #Path to save the TSNE PCA results
     pca_tsne_sub_folder = 'tsne_pca_with_artifact1/train'

     # sub folder under training_root_folder where trained model is stored
     model_path_sub_folder = 'saved_models/Epoch_50_after_protopushing.pth'

     model_path = training_root_folder + model_path_sub_folder

     # Directory where features are stored or will be stored
     save_dir = Path(training_root_folder) / pca_tsne_sub_folder
     save_dir.mkdir(parents=True, exist_ok=True)

     # --- Decide whether to load or extract ---
     if use_saved:
          feature_file = save_dir / "feature_maps.npy"
          proto_file  = save_dir / "prototypes.npy"
          img_lbl_file = save_dir / "image_labels.npy"
          proto_lbl_file = save_dir / "prototype_labels.npy"
          img_name_file = save_dir / "image_names.npy"

          if all(f.exists() for f in [feature_file, proto_file, img_lbl_file, proto_lbl_file, img_name_file]):
               print(" Using previously saved feature maps and prototypes.")
               feature_maps = np.load(feature_file)
               prototypes = np.load(proto_file)
               image_labels = np.load(img_lbl_file)

# --- Diagnostic: check if image_labels.npy has problematic values ---
               print("[DEBUG load] image_labels.npy stats:")
               print(f"  shape={image_labels.shape}, dtype={image_labels.dtype}")
               print(f"  min={np.min(image_labels)}, max={np.max(image_labels)}")
               print(f"  unique values={np.unique(image_labels)}")
               if np.max(image_labels) > 5 or np.min(image_labels) < 0:
                    print("  >>> WARNING: image_labels.npy has values outside [0,5]. Likely cause of class_6/7 in outputs.")
               if np.min(image_labels) >= 1 and np.max(image_labels) <= 5:
                    print("  >>> Likely 1-based (1..5). Code expects 0-based (0..4); artifact=5.")
                # image_labels = np.asarray(image_labels)

               prototype_labels = np.load(proto_lbl_file)
               image_names = np.load(img_name_file)
          else:
               print(" Error: One or more saved .npy files not found. Please rerun with --use_saved False")
               sys.exit(1)
     else:
          print(" Extracting features and prototypes from model...") 
 
          # Create dataset
          dataset = MyDataModuleDiabRet(params)

          # Extract features and prototypes
          feature_maps, prototypes, image_labels, prototype_labels, image_names = _extract_features_and_prototypes(
          model_path,
          dataset.train_dataloader()
          )
        
          image_labels = np.asarray(image_labels)
          # --- Diagnostic: check if train dataloader produced values 6 or 7 (after x-1 in extract) ---
          print("[DEBUG extract] image_labels from train dataloader (after x-1 in _extract):")
          print(f"  shape={image_labels.shape}, dtype={image_labels.dtype}")
          print(f"  min={np.min(image_labels)}, max={np.max(image_labels)}")
          print(f"  unique values={np.unique(image_labels)}")
          if np.max(image_labels) > 5 or np.min(image_labels) < -1:
               print("  >>> WARNING: Train dataloader yielded labels that became >5 or <0 after x-1. Likely cause of class_6/7.")
          if np.max(image_labels) == 5 and np.min(image_labels) >= 0:
               print("  >>> Values in [0,5]; 5 may be from dataloader (6th class) or will be artifact. Expect at most class_6.")          

          np.save(save_dir / 'feature_maps.npy', feature_maps)
          np.save(save_dir / 'prototypes.npy', prototypes)
          np.save(save_dir / 'image_labels.npy', image_labels)
          np.save(save_dir / 'prototype_labels.npy', prototype_labels)
          np.save(save_dir / 'image_names.npy', image_names)


     # Load special CSV
     images_with_artifact = set(pd.read_csv(images_with_artifact_csv_path)["image_name"].astype(str).tolist())
     print(f"Loaded {len(images_with_artifact)} special images from {images_with_artifact_csv_path}")

    # class from 0 to 4
     class_with_artifact = 4

     # Mark special ones
     new_labels = _mark_special_images(image_labels, image_names, images_with_artifact, class_with_artifact=class_with_artifact)

     # --- Diagnostic: labels passed to PCA/t-SNE (will produce class_{k+1} for each unique k) ---
     print("[DEBUG] new_labels (after _mark_special_images) passed to visualizations:")
     print(f"  unique = {np.unique(new_labels)}; max = {np.max(new_labels)}")
     if np.max(new_labels) > 5:
          print("  >>> These will produce plot filenames class_6, class_7, ... . Fix by normalizing/clipping image_labels to 0..4 before _mark_special_images.")

     embeds = pca_tsne_visualizations(feature_maps=feature_maps, 
                                   labels=np.asarray(new_labels, dtype=int), # includes 5 now
                                   prototypes=None, proto_labels=None,      # still 0..4
                                   out_dir=save_dir, tag="train", pca_components=50, 
                                   tsne_points_cap=5000000,random_state=42, special_label=class_with_artifact)

     print("Completed processing it")


def get_flag_value(flag, default=True):
    """Extract value of a CLI flag like --use_saved=True/False"""
    for arg in sys.argv:
        if arg.startswith(flag + "="):
            val = arg.split("=", 1)[1].lower()
            if val in ["true", "1", "yes"]:
                return True
            elif val in ["false", "0", "no"]:
                return False
    return default

def _corr_from_mean_table(mean_df: pd.DataFrame, pc_cols: list[int | str]):
    """
    Given a mean table indexed by severity (rows), compute Pearson r between
    the severity index and each PC column (assumes the index is numeric and sorted).
    Returns a DataFrame with columns: Component, Correlation_with_Severity
    """
    mean_df = mean_df.sort_index()
    sev_levels = mean_df.index.values.astype(float)
    rows = []
    for c in pc_cols:
        col = mean_df[c].values
        if np.all(np.isnan(col)) or len(np.unique(sev_levels[~np.isnan(col)])) < 2:
            r = np.nan
        else:
            r, _ = pearsonr(sev_levels[~np.isnan(col)], col[~np.isnan(col)])
        rows.append({"Component": str(c), "Correlation_with_Severity": r})
    return pd.DataFrame(rows)


def compute_group_means_matrix(
    X_pca: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    out_dir: str | os.PathLike | None = None,
    tag: str = "train",
    class_with_artifact: int = 0,
):
    """
    Compute & save PCA means per severity and correlations in TWO modes:

    1) MERGED mode: map patched label (5) -> class_with_artifact (0..4)
       - Outputs:
         * {tag}_pca_mean_per_severity__merged.csv
         * {tag}_pca_corr_with_severity__merged.csv
         * {tag}_pca_severity_heatmap__merged.png
         * {tag}_pca_severity_correlation__merged.png

    2) SEPARATE mode: keep patched as its own label (5)
       - Outputs:
         * {tag}_pca_mean_per_severity__separate.csv  (rows 0..4 and 5)
         * {tag}_pca_corr_with_severity__separate_clean_only.csv  (uses 0..4 only)
         * {tag}_pca_corr_with_severity__separate_including_patched_mapped.csv
             (for correlation only, map 5 -> class_with_artifact)
         * {tag}_pca_severity_heatmap__separate.png
         * {tag}_pca_severity_correlation__separate_clean_only.png
         * {tag}_pca_severity_correlation__separate_including_patched_mapped.png

    Notes
    -----
    - labels: 0..4 are clean severities; 5 == patched subset of class_with_artifact.
    - We use Pearson r on the mean table to quantify monotonic alignment with severity.
    """

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    labels = np.asarray(labels)
    labels += 1
    class_with_artifact = class_with_artifact + 1
    cols = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(X_pca[:, :n_components], columns=cols)
    df["Label"] = labels

    # =========================
    # 1) MERGED MODE
    # =========================
    sev_merged = labels.copy()
    sev_merged[sev_merged == 6] = class_with_artifact  # merge patched into its true severity
    df_merged = df.copy()
    df_merged["Severity"] = sev_merged

    mean_df_merged = df_merged.groupby("Severity")[cols].mean().sort_index()
    corr_df_merged = _corr_from_mean_table(mean_df_merged, cols)

    # Save CSVs
    if out_dir:
        mean_csv = os.path.join(out_dir, f"{tag}_pca_mean_per_severity__merged.csv")
        corr_csv = os.path.join(out_dir, f"{tag}_pca_corr_with_severity__merged.csv")
        mean_df_merged.to_csv(mean_csv)
        corr_df_merged.to_csv(corr_csv, index=False)

    # Heatmap (merged)
    plt.figure(figsize=(6, 3))
    sns.heatmap(mean_df_merged, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Mean PCA scores per Severity – MERGED – {tag}")
    plt.xlabel("Principal Component")
    plt.ylabel("Severity Level")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{tag}_pca_severity_heatmap__merged.png"), dpi=200)
    plt.close()

    # Correlation bar (merged)
    plt.figure(figsize=(4, 3))
    sns.barplot(x="Component", y="Correlation_with_Severity", data=corr_df_merged, palette="viridis")
    plt.title(f"Correlation of PCA comps with Severity – MERGED – {tag}")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{tag}_pca_severity_correlation__merged.png"), dpi=200)
    plt.close()

    # =========================
    # 2) SEPARATE MODE
    # =========================
    # Keep label 5 as a distinct row in the mean table
    df_sep = df.copy()
    df_sep["Severity"] = labels  # this includes 0..4 and 5 as-is
    mean_df_separate = df_sep.groupby("Severity")[cols].mean().sort_index()

    # Correlation (separate) variant A: CLEAN ONLY (0..4)  -> excludes 5 from correlation
    if 6 in mean_df_separate.index:
        mean_df_clean_only = mean_df_separate.loc[[i for i in mean_df_separate.index if i in [1,2,3,4,5]]]
    else:
        mean_df_clean_only = mean_df_separate.copy()  # if no 5 present, just use all
    corr_df_sep_clean = _corr_from_mean_table(mean_df_clean_only, cols)

    # Correlation (separate) variant B: include patched by mapping 5->class_with_artifact for correlation *only*
    # (This gives you a correlation number comparable to MERGED while still keeping the patched row visible in means.)
    mean_df_for_corr_including_patch = mean_df_separate.copy()
    if 6 in mean_df_for_corr_including_patch.index:
        # Move row 5 into the row class_with_artifact by averaging with any existing row
        # Build a temporary table with merged rows for correlation purpose
        tmp = mean_df_for_corr_including_patch.copy()
        # If both exist, average them; else just map 5 -> class_with_artifact
        if class_with_artifact in tmp.index:
            merged_row = pd.DataFrame([tmp.loc[[class_with_artifact, 6]].mean(axis=0)], index=[class_with_artifact])
            tmp = tmp.drop(index=[class_with_artifact, 6])
            tmp = pd.concat([tmp, merged_row], axis=0).sort_index()
        else:
            # if somehow class_with_artifact row doesn't exist, just rename 5 to it
            tmp = tmp.rename(index={6: class_with_artifact}).sort_index()
        corr_df_sep_including_patch_mapped = _corr_from_mean_table(tmp, cols)
    else:
        corr_df_sep_including_patch_mapped = _corr_from_mean_table(mean_df_separate, cols)

    # Save CSVs (separate)
    if out_dir:
        mean_csv = os.path.join(out_dir, f"{tag}_pca_mean_per_severity__separate.csv")
        mean_df_separate.to_csv(mean_csv)

        corr_clean_csv = os.path.join(out_dir, f"{tag}_pca_corr_with_severity__separate_clean_only.csv")
        corr_df_sep_clean.to_csv(corr_clean_csv, index=False)

        corr_including_patch_csv = os.path.join(out_dir, f"{tag}_pca_corr_with_severity__separate_including_patched_mapped.csv")
        corr_df_sep_including_patch_mapped.to_csv(corr_including_patch_csv, index=False)

    # Heatmap (separate)
    plt.figure(figsize=(6, 3 + 0.2*len(mean_df_separate.index)))
    sns.heatmap(mean_df_separate, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Mean PCA scores per Severity – SEPARATE – {tag}")
    plt.xlabel("Principal Component")
    plt.ylabel("Severity Level (6 = patched class)")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{tag}_pca_severity_heatmap__separate.png"), dpi=200)
    plt.close()

    # Correlation bars (separate) – CLEAN ONLY
    plt.figure(figsize=(4, 3))
    sns.barplot(x="Component", y="Correlation_with_Severity", data=corr_df_sep_clean, palette="magma")
    plt.title(f"Correlation with Severity – SEPARATE (clean 1–5) – {tag}")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{tag}_pca_severity_correlation__separate_clean_only.png"), dpi=200)
    plt.close()

    # Correlation bars (separate) – INCLUDING PATCH MAPPED
    plt.figure(figsize=(4, 3))
    sns.barplot(x="Component", y="Correlation_with_Severity", data=corr_df_sep_including_patch_mapped, palette="cividis")
    plt.title(f"Correlation with Severity – SEPARATE (patched mapped→{class_with_artifact}) – {tag}")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{tag}_pca_severity_correlation__separate_including_patched_mapped.png"), dpi=200)
    plt.close()

    # Print quick console summaries
    print("\n=== MERGED mode: mean PCA per severity ===")
    print(mean_df_merged.round(4))
    print("\n=== MERGED mode: correlation with severity ===")
    print(corr_df_merged.round(4))

    print("\n=== SEPARATE mode: mean PCA per severity (rows include 6=patched) ===")
    print(mean_df_separate.round(4))

    print("\n=== SEPARATE mode: correlation (clean only, severities 1..5) ===")
    print(corr_df_sep_clean.round(4))

    print(f"\n=== SEPARATE mode: correlation (patched mapped→{class_with_artifact}) ===")
    print(corr_df_sep_including_patch_mapped.round(4))

    # Return all artifacts for programmatic use
    return {
        "mean_df_merged": mean_df_merged,
        "corr_df_merged": corr_df_merged,
        "mean_df_separate": mean_df_separate,
        "corr_df_separate_clean_only": corr_df_sep_clean,
        "corr_df_separate_including_patched_mapped": corr_df_sep_including_patch_mapped,
    }


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Run ProtoPnet code')

    parser.add_argument('--run_name',
                help = 'Define the run_name for saving everything (example: test_resnet)',
                default='DummyName')
                
    parser.add_argument('--param_jsonpath',
                        default='config/params_example_ordinal.json',
                        required=True,
                        help='Define the parameter file used for training (example: config/params_example_ordinal.json)')
    
    parser.add_argument('--datapath',
                        required=True)
    
    parser.add_argument('--savepath',
                        default = 'savedmodel',
                        required=True)
                    
    parser.add_argument('--pretrained_path',
                        default = 'config/pretrained_model.ckpt')

    # NEW FLAG
    # Extract custom flag manually
    use_saved_flag = get_flag_value("--use_saved", default=True)
    print(f"use_saved = {use_saved_flag}")

    # Parse command line arguments
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = { **params_dict,**args_dict}
    params = Parameters.from_dict(params_exp)

    params.set_savepaths()
    base_runname = params.run_name

    num_folds = 1
    for fold in range(num_folds):
        params.cv_fold = fold
        params.run_name = f'Fold{fold}_{base_runname}'
        params.set_savepaths()
        params.save_path_ims.mkdir(parents=True, exist_ok=True)

        txt_logger = logging.getLogger("pytorch_lightning")
        filehandler = logging.FileHandler(params.save_path / "logfile.log")
        txt_logger.addHandler(filehandler)

        rank_zero_info(f'Start fold {fold}')
        main(params, use_saved=use_saved_flag)

        txt_logger.removeHandler(filehandler)