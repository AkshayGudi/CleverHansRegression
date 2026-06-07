"""
Entry point for generating PRP (Prototypical Relevance Propagation) heatmaps
for a trained INSightR-Net model.

Usage:
    # Generate PRP for all prototypes (using stored prototype images):
    python3 main_generate_prp.py \
        --model_path=path/to/saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=prp_output

    # Generate PRP for a specific prototype only:
    python3 main_generate_prp.py \
        --model_path=path/to/saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=prp_output \
        --prototype_number=5

    # Generate PRP for several prototypes (stored prototype images):
    python3 main_generate_prp.py \
        --model_path=path/to/saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=prp_output \
        --prototypes 5 12 58

    # Generate PRP for a test image (across specified or all prototypes):
    python3 main_generate_prp.py \
        --model_path=path/to/saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=prp_output \
        --test_image_path=path/to/test_image.jpeg

    Note: Use the *_after_protopushing.pth file at the latest epoch from
    the saved_models/ folder. This ensures prototypes are pushed to real
    images and prototype_images are populated for visualization.

Example usage:
cd /sc/home/akshay.gudi/code/CleverHansRegression && conda activate new_insight_env

python3 main_generate_prp.py \
  --model_path=/sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v4_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_25_Apr_3/saved_models/Epoch_50_after_protopushing.pth \
  --param_jsonpath=/sc/home/akshay.gudi/code/CleverHansRegression/config/params_example_ordinal.json \
  --output_dir=/sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v4_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_25_Apr_3/img/prp_class3_prototypes_epoch50 \
  --prototypes 2 8 11 16 22 23 24 25 26 27 31 33 37 38 39 

"""
# New version of LRP code

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import load_json
from define_parameters import NetworkParams
from insight_training.model import PPNet, construct_PPNet
from insight_prp import PRPCanonizedModel, generate_prp_all_prototypes, generate_prp_for_image
from thesis_figure_export import (
    ThesisFigureExport,
    THESIS_3PANEL,
    add_thesis_export_args,
    export_config_from_args,
    log_export_settings,
    save_figure_multi,
)

# Updated by ``main()`` from CLI; used by comparison figure writers.
_FIG_EXPORT: ThesisFigureExport = THESIS_3PANEL


def create_insightr_overlay(orig_img_np, actmap_np):
    """
    Recreate the INSightR-Net activation overlay for a prototype.

    This reproduces the same overlay that INSightR-Net generates during
    prototype pushing (see helpers.plot_prototypes and helpers.plot_heatmap).

    The INSightR-Net overlay uses:
      1. The prototype's stored activation map (prototype_actmaps),
         upsampled to full image resolution.
      2. A JET colormap applied to the normalized activation map.
      3. A fixed blending: 0.5 * original_image + 0.3 * jet_heatmap.

    Args:
        orig_img_np: Original prototype image as numpy array (H, W, 3), values in [0, 1], RGB.
        actmap_np: Raw activation map from model.prototype_actmaps[pno],
                   numpy array of shape (spatial_size, spatial_size).

    Returns:
        overlay: numpy array (H, W, 3) in [0, 1] — the INSightR-Net style overlay.
    """
    h, w = orig_img_np.shape[:2]

    # Upsample the small activation map (e.g. 9x9) to full image size,
    # matching what push_prototypes.py does with cv2.INTER_CUBIC.
    upsampled_act = cv2.resize(actmap_np, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize the activation map to [0, 1] range,
    # matching helpers.plot_heatmap: subtract min, divide by max.
    act_min = np.amin(upsampled_act)
    act_max = np.amax(upsampled_act)
    if act_max - act_min > 0:
        upsampled_act = (upsampled_act - act_min) / (act_max - act_min)
    else:
        upsampled_act = np.zeros_like(upsampled_act)

    # Apply JET colormap (same as helpers.plot_heatmap).
    # cv2.applyColorMap produces BGR, so we flip to RGB with [::-1].
    jet_heatmap = cv2.applyColorMap(np.uint8(255 * upsampled_act), cv2.COLORMAP_JET)
    jet_heatmap = np.float32(jet_heatmap) / 255.0
    jet_heatmap = jet_heatmap[..., ::-1]  # BGR → RGB

    # Blend using the same weights as helpers.plot_prototypes:
    # overlay = 0.5 * original + 0.3 * heatmap
    overlay = np.clip(0.5 * orig_img_np + 0.3 * jet_heatmap, 0, 1)
    return overlay


def create_insightr_red_overlay(orig_img_np, actmap_np, max_alpha=0.85):
    """Create a PRP-style red/yellow overlay from prototype activation map."""
    h, w = orig_img_np.shape[:2]
    upsampled_act = cv2.resize(actmap_np, (w, h), interpolation=cv2.INTER_CUBIC)
    act_min = np.amin(upsampled_act)
    act_max = np.amax(upsampled_act)
    if act_max - act_min > 0:
        upsampled_act = (upsampled_act - act_min) / (act_max - act_min)
    else:
        upsampled_act = np.zeros_like(upsampled_act)

    intensity = np.sqrt(np.clip(upsampled_act, 0, 1))
    alpha = (intensity * max_alpha)[:, :, np.newaxis]

    hm_color = np.zeros((h, w, 3), dtype=np.float32)
    hm_color[:, :, 0] = 1.0
    hm_color[:, :, 1] = intensity * 0.4
    hm_color[:, :, 2] = 0.0

    dim_factor = 1.0 - 0.3 * (1.0 - intensity[:, :, np.newaxis])
    dimmed_original = orig_img_np * dim_factor
    overlay = (1 - alpha) * dimmed_original + alpha * hm_color
    return np.clip(overlay, 0, 1)


_PRP_IMG_FORMATS = ("png", "svg")


def create_comparison_image(orig_img_np, insightr_overlay, prp_overlay, proto_dir, pno,
                            basename="comparison",
                            export_cfg: ThesisFigureExport | None = None):
    """
    Create a side-by-side comparison figure with 3 panels:
      1. Original prototype image
      2. INSightR-Net activation overlay (JET colormap, from training)
      3. PRP relevance overlay (red/yellow highlights from LRP)

    This allows direct visual comparison of where INSightR-Net's
    activation-based explanation focuses vs. where PRP's pixel-level
    relevance propagation focuses.

    Args:
        orig_img_np: Original prototype image, numpy array (H, W, 3) in [0, 1].
        insightr_overlay: INSightR-Net activation overlay, numpy array (H, W, 3) in [0, 1].
        prp_overlay: PRP relevance overlay, numpy array (H, W, 3) in [0, 1].
        proto_dir: Prototype output folder; figure is written to its png/ and svg/ subfolders.
        pno: Prototype index (int), used in the figure title.
        basename: Output file stem (default "comparison").
    """
    cfg = export_cfg or _FIG_EXPORT
    fig, axes = plt.subplots(1, 3, figsize=cfg.figsize)
    fs = cfg.panel_title_fontsize

    axes[0].imshow(orig_img_np)
    axes[0].set_title('Original Image', fontsize=fs, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(insightr_overlay)
    axes[1].set_title('INSightR-Net Activation Overlay', fontsize=fs, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(prp_overlay)
    axes[2].set_title('PRP Relevance Overlay', fontsize=fs, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(
        f'Prototype {pno} — Attention Comparison',
        fontsize=cfg.suptitle_fontsize, fontweight='bold', y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_multi(fig, proto_dir, basename, cfg, formats=_PRP_IMG_FORMATS)
    plt.close(fig)


def create_red_comparison_image(orig_img_np, insightr_red_overlay, prp_overlay, proto_dir, pno,
                                basename="comparison_redoverlay",
                                export_cfg: ThesisFigureExport | None = None):
    """Create 3-panel comparison using red overlay for activation.

    Writes to ``<proto_dir>/png/<basename>.png`` and ``<proto_dir>/svg/<basename>.svg``.
    """
    cfg = export_cfg or _FIG_EXPORT
    fig, axes = plt.subplots(1, 3, figsize=cfg.figsize)
    fs = cfg.panel_title_fontsize
    axes[0].imshow(orig_img_np)
    axes[0].set_title('Original Image', fontsize=fs, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(insightr_red_overlay)
    axes[1].set_title('INSightR-Net Red Activation Overlay', fontsize=fs, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(prp_overlay)
    axes[2].set_title('PRP Relevance Overlay', fontsize=fs, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(
        f'Prototype {pno} — Red Attention Comparison',
        fontsize=cfg.suptitle_fontsize, fontweight='bold', y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_multi(fig, proto_dir, basename, cfg, formats=_PRP_IMG_FORMATS)
    plt.close(fig)


def load_ppnet_from_checkpoint(model_path, network_params, device):
    """
    Load PPNet from a saved model file.

    Supports:
      - .pth files from saved_models/ (ppnet.state_dict() — preferred)
      - .ckpt files from checkpoints/ (Lightning checkpoint with 'ppnet.' prefixed keys)
    """
    checkpoint = torch.load(model_path, map_location=device)

    ppnet = construct_PPNet(network_params=network_params)
    ppnet = ppnet.to(device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Lightning .ckpt — extract ppnet.* keys
        state_dict = checkpoint['state_dict']
        ppnet_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ppnet.'):
                ppnet_state_dict[key[len('ppnet.'):]] = value
        ppnet.load_state_dict(ppnet_state_dict, strict=True)
        print(f"Loaded ppnet from Lightning checkpoint: {model_path}")
    else:
        # .pth from saved_models/ — direct state_dict
        ppnet.load_state_dict(checkpoint, strict=True)
        print(f"Loaded ppnet from saved model: {model_path}")

    ppnet.eval()
    return ppnet


def _resolve_prototype_indices(prototype_number, prototypes_list):
    """
    Returns None (meaning all prototypes), or a list of int indices.
    prototypes_list takes precedence if both are given (caller should error first).
    """
    if prototypes_list is not None:
        return list(dict.fromkeys(prototypes_list))
    if prototype_number is not None:
        return [prototype_number]
    return None


def generate_prp_for_one_stored_prototype(pno, prp_model, device, output_dir):
    """
    PRP from the model's stored prototype_images[pno]; writes prototype_{pno}/.
    Returns True if saved, False if skipped (no stored image).
    """
    from insight_prp import generate_prp_image, _create_overlay, save_img_multi

    proto_img = prp_model.prototype_images[pno]
    if proto_img.max() == 0:
        print(f"Prototype {pno}: no stored image (all zeros), skipping.")
        return False

    img_float = proto_img.float() / 255.0
    # Dimension fix: prototype_images are stored as (H, W, C). PyTorch
    # expects (C, H, W); the previous permute(2, 1, 0) gave (C, W, H) and
    # silently transposed the spatial axes. Use (2, 0, 1) for HWC -> CHW.
    img_tensor = img_float.permute(2, 0, 1).unsqueeze(0)

    print(f"Generating PRP heatmap for prototype {pno}...")
    heatmap = generate_prp_image(img_tensor, pno, prp_model, device)

    proto_dir = output_dir / f"prototype_{pno}"
    proto_dir.mkdir(parents=True, exist_ok=True)

    orig_img_np = proto_img.cpu().numpy() / 255.0
    save_img_multi(proto_dir, "heatmap", heatmap, cmap="seismic", vmin=-1, vmax=1)
    save_img_multi(proto_dir, "original", orig_img_np)
    prp_overlay = _create_overlay(orig_img_np, heatmap)
    save_img_multi(proto_dir, "overlay", prp_overlay, vmin=0, vmax=1)

    actmap = prp_model.prototype_actmaps[pno].cpu().numpy()
    insightr_overlay = create_insightr_overlay(orig_img_np, actmap)
    insightr_red_overlay = create_insightr_red_overlay(orig_img_np, actmap)
    create_comparison_image(
        orig_img_np, insightr_overlay, prp_overlay, proto_dir, pno, basename="comparison"
    )
    create_red_comparison_image(
        orig_img_np, insightr_red_overlay, prp_overlay, proto_dir, pno,
        basename="comparison_redoverlay",
    )

    print(f"Saved to {proto_dir}/ (png/ and svg/ subfolders):")
    print("  original, heatmap, overlay, comparison, comparison_redoverlay")
    return True


def load_test_image(image_path, img_size=540):
    """Load and preprocess a test image.

    Returns a tuple ``(img_tensor, raw_rgb_np)`` where:
      - ``img_tensor`` is the preprocessed tensor fed to the model with
        the standard PyTorch (C, H, W) layout, BGR channels (consistent
        with how the model was trained via ``cv2.imread`` + permute).
      - ``raw_rgb_np`` is the un-permuted H x W x 3 RGB image in [0, 1],
        suitable for overlaying heatmaps on top of the test image as a
        human would view it.
    """
    jpeg_im = cv2.imread(str(image_path))
    if jpeg_im is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if jpeg_im.shape[0] != img_size or jpeg_im.shape[1] != img_size:
        jpeg_im = cv2.resize(jpeg_im, (img_size, img_size))

    norm = jpeg_im / 255.0
    # Dimension fix: cv2.imread returns (H, W, C); PyTorch expects
    # (C, H, W). The previous permute(2, 1, 0) yielded (C, W, H) and
    # silently transposed the spatial axes. Use (2, 0, 1) for HWC -> CHW.
    img_tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
    raw_rgb_np = cv2.cvtColor(jpeg_im, cv2.COLOR_BGR2RGB) / 255.0
    return img_tensor, raw_rgb_np.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Generate PRP heatmaps for INSightR-Net')

    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pth from saved_models/ preferred, .ckpt also supported)')
    parser.add_argument('--param_jsonpath', required=True,
                        help='Path to parameter JSON file (e.g., config/params_example_ordinal.json)')
    parser.add_argument('--output_dir', default='prp_output',
                        help='Directory to save PRP heatmaps (default: prp_output)')
    parser.add_argument('--prototype_number', type=int, default=None,
                        help='Generate PRP for a single prototype (default: all). Mutually exclusive with --prototypes.')
    parser.add_argument('--prototypes', type=int, nargs='+', default=None,
                        help='Generate PRP for these prototype indices, e.g. --prototypes 5 12 58. Mutually exclusive with --prototype_number.')
    parser.add_argument('--test_image_path', type=str, default=None,
                        help='Path to a test image (optional; if not provided, uses stored prototype images)')
    add_thesis_export_args(parser, panel_count=3)

    args = parser.parse_args()

    global _FIG_EXPORT
    _FIG_EXPORT = export_config_from_args(args, panel_count=3)
    log_export_settings(_FIG_EXPORT, "main_generate_prp")

    if args.prototype_number is not None and args.prototypes is not None:
        parser.error('Use only one of --prototype_number and --prototypes')

    proto_indices = _resolve_prototype_indices(args.prototype_number, args.prototypes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load parameters
    params_dict = load_json(args.param_jsonpath)
    network_params_dict = params_dict.get('network_params', {})
    network_params = NetworkParams.from_dict(network_params_dict)
    print(f"Network config: {network_params.base_architecture}, "
          f"proto_shape={network_params.proto_shape}, "
          f"img_size={network_params.img_size}")

    # Load model
    ppnet = load_ppnet_from_checkpoint(args.model_path, network_params, device)

    # Store base_architecture for canonization auto-detection
    ppnet.base_architecture = network_params.base_architecture

    # Canonize the model for PRP
    print("Canonizing model for PRP...")
    prp_model = PRPCanonizedModel(ppnet)
    prp_model = prp_model.to(device)
    print("Canonization complete.")

    if proto_indices is not None:
        n = prp_model.num_prototypes
        invalid = [p for p in proto_indices if p < 0 or p >= n]
        if invalid:
            raise ValueError(
                f"Invalid prototype index(es) {invalid}; model has num_prototypes={n} (valid 0..{n - 1})."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test_image_path is not None:
        # Mode: PRP for a test image
        print(f"Loading test image: {args.test_image_path}")
        img_tensor, raw_rgb_np = load_test_image(
            args.test_image_path, img_size=network_params.img_size
        )

        print("Generating PRP heatmaps for test image...")
        heatmaps = generate_prp_for_image(
            img_tensor,
            prp_model,
            device,
            str(output_dir),
            prototype_indices=proto_indices,
            raw_image_np=raw_rgb_np,
        )
        print(f"Saved {len(heatmaps)} heatmap(s) to {output_dir}")

    elif proto_indices is not None:
        # Mode: PRP for one or more prototypes using stored prototype images
        saved = 0
        for pno in proto_indices:
            if generate_prp_for_one_stored_prototype(pno, prp_model, device, output_dir):
                saved += 1
        if saved == 0:
            print("No prototypes were saved. Run prototype pushing first or use --test_image_path.")

    else:
        # Mode: PRP for all prototypes
        print("Generating PRP heatmaps for all prototypes...")

        # Build a comparison callback that will be called for each prototype.
        # It creates the INSightR-Net activation overlay from prototype_actmaps
        # and combines it with the PRP overlay into a single comparison figure.
        def _comparison_callback(orig_img_np, prp_overlay, pno, proto_dir):
            actmap = prp_model.prototype_actmaps[pno].cpu().numpy()
            insightr_overlay = create_insightr_overlay(orig_img_np, actmap)
            insightr_red_overlay = create_insightr_red_overlay(orig_img_np, actmap)
            create_comparison_image(
                orig_img_np, insightr_overlay, prp_overlay, proto_dir, pno, basename="comparison"
            )
            create_red_comparison_image(
                orig_img_np, insightr_red_overlay, prp_overlay, proto_dir, pno,
                basename="comparison_redoverlay",
            )

        heatmaps = generate_prp_all_prototypes(
            prp_model, device, str(output_dir), comparison_fn=_comparison_callback
        )
        print(f"Done. Generated {len(heatmaps)} heatmaps in {output_dir}")


if __name__ == "__main__":
    main()
