"""
Entry point for generating PLRP-PRP (Pruned Layer-wise Relevance Propagation
applied to Prototypical Relevance Propagation) heatmaps for a trained
INSightR-Net model.

This is a copy of ../main_generate_prp.py with two extra flags:
    --plrp_p_pos  proportion of positive relevance to prune at every
                  parametric layer (Conv2d beta=0 / Linear epsilon).
                  Default: 0.0 (no pruning, behaves identically to
                  the standard PRP pipeline).
    --plrp_p_neg  proportion of negative relevance to prune. Default 0.0.

Run from /sc/home/akshay.gudi/code/CleverHansRegression with the same
checkpoint, params JSON, and image set as your existing PRP runs.

Examples
--------
1) Sanity check -- p=0 must produce the same heatmap as main_generate_prp.py:

    cd /sc/home/akshay.gudi/code/CleverHansRegression
    conda activate new_insight_env

    python3 plrp_ext/main_generate_plrp.py \
        --model_path=/path/to/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=plrp_ext/outputs/sanity_p0/proto16 \
        --prototypes 16 \
        --plrp_p_pos 0.0 \
        --plrp_p_neg 0.0

2) Real PLRP-lambda run (paper defaults p_pos=0.25, p_neg=0.125):

    python3 plrp_ext/main_generate_plrp.py \
        --model_path=/path/to/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=plrp_ext/outputs/p_pos_025/proto16 \
        --prototypes 16 \
        --plrp_p_pos 0.25 \
        --plrp_p_neg 0.125

3) PLRP-lambda on a test image:

    python3 plrp_ext/main_generate_plrp.py \
        --model_path=/path/to/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=plrp_ext/outputs/p_pos_025/test_image \
        --test_image_path=/path/to/test_image.jpeg \
        --prototypes 16 \
        --plrp_p_pos 0.25 \
        --plrp_p_neg 0.125
"""

# When this file is run as ``python3 plrp_ext/main_generate_plrp.py``, Python
# puts ``plrp_ext/`` first on sys.path, not the CleverHansRegression project
# root. Top-level packages like ``helpers`` and ``define_parameters`` then
# fail to import. Prepend the repo root so the script behaves like
# ``main_generate_prp.py`` run from the project directory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import load_json
from define_parameters import NetworkParams
from insight_training.model import PPNet, construct_PPNet

from plrp_ext.lrp_general6_plrp import set_plrp_params, get_plrp_params
from plrp_ext.insight_prp_plrp import (
    PRPCanonizedModel,
    generate_prp_all_prototypes,
    generate_prp_for_image,
    generate_prp_image,
    _create_overlay,
)


def create_insightr_overlay(orig_img_np, actmap_np):
    """Reproduces the INSightR-Net JET-colormap activation overlay."""
    h, w = orig_img_np.shape[:2]

    upsampled_act = cv2.resize(actmap_np, (w, h), interpolation=cv2.INTER_CUBIC)

    act_min = np.amin(upsampled_act)
    act_max = np.amax(upsampled_act)
    if act_max - act_min > 0:
        upsampled_act = (upsampled_act - act_min) / (act_max - act_min)
    else:
        upsampled_act = np.zeros_like(upsampled_act)

    jet_heatmap = cv2.applyColorMap(np.uint8(255 * upsampled_act), cv2.COLORMAP_JET)
    jet_heatmap = np.float32(jet_heatmap) / 255.0
    jet_heatmap = jet_heatmap[..., ::-1]

    overlay = np.clip(0.5 * orig_img_np + 0.3 * jet_heatmap, 0, 1)
    return overlay


def create_insightr_red_overlay(orig_img_np, actmap_np, max_alpha=0.85):
    """Red/yellow PRP-style overlay built from an INSightR-Net activation map."""
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


def create_comparison_image(orig_img_np, insightr_overlay, prp_overlay, save_path, pno):
    """Three-panel figure: original | INSightR-Net activation | PLRP-PRP."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_img_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(insightr_overlay)
    axes[1].set_title('INSightR-Net Activation Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(prp_overlay)
    axes[2].set_title('PLRP-PRP Relevance Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(f'Prototype {pno} -- Attention Comparison (PLRP-PRP)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_red_comparison_image(orig_img_np, insightr_red_overlay, prp_overlay, save_path, pno):
    """Three-panel figure: original | INSightR-Net red-overlay | PLRP-PRP."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_img_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(insightr_red_overlay)
    axes[1].set_title('INSightR-Net Red Activation Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(prp_overlay)
    axes[2].set_title('PLRP-PRP Relevance Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(f'Prototype {pno} -- Red Attention Comparison (PLRP-PRP)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def load_ppnet_from_checkpoint(model_path, network_params, device):
    """Loads PPNet from .pth (saved_models) or .ckpt (Lightning) checkpoints."""
    checkpoint = torch.load(model_path, map_location=device)

    ppnet = construct_PPNet(network_params=network_params)
    ppnet = ppnet.to(device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        ppnet_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ppnet.'):
                ppnet_state_dict[key[len('ppnet.'):]] = value
        ppnet.load_state_dict(ppnet_state_dict, strict=True)
        print(f"Loaded ppnet from Lightning checkpoint: {model_path}")
    else:
        ppnet.load_state_dict(checkpoint, strict=True)
        print(f"Loaded ppnet from saved model: {model_path}")

    ppnet.eval()
    return ppnet


def _resolve_prototype_indices(prototype_number, prototypes_list):
    if prototypes_list is not None:
        return list(dict.fromkeys(prototypes_list))
    if prototype_number is not None:
        return [prototype_number]
    return None


def generate_prp_for_one_stored_prototype(pno, prp_model, device, output_dir):
    """PLRP-PRP from the model's stored prototype_images[pno]; writes prototype_{pno}/."""
    proto_img = prp_model.prototype_images[pno]
    if proto_img.max() == 0:
        print(f"Prototype {pno}: no stored image (all zeros), skipping.")
        return False

    img_float = proto_img.float() / 255.0
    img_tensor = img_float.permute(2, 0, 1).unsqueeze(0)

    print(f"Generating PLRP-PRP heatmap for prototype {pno}...")
    heatmap = generate_prp_image(img_tensor, pno, prp_model, device)

    proto_dir = output_dir / f"prototype_{pno}"
    proto_dir.mkdir(parents=True, exist_ok=True)

    orig_img_np = proto_img.cpu().numpy() / 255.0
    plt.imsave(str(proto_dir / "heatmap.png"), heatmap, cmap="seismic", vmin=-1, vmax=1)
    plt.imsave(str(proto_dir / "original.png"), orig_img_np)
    prp_overlay = _create_overlay(orig_img_np, heatmap)
    plt.imsave(str(proto_dir / "overlay.png"), prp_overlay, vmin=0, vmax=1)

    actmap = prp_model.prototype_actmaps[pno].cpu().numpy()
    insightr_overlay = create_insightr_overlay(orig_img_np, actmap)
    insightr_red_overlay = create_insightr_red_overlay(orig_img_np, actmap)
    create_comparison_image(
        orig_img_np, insightr_overlay, prp_overlay, proto_dir / "comparison.png", pno
    )
    create_red_comparison_image(
        orig_img_np, insightr_red_overlay, prp_overlay, proto_dir / "comparison_redoverlay.png", pno
    )

    print(f"Saved to {proto_dir}/:")
    print("  original.png, heatmap.png, overlay.png, comparison.png, comparison_redoverlay.png")
    return True


def load_test_image(image_path, img_size=540):
    """Load and preprocess a test image (BGR for model, RGB for visualization)."""
    jpeg_im = cv2.imread(str(image_path))
    if jpeg_im is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if jpeg_im.shape[0] != img_size or jpeg_im.shape[1] != img_size:
        jpeg_im = cv2.resize(jpeg_im, (img_size, img_size))

    norm = jpeg_im / 255.0
    img_tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
    raw_rgb_np = cv2.cvtColor(jpeg_im, cv2.COLOR_BGR2RGB) / 255.0
    return img_tensor, raw_rgb_np.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Generate PLRP-PRP heatmaps for INSightR-Net')

    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pth from saved_models/ preferred, .ckpt also supported)')
    parser.add_argument('--param_jsonpath', required=True,
                        help='Path to parameter JSON file (e.g., config/params_example_ordinal.json)')
    parser.add_argument('--output_dir', default='plrp_ext/outputs/default',
                        help='Directory to save PLRP-PRP heatmaps')
    parser.add_argument('--prototype_number', type=int, default=None,
                        help='Generate PLRP-PRP for a single prototype. Mutually exclusive with --prototypes.')
    parser.add_argument('--prototypes', type=int, nargs='+', default=None,
                        help='Generate PLRP-PRP for these prototype indices.')
    parser.add_argument('--test_image_path', type=str, default=None,
                        help='Path to a test image (optional; if not provided, uses stored prototype images)')

    parser.add_argument('--plrp_p_pos', type=float, default=0.0,
                        help='Proportion of positive relevance to prune at each parametric layer. Range [0, 1). Default 0.0 (= unpruned PRP).')
    parser.add_argument('--plrp_p_neg', type=float, default=0.0,
                        help='Proportion of negative relevance to prune at each parametric layer. Range [0, 1). Default 0.0.')

    args = parser.parse_args()

    if args.prototype_number is not None and args.prototypes is not None:
        parser.error('Use only one of --prototype_number and --prototypes')

    set_plrp_params(p_pos=args.plrp_p_pos, p_neg=args.plrp_p_neg)
    eff_p_pos, eff_p_neg = get_plrp_params()
    if eff_p_pos == 0.0 and eff_p_neg == 0.0:
        print("PLRP pruning DISABLED (p_pos=0, p_neg=0). Output should match unpruned PRP exactly.")
    else:
        print(f"PLRP-lambda ENABLED with p_pos={eff_p_pos}, p_neg={eff_p_neg}.")

    proto_indices = _resolve_prototype_indices(args.prototype_number, args.prototypes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    params_dict = load_json(args.param_jsonpath)
    network_params_dict = params_dict.get('network_params', {})
    network_params = NetworkParams.from_dict(network_params_dict)
    print(f"Network config: {network_params.base_architecture}, "
          f"proto_shape={network_params.proto_shape}, "
          f"img_size={network_params.img_size}")

    ppnet = load_ppnet_from_checkpoint(args.model_path, network_params, device)
    ppnet.base_architecture = network_params.base_architecture

    print("Canonizing model for PLRP-PRP...")
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
        print(f"Loading test image: {args.test_image_path}")
        img_tensor, raw_rgb_np = load_test_image(
            args.test_image_path, img_size=network_params.img_size
        )

        print("Generating PLRP-PRP heatmaps for test image...")
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
        saved = 0
        for pno in proto_indices:
            if generate_prp_for_one_stored_prototype(pno, prp_model, device, output_dir):
                saved += 1
        if saved == 0:
            print("No prototypes were saved. Run prototype pushing first or use --test_image_path.")

    else:
        print("Generating PLRP-PRP heatmaps for all prototypes...")

        def _comparison_callback(orig_img_np, prp_overlay, pno, proto_dir):
            actmap = prp_model.prototype_actmaps[pno].cpu().numpy()
            insightr_overlay = create_insightr_overlay(orig_img_np, actmap)
            insightr_red_overlay = create_insightr_red_overlay(orig_img_np, actmap)
            comparison_path = Path(proto_dir) / "comparison.png"
            comparison_red_path = Path(proto_dir) / "comparison_redoverlay.png"
            create_comparison_image(orig_img_np, insightr_overlay, prp_overlay, comparison_path, pno)
            create_red_comparison_image(orig_img_np, insightr_red_overlay, prp_overlay, comparison_red_path, pno)

        heatmaps = generate_prp_all_prototypes(
            prp_model, device, str(output_dir), comparison_fn=_comparison_callback
        )
        print(f"Done. Generated {len(heatmaps)} heatmaps in {output_dir}")


if __name__ == "__main__":
    main()
