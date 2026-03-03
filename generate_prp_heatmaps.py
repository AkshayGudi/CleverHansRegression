"""
Generate PRP (Prototype Relevance Propagation) heatmaps for a trained INSightR-Net model.

Usage examples:

  # Per-prototype heatmap for prototype 7:
  python generate_prp_heatmaps.py \
      --model_path saved_models/Epoch_50_after_protopushing.pth \
      --param_jsonpath config/params_example_ordinal.json \
      --image_path test_images/sample.png \
      --mode prototype \
      --prototype_number 7 \
      --output_dir prp_results/

  # Full prediction heatmap (through weighted mean):
  python generate_prp_heatmaps.py \
      --model_path saved_models/Epoch_50_after_protopushing.pth \
      --param_jsonpath config/params_example_ordinal.json \
      --image_path test_images/sample.png \
      --mode prediction \
      --output_dir prp_results/

  # Top-k prototype heatmaps:
  python generate_prp_heatmaps.py \
      --model_path saved_models/Epoch_50_after_protopushing.pth \
      --param_jsonpath config/params_example_ordinal.json \
      --image_path test_images/sample.png \
      --mode topk \
      --topk 5 \
      --output_dir prp_results/

  # Using a PyTorch Lightning checkpoint:
  python generate_prp_heatmaps.py \
      --checkpoint_path checkpoints/bestmodel_stage2.ckpt \
      --param_jsonpath config/params_example_ordinal.json \
      --image_path test_images/sample.png \
      --mode prediction \
      --output_dir prp_results/
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from define_parameters import Parameters
from helpers import load_json
from insight_training.model import construct_PPNet
from insight_training.prp import (
    PRPCanonizedModel,
    generate_prp_per_prototype,
    generate_prp_full_prediction,
    generate_prp_topk_prototypes,
    invert_normalize,
)


def load_model_from_state_dict(model_path, network_params, device):
    """Load a PPNet model from a state_dict .pth file."""
    ppnet = construct_PPNet(network_params=network_params)
    state_dict = torch.load(model_path, map_location=device)
    ppnet.load_state_dict(state_dict)
    ppnet = ppnet.to(device)
    ppnet.eval()
    return ppnet


def load_model_from_checkpoint(checkpoint_path, params, device):
    """Load a PPNet model from a PyTorch Lightning .ckpt file."""
    from insight_training.lightning_module_protodiab import LitModelProto
    lit_model = LitModelProto.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        params=params,
        dataloader_push=None,
        map_location=device,
    )
    ppnet = lit_model.ppnet.to(device)
    ppnet.eval()
    return ppnet


def preprocess_image(image_path, img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Load an image, resize, normalize, and return as a [1, 3, H, W] tensor."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

    for c in range(3):
        img_tensor[c] = (img_tensor[c] - mean[c]) / std[c]

    return img_tensor.unsqueeze(0)


def save_heatmap(heatmap, save_path, title=None):
    """Save a heatmap as a seismic colormap image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='seismic', vmin=-1, vmax=1)
    if title:
        plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def save_overlay(heatmap, img_tensor, save_path, alpha_img=0.4, alpha_hm=0.6):
    """Save an overlay of the heatmap on the original image."""
    img_denorm = invert_normalize(img_tensor.squeeze(0).cpu())
    img_np = img_denorm.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    hm_colored = plt.cm.seismic((heatmap + 1) / 2.0)[:, :, :3]
    overlay = alpha_img * img_np + alpha_hm * hm_colored
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate PRP heatmaps for INSightR-Net')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model state_dict (.pth file)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to PyTorch Lightning checkpoint (.ckpt file)')
    parser.add_argument('--param_jsonpath', type=str, required=True,
                        help='Path to parameter JSON file used during training')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the test image')
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prototype', 'prediction', 'topk'],
                        help='Heatmap mode: single prototype, full prediction, or top-k prototypes')
    parser.add_argument('--prototype_number', type=int, default=0,
                        help='Prototype index for "prototype" mode')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top prototypes for "topk" mode')
    parser.add_argument('--output_dir', type=str, default='prp_results',
                        help='Directory to save output heatmaps')
    parser.add_argument('--datapath', type=str, default='.',
                        help='Data path (needed for Parameters loading)')
    parser.add_argument('--savepath', type=str, default='.',
                        help='Save path (needed for Parameters loading)')
    parser.add_argument('--pretrained_path', type=str, default='config/pretrained_model.ckpt',
                        help='Save path (needed for Parameters loading)')
    

    args = parser.parse_args()

    if args.model_path is None and args.checkpoint_path is None:
        parser.error("Provide either --model_path or --checkpoint_path")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load parameters ---
    params_dict = load_json(args.param_jsonpath)
    params_exp = {
        **params_dict,
        'param_jsonpath': args.param_jsonpath,
        'datapath': args.datapath,
        'savepath': args.savepath,
        'pretrained_path': 'config/pretrained_model.ckpt'
    }
    params = Parameters.from_dict(params_exp)
    network_params = params.network_params

    # --- Load model ---
    print("Loading trained model...")
    if args.model_path:
        ppnet = load_model_from_state_dict(args.model_path, network_params, device)
    else:
        ppnet = load_model_from_checkpoint(args.checkpoint_path, params, device)

    base_arch = network_params.base_architecture
    print(f"Base architecture: {base_arch}")
    print(f"Num prototypes: {ppnet.num_prototypes}")
    print(f"Proto activation: {ppnet.proto_activation}")
    print(f"Spatial size: {ppnet.output_size_conv}")

    # --- Canonize model for PRP ---
    print("Canonizing model for PRP...")
    prp_model = PRPCanonizedModel(ppnet, base_arch)
    print("Canonization complete.")

    # --- Preprocess image ---
    img_tensor = preprocess_image(args.image_path, network_params.img_size)
    img_tensor = img_tensor.to(device)
    image_name = Path(args.image_path).stem

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Generate heatmaps ---
    if args.mode == 'prototype':
        pno = args.prototype_number
        print(f"Generating PRP heatmap for prototype {pno}...")
        heatmap = generate_prp_per_prototype(img_tensor, pno, prp_model, device)

        save_path = os.path.join(args.output_dir, f'{image_name}_prp_proto{pno}.png')
        save_heatmap(heatmap, save_path, title=f'PRP - Prototype {pno}')
        print(f"Saved: {save_path}")

        overlay_path = os.path.join(args.output_dir, f'{image_name}_prp_proto{pno}_overlay.png')
        save_overlay(heatmap, img_tensor, overlay_path)
        print(f"Saved: {overlay_path}")

    elif args.mode == 'prediction':
        print("Generating full-prediction PRP heatmap...")
        heatmap, pred_value = generate_prp_full_prediction(img_tensor, prp_model, device)

        save_path = os.path.join(args.output_dir, f'{image_name}_prp_prediction.png')
        save_heatmap(heatmap, save_path, title=f'PRP Full Prediction (pred={pred_value:.3f})')
        print(f"Prediction value: {pred_value:.4f}")
        print(f"Saved: {save_path}")

        overlay_path = os.path.join(args.output_dir, f'{image_name}_prp_prediction_overlay.png')
        save_overlay(heatmap, img_tensor, overlay_path)
        print(f"Saved: {overlay_path}")

    elif args.mode == 'topk':
        print(f"Generating PRP heatmaps for top-{args.topk} prototypes...")
        result = generate_prp_topk_prototypes(img_tensor, prp_model, device, k=args.topk)

        print(f"Prediction value: {result['prediction']:.4f}")
        for i, (pno, contrib, hm) in enumerate(zip(
                result['prototype_indices'],
                result['contributions'],
                result['heatmaps'])):

            rank = i + 1
            save_path = os.path.join(
                args.output_dir,
                f'{image_name}_prp_top{rank}_proto{pno}.png')
            proto_label = prp_model.proto_classes[pno].item()
            save_heatmap(hm, save_path,
                         title=f'Rank {rank}: Proto {pno} (label={proto_label:.2f}, contrib={contrib:.4f})')
            print(f"  Rank {rank}: Prototype {pno} (label={proto_label:.2f}, contribution={contrib:.4f})")
            print(f"  Saved: {save_path}")

            overlay_path = os.path.join(
                args.output_dir,
                f'{image_name}_prp_top{rank}_proto{pno}_overlay.png')
            save_overlay(hm, img_tensor, overlay_path)

    print("\nDone!")


if __name__ == '__main__':
    main()