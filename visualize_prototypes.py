"""
Visualize how prototypes see a given input image in a trained INSightR-Net model.

For a given image, this script shows:
  - The model's prediction (weighted mean of prototype contributions)
  - The top-k most influential prototypes, each with:
      * Activation heatmap overlaid on the test image
      * Bounding box around the most activated region
      * The stored prototype image (the training patch this prototype represents)
      * The prototype's activation heatmap on its own stored image
      * Numerical contribution to the final prediction

Usage:
  python visualize_prototypes.py \
      --model_path saved_models/Epoch_50_after_protopushing.pth \
      --param_jsonpath config/params_example_ordinal.json \
      --image_path test_images/sample.png \
      --topk 10 \
      --output_dir prototype_visualizations/

  # Using a Lightning checkpoint:
  python visualize_prototypes.py \
      --checkpoint_path checkpoints/bestmodel_stage2.ckpt \
      --param_jsonpath config/params_example_ordinal.json \
      --image_path test_images/sample.png \
      --topk 5 \
      --output_dir prototype_visualizations/
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from define_parameters import Parameters
from helpers import find_high_activation_crop, load_json, plot_heatmap, plot_rectangle
from insight_training.model import construct_PPNet


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


def load_and_preprocess_image(image_path, img_size):
    """Load an image with OpenCV (matching the training pipeline) and return
    both the raw display image and the normalized tensor."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))

    img_rgb = img_bgr[:, :, ::-1].copy()
    img_display = img_rgb.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(
        img_bgr.transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0)

    return img_tensor, img_display


def analyze_image(ppnet, img_tensor, device, topk=10):
    """Run the image through the model and return all prototype analysis info."""
    ppnet.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits, distances, min_distances, prototype_activations, img_activations = \
            ppnet.forward_testtime(img_tensor)

    prediction = torch.clamp(logits.squeeze(), min=1, max=5).item()

    class_idx = ppnet.proto_classes.unsqueeze(0)
    weight_squared = ppnet.last_layer.weight.data.square()
    ll_noclass = weight_squared / class_idx

    contributions = (prototype_activations * ll_noclass).squeeze()
    total_weight = torch.sum(contributions).item()

    _, topk_indices = torch.topk(contributions.abs().squeeze(), k=min(topk, ppnet.num_prototypes))
    topk_indices = topk_indices.cpu().tolist()

    results = []
    for pno in topk_indices:
        proto_label = ppnet.proto_classes[pno].item()
        similarity = prototype_activations.squeeze()[pno].item()
        weight = ll_noclass.squeeze()[pno].item()
        contribution = contributions[pno].item()
        pct_contribution = (contribution / total_weight * 100) if total_weight != 0 else 0

        activation_map = img_activations[0, pno].cpu().numpy()

        upsampled_act = cv2.resize(
            activation_map,
            (ppnet.img_size, ppnet.img_size),
            interpolation=cv2.INTER_CUBIC,
        )

        proto_stored_img = ppnet.prototype_images[pno].cpu().numpy()
        proto_stored_actmap = ppnet.prototype_actmaps[pno].cpu().numpy()

        upsampled_proto_act = cv2.resize(
            proto_stored_actmap,
            (proto_stored_img.shape[1], proto_stored_img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        results.append({
            'index': pno,
            'proto_label': proto_label,
            'similarity': similarity,
            'weight': weight,
            'contribution': contribution,
            'pct_contribution': pct_contribution,
            'upsampled_activation': upsampled_act,
            'proto_image': proto_stored_img,
            'proto_actmap': upsampled_proto_act,
        })

    return {
        'prediction': prediction,
        'total_weight': total_weight,
        'prototypes': results,
    }


def create_summary_figure(img_display, analysis, output_path, topk_show=5):
    """Create a summary figure showing the test image and top-k prototype matches."""
    prototypes = analysis['prototypes'][:topk_show]
    n_protos = len(prototypes)

    fig, axes = plt.subplots(n_protos, 5, figsize=(28, 5 * n_protos + 1.5))
    if n_protos == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Prediction: {analysis['prediction']:.2f}    "
        f"(Weighted mean of {len(analysis['prototypes'])} prototype contributions)",
        fontsize=16, fontweight='bold', y=0.98,
    )

    col_titles = [
        'Test Image + Heatmap',
        'Test Image + BBox',
        'Prototype Image',
        'Prototype + Heatmap',
        'Contribution',
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for i, proto in enumerate(prototypes):
        rank = i + 1
        pno = proto['index']
        act_map = proto['upsampled_activation']

        heatmap = plot_heatmap(act_map)
        overlay = np.clip(0.5 * img_display + 0.3 * heatmap, 0, 1)
        axes[i, 0].imshow(overlay)
        crop = find_high_activation_crop(act_map)
        rect = plot_rectangle(crop, edgecolor='cyan')
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_ylabel(f'Rank {rank}\nProto {pno}', fontsize=11, fontweight='bold', rotation=0, labelpad=60, va='center')

        axes[i, 1].imshow(img_display)
        rect2 = plot_rectangle(crop, edgecolor='cyan')
        axes[i, 1].add_patch(rect2)

        proto_img = proto['proto_image'][:, :, ::-1] / 255.0 if proto['proto_image'].max() > 1 else proto['proto_image']
        has_stored_image = proto_img.sum() > 0

        if has_stored_image:
            axes[i, 2].imshow(proto_img)
            proto_heatmap = plot_heatmap(proto['proto_actmap'])
            proto_overlay = np.clip(0.5 * proto_img + 0.3 * proto_heatmap, 0, 1)
            axes[i, 3].imshow(proto_overlay)
        else:
            axes[i, 2].text(0.5, 0.5, 'No pushed\nprototype yet', ha='center', va='center',
                            fontsize=11, transform=axes[i, 2].transAxes, color='gray')
            axes[i, 3].text(0.5, 0.5, 'N/A', ha='center', va='center',
                            fontsize=11, transform=axes[i, 3].transAxes, color='gray')

        info_text = (
            f"Proto label: {proto['proto_label']:.2f}\n"
            f"Similarity: {proto['similarity']:.4f}\n"
            f"Weight (w²/c): {proto['weight']:.4f}\n"
            f"Contribution: {proto['contribution']:.4f}\n"
            f"Share: {proto['pct_contribution']:.1f}%"
        )
        axes[i, 4].text(0.1, 0.5, info_text, ha='left', va='center',
                        fontsize=12, fontfamily='monospace',
                        transform=axes[i, 4].transAxes,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax in axes[:, 4]:
        ax.set_frame_on(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_individual_figures(img_display, analysis, output_dir):
    """Save individual per-prototype visualization images."""
    for i, proto in enumerate(analysis['prototypes']):
        rank = i + 1
        pno = proto['index']
        act_map = proto['upsampled_activation']

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(
            f"Rank {rank} — Prototype {pno} (label={proto['proto_label']:.2f})  |  "
            f"Similarity={proto['similarity']:.4f}  |  "
            f"Contribution={proto['pct_contribution']:.1f}%  |  "
            f"Prediction={analysis['prediction']:.2f}",
            fontsize=12, fontweight='bold',
        )

        heatmap = plot_heatmap(act_map)
        overlay = np.clip(0.5 * img_display + 0.3 * heatmap, 0, 1)
        crop = find_high_activation_crop(act_map)

        axes[0].imshow(img_display)
        rect = plot_rectangle(crop, edgecolor='cyan')
        axes[0].add_patch(rect)
        axes[0].set_title('Test Image + BBox')

        axes[1].imshow(overlay)
        rect2 = plot_rectangle(crop, edgecolor='cyan')
        axes[1].add_patch(rect2)
        axes[1].set_title('Activation Heatmap')

        proto_img = proto['proto_image'][:, :, ::-1] / 255.0 if proto['proto_image'].max() > 1 else proto['proto_image']
        has_stored_image = proto_img.sum() > 0

        if has_stored_image:
            axes[2].imshow(proto_img)
            axes[2].set_title(f'Prototype Image (label={proto["proto_label"]:.2f})')

            proto_heatmap = plot_heatmap(proto['proto_actmap'])
            proto_overlay = np.clip(0.5 * proto_img + 0.3 * proto_heatmap, 0, 1)
            axes[3].imshow(proto_overlay)
            axes[3].set_title('Prototype Activation')
        else:
            axes[2].text(0.5, 0.5, 'No pushed prototype yet', ha='center', va='center', fontsize=11, color='gray')
            axes[3].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=11, color='gray')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'rank{rank}_proto{pno}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize how prototypes see a given input image')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model state_dict (.pth file)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to PyTorch Lightning checkpoint (.ckpt file)')
    parser.add_argument('--param_jsonpath', type=str, required=True,
                        help='Path to parameter JSON file used during training')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the test image')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top prototypes to visualize (default: 10)')
    parser.add_argument('--output_dir', type=str, default='prototype_visualizations',
                        help='Directory to save output visualizations')
    parser.add_argument('--datapath', type=str, default='.',
                        help='Data path (needed for Parameters loading)')
    parser.add_argument('--savepath', type=str, default='.',
                        help='Save path (needed for Parameters loading)')
    parser.add_argument('--individual', action='store_true',
                        help='Also save individual per-prototype figures')

    args = parser.parse_args()

    if args.model_path is None and args.checkpoint_path is None:
        parser.error("Provide either --model_path or --checkpoint_path")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
    print("Loading model...")
    if args.model_path:
        ppnet = load_model_from_state_dict(args.model_path, network_params, device)
        print(f"  Loaded state_dict from: {args.model_path}")
    else:
        ppnet = load_model_from_checkpoint(args.checkpoint_path, params, device)
        print(f"  Loaded checkpoint from: {args.checkpoint_path}")

    print(f"  Architecture: {network_params.base_architecture}")
    print(f"  Num prototypes: {ppnet.num_prototypes}")
    print(f"  Proto labels range: [{ppnet.proto_classes.min():.2f}, {ppnet.proto_classes.max():.2f}]")

    # --- Load and preprocess image ---
    print(f"Loading image: {args.image_path}")
    img_tensor, img_display = load_and_preprocess_image(args.image_path, network_params.img_size)

    # --- Analyze ---
    print(f"Analyzing with top-{args.topk} prototypes...")
    analysis = analyze_image(ppnet, img_tensor, device, topk=args.topk)

    print(f"\n{'='*60}")
    print(f"  Prediction: {analysis['prediction']:.3f}")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Proto':<7} {'Label':<8} {'Similarity':<12} {'Weight':<10} {'Contrib':<10} {'Share':<8}")
    print(f"  {'-'*55}")
    for i, p in enumerate(analysis['prototypes']):
        print(f"  {i+1:<5} {p['index']:<7} {p['proto_label']:<8.2f} {p['similarity']:<12.4f} {p['weight']:<10.4f} {p['contribution']:<10.4f} {p['pct_contribution']:<8.1f}%")
    print(f"{'='*60}\n")

    # --- Save visualizations ---
    os.makedirs(args.output_dir, exist_ok=True)
    image_name = Path(args.image_path).stem

    summary_path = os.path.join(args.output_dir, f'{image_name}_summary.png')
    print(f"Saving summary figure: {summary_path}")
    create_summary_figure(img_display, analysis, summary_path, topk_show=min(args.topk, 10))

    if args.individual:
        indiv_dir = os.path.join(args.output_dir, f'{image_name}_individual')
        os.makedirs(indiv_dir, exist_ok=True)
        print(f"Saving individual figures to: {indiv_dir}/")
        create_individual_figures(img_display, analysis, indiv_dir)

    print("Done!")


if __name__ == '__main__':
    main()