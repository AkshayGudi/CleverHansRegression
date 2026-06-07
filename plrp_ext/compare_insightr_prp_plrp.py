"""
plrp_ext/compare_insightr_prp_plrp.py
=====================================

Side-by-side **4-panel** visualization for each prototype (or each prototype
on a test image):

    Panel 1: Original image
    Panel 2: INSightR-Net activation overlay 
    Panel 3: Baseline PRP relevance overlay
    Panel 4: PLRP-PRP relevance overlay

Usage commands:
-----
    cd /path/to/CleverHansRegression
    conda activate new_insight_env

    # 1) Stored prototype mode (uses each prototype's saved image + actmap):
    python3 plrp_ext/compare_insightr_prp_plrp.py \\
        --model_path .../Epoch_50_after_protopushing.pth \\
        --param_jsonpath config/params_example_ordinal.json \\
        --prototypes 22 27 33 \\
        --plrp_p_pos 0.25 --plrp_p_neg 0.125 \\
        --output_dir plrp_ext/outputs/three_way/class3_v14_2_p025

    # 2) Test image mode (computes the activation map on the fly):
    python3 plrp_ext/compare_insightr_prp_plrp.py \\
        --model_path .../Epoch_50_after_protopushing.pth \\
        --param_jsonpath config/params_example_ordinal.json \\
        --test_image_path /path/to/test.jpeg \\
        --prototypes 22 27 33 \\
        --plrp_p_pos 0.25 --plrp_p_neg 0.125 \\
        --output_dir plrp_ext/outputs/three_way_testimg/class3_v14_2_p025

    # 3) Sanity check (p_pos=p_neg=0 should make panel 4 == panel 3 exactly):
    python3 plrp_ext/compare_insightr_prp_plrp.py \\
        --model_path .../Epoch_50_after_protopushing.pth \\
        --param_jsonpath config/params_example_ordinal.json \\
        --prototypes 22 \\
        --plrp_p_pos 0.0 --plrp_p_neg 0.0 \\
        --output_dir plrp_ext/outputs/three_way_sanity
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from helpers import load_json
from define_parameters import NetworkParams
from insight_training.model import construct_PPNet

from insight_prp import (
    PRPCanonizedModel as build_baseline_canon,
    generate_prp_image as generate_prp_baseline,
    _create_overlay as create_prp_red_overlay,
)

from plrp_ext.lrp_general6_plrp import set_plrp_params, get_plrp_params
from plrp_ext.insight_prp_plrp import (
    PRPCanonizedModel as build_plrp_canon,
    generate_prp_image as generate_prp_plrp,
    save_img_multi,
)

_IMG_FORMATS = ("png", "svg")

from thesis_figure_export import (
    ThesisFigureExport,
    THESIS_4PANEL,
    add_thesis_export_args,
    export_config_from_args,
    log_export_settings,
    save_figure,
    save_figure_multi,
)

_FIG_EXPORT: ThesisFigureExport = THESIS_4PANEL


# ---------------------------------------------------------------------------
# Checkpoint and image I/O
# ---------------------------------------------------------------------------

def load_ppnet_from_checkpoint(model_path, network_params, device):
    """Load a *fresh* PPNet instance from a .pth or Lightning .ckpt file.

    A fresh instance is essential: ``PRPCanonizedModel`` mutates the model in
    place, so re-using the same instance for both pipelines would corrupt the
    other.
    """
    checkpoint = torch.load(model_path, map_location=device)
    ppnet = construct_PPNet(network_params=network_params)
    ppnet = ppnet.to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        ppnet_state_dict = {
            k[len("ppnet."):]: v for k, v in state_dict.items() if k.startswith("ppnet.")
        }
        ppnet.load_state_dict(ppnet_state_dict, strict=True)
    else:
        ppnet.load_state_dict(checkpoint, strict=True)

    ppnet.eval()
    return ppnet


def load_test_image(image_path, img_size):
    """Preprocess a test image identically to main_generate_prp.py.

    The model expects BGR input (the original training pipeline used
    ``cv2.imread`` directly). For visualization we return a separate
    RGB-converted copy in [0, 1].
    """
    jpeg_im = cv2.imread(str(image_path))
    if jpeg_im is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if jpeg_im.shape[0] != img_size or jpeg_im.shape[1] != img_size:
        jpeg_im = cv2.resize(jpeg_im, (img_size, img_size))

    norm = jpeg_im / 255.0
    # HWC -> CHW with the correct permutation (HWC -> CHW is (2, 0, 1),
    # NOT (2, 1, 0) which silently swaps the spatial axes).
    img_tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
    raw_rgb_np = cv2.cvtColor(jpeg_im, cv2.COLOR_BGR2RGB) / 255.0
    return img_tensor, raw_rgb_np.astype(np.float32)


def _resolve_prototype_indices(prototype_number, prototypes_list):
    if prototypes_list is not None:
        return list(dict.fromkeys(prototypes_list))
    if prototype_number is not None:
        return [prototype_number]
    return None


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def create_insightr_activation_overlay(orig_img_np, actmap_np, max_alpha=0.85):
    """Red/yellow activation overlay for the INSightR-Net activation map.

      * High activation -> bright red/yellow
      * Low activation  -> dimmed original image
    """
    h, w = orig_img_np.shape[:2]
    upsampled = cv2.resize(actmap_np, (w, h), interpolation=cv2.INTER_CUBIC)
    amin, amax = float(upsampled.min()), float(upsampled.max())
    if amax - amin > 0:
        upsampled = (upsampled - amin) / (amax - amin)
    else:
        upsampled = np.zeros_like(upsampled)

    intensity = np.sqrt(np.clip(upsampled, 0, 1))
    alpha = (intensity * max_alpha)[:, :, np.newaxis]

    hm_color = np.zeros((h, w, 3), dtype=np.float32)
    hm_color[:, :, 0] = 1.0                  # full red channel
    hm_color[:, :, 1] = intensity * 0.4      # yellow tint scales with intensity
    hm_color[:, :, 2] = 0.0                  # no blue

    dim_factor = 1.0 - 0.3 * (1.0 - intensity[:, :, np.newaxis])
    dimmed_original = orig_img_np * dim_factor
    overlay = (1 - alpha) * dimmed_original + alpha * hm_color
    return np.clip(overlay, 0, 1)


def _build_four_panel_fig(orig_img, insightr_overlay, baseline_overlay, plrp_overlay,
                          pno, p_pos, p_neg,
                          export_cfg: ThesisFigureExport | None = None):
    """Build (but do not save) the four-panel comparison figure."""
    cfg = export_cfg or _FIG_EXPORT
    fig, axes = plt.subplots(1, 4, figsize=cfg.figsize)
    fs = cfg.panel_title_fontsize

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=fs, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(insightr_overlay)
    axes[1].set_title("INSightR-Net Activation", fontsize=fs, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(baseline_overlay)
    axes[2].set_title("Baseline PRP", fontsize=fs, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(plrp_overlay)
    axes[3].set_title(
        f"PLRP-PRP  (p_pos={p_pos}, p_neg={p_neg})",
        fontsize=fs, fontweight="bold",
    )
    axes[3].axis("off")

    fig.suptitle(
        f"Prototype {pno}  -  INSightR-Net  ->  Baseline PRP  ->  PLRP-PRP",
        fontsize=cfg.suptitle_fontsize, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def save_four_panel(orig_img, insightr_overlay, baseline_overlay, plrp_overlay,
                    save_path, pno, p_pos, p_neg,
                    export_cfg: ThesisFigureExport | None = None):
    """Save a four-panel comparison as PNG + SVG siblings of ``save_path`` (flat)."""
    cfg = export_cfg or _FIG_EXPORT
    fig = _build_four_panel_fig(orig_img, insightr_overlay, baseline_overlay,
                                plrp_overlay, pno, p_pos, p_neg, export_cfg=cfg)
    sp = Path(save_path)
    for fmt in _IMG_FORMATS:
        save_figure(fig, sp.with_suffix("." + fmt), cfg)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core 3-way generation per prototype
# ---------------------------------------------------------------------------

def compute_insightr_actmap_for_test_image(ppnet_orig, img_tensor, pno, device):
    """Compute INSightR-Net activation map for prototype ``pno`` on a test image.

    Uses the **uncanonized** ppnet, calling ``forward_testtime`` which returns
    ``img_activations`` of shape (batch, num_proto, H_feat, W_feat).
    """
    ppnet_orig.train(False)
    with torch.no_grad():
        x = img_tensor.to(device)
        _, _, _, _, img_activations = ppnet_orig.forward_testtime(x)
    return img_activations[0, pno].detach().cpu().numpy()


def run_stored_prototype(pno, ppnet_baseline_canon, ppnet_plrp_canon,
                         device, output_dir, p_pos, p_neg):
    """Run all three explanations for one stored prototype.

    Returns True on success, False if the prototype has no stored image
    """
    proto_img_tensor = ppnet_baseline_canon.prototype_images[pno]
    if proto_img_tensor.max() == 0:
        print(f"Prototype {pno}: no stored image (push_prototypes has not seen it). Skipping.")
        return False

    # Stored prototype image is (H, W, C) uint8. Convert to (1, 3, H, W) float in [0, 1].
    img_float = proto_img_tensor.float() / 255.0
    img_tensor = img_float.permute(2, 0, 1).unsqueeze(0)
    orig_img_np = proto_img_tensor.detach().cpu().numpy() / 255.0

    # Pre-stored INSightR-Net activation map (captured at prototype-push time).
    actmap = ppnet_baseline_canon.prototype_actmaps[pno].detach().cpu().numpy()

    baseline_heatmap = generate_prp_baseline(img_tensor, pno, ppnet_baseline_canon, device)
    baseline_overlay = create_prp_red_overlay(orig_img_np, baseline_heatmap)

    plrp_heatmap = generate_prp_plrp(img_tensor, pno, ppnet_plrp_canon, device)
    plrp_overlay = create_prp_red_overlay(orig_img_np, plrp_heatmap)

    insightr_overlay = create_insightr_activation_overlay(orig_img_np, actmap)

    proto_dir = output_dir / f"prototype_{pno}"
    proto_dir.mkdir(parents=True, exist_ok=True)

    save_img_multi(proto_dir, "original", orig_img_np)
    save_img_multi(proto_dir, "insightr_activation_overlay", insightr_overlay, vmin=0, vmax=1)
    save_img_multi(proto_dir, "baseline_prp_overlay", baseline_overlay, vmin=0, vmax=1)
    save_img_multi(proto_dir, "plrp_prp_overlay", plrp_overlay, vmin=0, vmax=1)
    save_img_multi(proto_dir, "baseline_prp_heatmap", baseline_heatmap,
                   cmap="seismic", vmin=-1, vmax=1)
    save_img_multi(proto_dir, "plrp_prp_heatmap", plrp_heatmap,
                   cmap="seismic", vmin=-1, vmax=1)

    fig = _build_four_panel_fig(orig_img_np, insightr_overlay, baseline_overlay,
                                plrp_overlay, pno, p_pos, p_neg)
    save_figure_multi(fig, proto_dir, "comparison_4panel", _FIG_EXPORT, formats=_IMG_FORMATS)
    plt.close(fig)
    print(f"Prototype {pno}: saved to {proto_dir}/ (png/ and svg/ subfolders)")
    return True


def run_test_image(img_tensor, raw_rgb_np, prototype_indices,
                   ppnet_orig, ppnet_baseline_canon, ppnet_plrp_canon,
                   device, output_dir, p_pos, p_neg):
    """Run all three explanations for each prototype on a single test image."""
    if prototype_indices is None:
        prototype_indices = list(range(ppnet_baseline_canon.num_prototypes))

    for pno in prototype_indices:
        actmap = compute_insightr_actmap_for_test_image(ppnet_orig, img_tensor, pno, device)
        baseline_heatmap = generate_prp_baseline(img_tensor, pno, ppnet_baseline_canon, device)
        baseline_overlay = create_prp_red_overlay(raw_rgb_np, baseline_heatmap)
        plrp_heatmap = generate_prp_plrp(img_tensor, pno, ppnet_plrp_canon, device)
        plrp_overlay = create_prp_red_overlay(raw_rgb_np, plrp_heatmap)
        insightr_overlay = create_insightr_activation_overlay(raw_rgb_np, actmap)

        for _fmt in _IMG_FORMATS:
            plt.imsave(str(output_dir / f"proto_{pno}_original.{_fmt}"), raw_rgb_np)
            plt.imsave(str(output_dir / f"proto_{pno}_insightr_activation_overlay.{_fmt}"),
                       insightr_overlay, vmin=0, vmax=1)
            plt.imsave(str(output_dir / f"proto_{pno}_baseline_prp_overlay.{_fmt}"),
                       baseline_overlay, vmin=0, vmax=1)
            plt.imsave(str(output_dir / f"proto_{pno}_plrp_prp_overlay.{_fmt}"),
                       plrp_overlay, vmin=0, vmax=1)
        save_four_panel(raw_rgb_np, insightr_overlay, baseline_overlay, plrp_overlay,
                        output_dir / f"proto_{pno}_comparison_4panel.png",
                        pno, p_pos, p_neg)
        print(f"Prototype {pno}: saved 4-panel (png+svg) to "
              f"{output_dir}/proto_{pno}_comparison_4panel.*")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-way side-by-side comparison: "
                    "INSightR-Net activation | Baseline PRP | PLRP-PRP."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to trained model (.pth from saved_models/ or Lightning .ckpt).")
    parser.add_argument("--param_jsonpath", required=True,
                        help="Path to network parameters JSON (e.g. config/params_example_ordinal.json).")
    parser.add_argument("--output_dir", default="plrp_ext/outputs/compare_3way",
                        help="Directory to write per-prototype outputs.")
    parser.add_argument("--prototype_number", type=int, default=None,
                        help="Single prototype index. Mutually exclusive with --prototypes.")
    parser.add_argument("--prototypes", type=int, nargs="+", default=None,
                        help="List of prototype indices. Mutually exclusive with --prototype_number.")
    parser.add_argument("--test_image_path", type=str, default=None,
                        help="Optional: path to a test image. If omitted, uses stored prototype images.")
    parser.add_argument("--plrp_p_pos", type=float, default=0.25,
                        help="PLRP-lambda: proportion of positive relevance to prune. Default 0.25.")
    parser.add_argument("--plrp_p_neg", type=float, default=0.125,
                        help="PLRP-lambda: proportion of negative relevance to prune. Default 0.125.")
    add_thesis_export_args(parser, panel_count=4)

    args = parser.parse_args()

    global _FIG_EXPORT
    _FIG_EXPORT = export_config_from_args(args, panel_count=4)
    log_export_settings(_FIG_EXPORT, "compare_insightr_prp_plrp")

    if args.prototype_number is not None and args.prototypes is not None:
        parser.error("Use only one of --prototype_number and --prototypes.")
    if not 0.0 <= args.plrp_p_pos < 1.0:
        parser.error("--plrp_p_pos must be in [0, 1).")
    if not 0.0 <= args.plrp_p_neg < 1.0:
        parser.error("--plrp_p_neg must be in [0, 1).")

    # Configure PLRP state. Only the plrp_ext.lrp_general6_plrp wrappers
    # consume this state, so the baseline pipeline is unaffected.
    set_plrp_params(p_pos=args.plrp_p_pos, p_neg=args.plrp_p_neg)
    eff_p_pos, eff_p_neg = get_plrp_params()
    print(f"PLRP-lambda state: p_pos={eff_p_pos}, p_neg={eff_p_neg}")
    if eff_p_pos == 0.0 and eff_p_neg == 0.0:
        print("  -> PLRP DISABLED; PLRP panel should equal baseline PRP panel exactly.")
    else:
        print("  -> PLRP ENABLED; baseline panel is unaffected by this state.")

    proto_indices = _resolve_prototype_indices(args.prototype_number, args.prototypes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    params_dict = load_json(args.param_jsonpath)
    network_params_dict = params_dict.get("network_params", {})
    network_params = NetworkParams.from_dict(network_params_dict)
    print(f"Network config: {network_params.base_architecture}, "
          f"proto_shape={network_params.proto_shape}, "
          f"img_size={network_params.img_size}")

    # Load three INDEPENDENT model instances because one is needed for INSightR-Net activations,
    # one for baseline PRP, and one for PLRP-PRP.
    print("Loading three PPNet instances:")
    print("  1/3 -> ppnet_orig          (uncanonized; for INSightR-Net activations)")
    ppnet_orig = load_ppnet_from_checkpoint(args.model_path, network_params, device)
    ppnet_orig.base_architecture = network_params.base_architecture

    print("  2/3 -> ppnet_baseline      (will be canonized for baseline PRP)")
    ppnet_baseline = load_ppnet_from_checkpoint(args.model_path, network_params, device)
    ppnet_baseline.base_architecture = network_params.base_architecture

    print("  3/3 -> ppnet_plrp          (will be canonized for PLRP-PRP)")
    ppnet_plrp = load_ppnet_from_checkpoint(args.model_path, network_params, device)
    ppnet_plrp.base_architecture = network_params.base_architecture

    print("Canonizing baseline (insight_prp.PRPCanonizedModel)...")
    ppnet_baseline_canon = build_baseline_canon(ppnet_baseline).to(device)
    print("Canonizing PLRP (plrp_ext.insight_prp_plrp.PRPCanonizedModel)...")
    ppnet_plrp_canon = build_plrp_canon(ppnet_plrp).to(device)
    print("Canonization complete.\n")

    if proto_indices is not None:
        n = ppnet_baseline_canon.num_prototypes
        invalid = [p for p in proto_indices if p < 0 or p >= n]
        if invalid:
            raise ValueError(
                f"Invalid prototype index(es) {invalid}; valid range is 0..{n - 1}."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test_image_path is not None:
        print(f"Mode: test image -> {args.test_image_path}")
        img_tensor, raw_rgb_np = load_test_image(
            args.test_image_path, img_size=network_params.img_size
        )
        run_test_image(img_tensor, raw_rgb_np, proto_indices,
                       ppnet_orig, ppnet_baseline_canon, ppnet_plrp_canon,
                       device, output_dir, eff_p_pos, eff_p_neg)
        print(f"\nDone. Outputs in {output_dir}/")
    else:
        if proto_indices is None:
            proto_indices = list(range(ppnet_baseline_canon.num_prototypes))
        print(f"Mode: stored prototypes -> {len(proto_indices)} prototype(s)")
        saved = 0
        for pno in proto_indices:
            if run_stored_prototype(pno, ppnet_baseline_canon, ppnet_plrp_canon,
                                    device, output_dir, eff_p_pos, eff_p_neg):
                saved += 1
        if saved == 0:
            print("No prototypes were saved. Either run prototype pushing first, "
                  "or pass --test_image_path.")
        else:
            print(f"\nDone. Saved {saved} prototype 4-panel comparison(s) to {output_dir}/")


if __name__ == "__main__":
    main()
