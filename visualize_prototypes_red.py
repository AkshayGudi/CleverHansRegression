#!/usr/bin/env python3
"""
Standalone prototype "red attention" visualizer for INSightR-Net checkpoints.

Regenerates the same kind of red-overlay figure that the training pipeline
writes to ``img/prototypes/red_activation/epoch_<N>/`` (see
``insight_training/push_prototypes.py`` and the second figure block in
``helpers.plot_prototypes``), but with a cleaner layout intended for the
thesis report:

  - Single bold title:        ``Prototype: <n>``
  - Three side-by-side panels:  Original Image | Red Attention Overlay | Prototype Patch
  - No "Prototype Patch" legend on panel 1.
  - No "Prototype Class: ..." or "Class 0: ..." annotation text on the left.
  - Keeps the blue bounding box of the prototype patch on panels 1 and 2.

Inputs are read directly from the checkpoint buffers that prototype pushing
populates:

  - ``ppnet.prototype_images``  (P, H, W, 3) uint8
  - ``ppnet.prototype_actmaps`` (P, h_lat, w_lat) float
  - ``ppnet.proto_bounding_boxes`` (P, 6) with [img_idx, lo_y, up_y, lo_x, up_x, class]

Usage example::

    cd /sc/home/akshay.gudi/code/CleverHansRegression
    source /sc/home/akshay.gudi/conda3/etc/profile.d/conda.sh
    conda activate new_insight_env

    PYTHONUNBUFFERED=1 python3 visualize_prototypes_red.py \
        --ckpt bld_art_25_Apr/class3_v8_fix_50_43/exp1/DR_25_Jan_2026_1/Fold0_DR_01_May_27/saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --output_dir bld_art_25_Apr/class3_v8_fix_50_43/exp1/DR_25_Jan_2026_1/Fold0_DR_01_May_27/img/prototypes/red_activation_clean/epoch_50

    # Only specific prototypes:
    PYTHONUNBUFFERED=1 python3 visualize_prototypes_red.py \
        --ckpt .../Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --output_dir .../red_activation_clean/epoch_50 \
        --prototypes 0 7 13 24 31
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json, find_high_activation_crop  # noqa: E402
from insight_training.model import construct_PPNet  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading (same convention as main_generate_prp.py / relevance_ordering)
# ---------------------------------------------------------------------------


def load_ppnet(ckpt_path: str, network_params: NetworkParams, device: torch.device):
    """Load INSightR-Net weights from a .pth or Lightning .ckpt."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    ppnet = construct_PPNet(network_params=network_params)
    ppnet = ppnet.to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        ppnet_sd = {
            k[len("ppnet."):]: v for k, v in state_dict.items() if k.startswith("ppnet.")
        }
        ppnet.load_state_dict(ppnet_sd, strict=True)
    else:
        ppnet.load_state_dict(checkpoint, strict=True)

    ppnet.eval()
    return ppnet


# ---------------------------------------------------------------------------
# Red attention overlay (same recipe as helpers.plot_prototypes Figure 2)
# ---------------------------------------------------------------------------


def _create_red_attention_overlay(
    original_img: np.ndarray, activation_map: np.ndarray, max_alpha: float = 0.85
) -> np.ndarray:
    act = activation_map - np.amin(activation_map)
    act_max = np.amax(act)
    if act_max > 0:
        act = act / act_max
    else:
        act = np.zeros_like(act)

    intensity = np.sqrt(np.clip(act, 0, 1))
    alpha = (intensity * max_alpha)[:, :, np.newaxis]

    hm_color = np.zeros((act.shape[0], act.shape[1], 3), dtype=np.float32)
    hm_color[:, :, 0] = 1.0
    hm_color[:, :, 1] = intensity * 0.4
    hm_color[:, :, 2] = 0.0

    dim_factor = 1.0 - 0.3 * (1.0 - intensity[:, :, np.newaxis])
    dimmed_original = original_img * dim_factor
    overlay = (1 - alpha) * dimmed_original + alpha * hm_color
    return np.clip(overlay, 0, 1)


def _plot_rectangle(box_coords, edgecolor: str = "b"):
    """Same convention as ``helpers.plot_rectangle``.

    ``box_coords`` MUST be ``[lower_y, upper_y, lower_x, upper_x]`` (4 values).
    Do NOT prepend the image-index. The training code calls this with
    ``proto_bound_j[1:]``; this script must do the same.
    """
    rect = patches.Rectangle(
        (box_coords[2], box_coords[1]),
        box_coords[3] - box_coords[2],
        -box_coords[1] + box_coords[0],
        linewidth=1.5,
        facecolor="none",
        edgecolor=edgecolor,
    )
    return rect


# ---------------------------------------------------------------------------
# Class lookup from existing training output
# ---------------------------------------------------------------------------


_PROTO_FNAME_RE = re.compile(r"prototype_(\d+)_class_(-?\d+)\.(?:png|svg|pdf|jpg|jpeg)$")


def parse_proto_class_map(ref_dir: Path) -> Dict[int, int]:
    """Scan a directory of training-style filenames and return ``{proto_idx: class}``.

    Recognises names like ``prototype_<n>_class_<c>.png`` (any image extension).
    """
    mapping: Dict[int, int] = {}
    if not ref_dir.is_dir():
        return mapping
    for f in sorted(ref_dir.iterdir()):
        m = _PROTO_FNAME_RE.match(f.name)
        if m:
            mapping[int(m.group(1))] = int(m.group(2))
    return mapping


def autodetect_ref_dir(out_dir: Path) -> Optional[Path]:
    """Find a likely training output folder with ``prototype_*_class_*.png`` files.

    Heuristics, tried in order:
      1. Replace the first ``red_activation_clean`` component of ``out_dir`` with
         ``red_activation`` (e.g. ``.../red_activation_clean/epoch_50`` ->
         ``.../red_activation/epoch_50``).
      2. Look at ``<grandparent>/red_activation/<basename>``.
    Returns the first existing directory, or ``None``.
    """
    parts = list(out_dir.parts)
    candidates: List[Path] = []
    for i, p in enumerate(parts):
        if p == "red_activation_clean":
            new_parts = parts.copy()
            new_parts[i] = "red_activation"
            candidates.append(Path(*new_parts))
            break
    grand = out_dir.parent.parent
    candidates.append(grand / "red_activation" / out_dir.name)
    for c in candidates:
        if c.is_dir():
            return c
    return None


# ---------------------------------------------------------------------------
# One prototype -> one PNG
# ---------------------------------------------------------------------------


def render_prototype_red(
    proto_img_full_bgr: np.ndarray,
    actmap_lat: np.ndarray,
    bb_info: np.ndarray,
    proto_idx: int,
    savepaths,
    figsize=(15, 4.4),
    dpi: int = 150,
) -> None:
    """Render the 3-panel red-overlay figure for one prototype.

    Parameters
    ----------
    proto_img_full_bgr : (H, W, 3) uint8 array, BGR (as stored in the model).
    actmap_lat         : (h_lat, w_lat) float, prototype activation map.
    bb_info            : (6,) array [img_idx, lo_y, up_y, lo_x, up_x, class].
                         Pass ``None`` to recompute from ``actmap_lat``.
    proto_idx          : Prototype index (used in the title and filename).
    savepaths          : One :class:`pathlib.Path` or an iterable of paths.
                         The figure is rendered once and written to every
                         path; the extension of each path controls the format
                         (e.g. ``.png`` and ``.svg``).
    """
    H, W = proto_img_full_bgr.shape[:2]

    orig_rgb = proto_img_full_bgr[:, :, ::-1].astype(np.float32) / 255.0

    upsampled_act = cv2.resize(
        actmap_lat.astype(np.float32),
        dsize=(W, H),
        interpolation=cv2.INTER_CUBIC,
    )

    if bb_info is None or int(bb_info[1]) == int(bb_info[2]) or int(bb_info[3]) == int(bb_info[4]):
        bb_yxyx = find_high_activation_crop(upsampled_act)
    else:
        bb_yxyx = (int(bb_info[1]), int(bb_info[2]), int(bb_info[3]), int(bb_info[4]))

    lo_y, up_y, lo_x, up_x = bb_yxyx
    proto_crop_rgb = orig_rgb[lo_y:up_y, lo_x:up_x]

    red_overlay = _create_red_attention_overlay(orig_rgb, upsampled_act)

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    ax[0].imshow(orig_rgb, vmin=0, vmax=1)
    ax[0].axis("off")
    ax[0].set_title("Original Image")
    ax[0].add_patch(_plot_rectangle([lo_y, up_y, lo_x, up_x], edgecolor="b"))

    ax[1].imshow(red_overlay, vmin=0, vmax=1)
    ax[1].axis("off")
    ax[1].set_title("Red Attention Overlay")
    ax[1].add_patch(_plot_rectangle([lo_y, up_y, lo_x, up_x], edgecolor="b"))

    if proto_crop_rgb.size == 0:
        ax[2].axis("off")
        ax[2].set_title("Prototype Patch")
    else:
        ax[2].imshow(proto_crop_rgb, vmin=0, vmax=1)
        ax[2].axis("off")
        ax[2].set_title("Prototype Patch")

    fig.suptitle(f"Prototype: {proto_idx}", fontsize=18, fontweight="normal", y=0.98)
    fig.subplots_adjust(top=0.84, left=0.04, right=0.98, wspace=0.05)

    if isinstance(savepaths, (str, Path)):
        savepaths = [savepaths]
    for sp in savepaths:
        sp = Path(sp)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render simplified 'Prototype: N' red-attention overlays from a "
            "trained INSightR-Net checkpoint (no prototype pushing or training)."
        )
    )
    p.add_argument("--ckpt", required=True, type=str, help="Path to *_after_protopushing.pth")
    p.add_argument(
        "--param_jsonpath",
        required=True,
        type=str,
        help="JSON used at training time (e.g. config/params_example_ordinal.json).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Folder to write prototype_<n>_red.png files into.",
    )
    p.add_argument(
        "--prototypes",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of prototype indices. Default: all stored prototypes.",
    )
    p.add_argument(
        "--filename_format",
        type=str,
        default="prototype_{p}_class_{c}",
        help=(
            "Output filename pattern (without extension). Available "
            "placeholders: '{p}' = prototype index, '{c}' = prototype class. "
            "Default mirrors the training output stem (e.g. "
            "prototype_0_class_1). Files are written as "
            "<output_dir>/<fmt>/<stem>.<fmt> for each chosen format."
        ),
    )
    p.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["png", "svg"],
        help=(
            "Image formats to write. For each format <fmt>, files are saved "
            "under <output_dir>/<fmt>/. Default: png svg."
        ),
    )
    p.add_argument(
        "--ref_dir",
        type=str,
        default=None,
        help=(
            "Optional path to the original training output folder "
            "(e.g. .../img/prototypes/red_activation/epoch_50) used to read "
            "the actual class of each prototype from existing filenames. "
            "If omitted, the script auto-detects a sibling 'red_activation/<epoch>' "
            "folder; if none is found, classes fall back to the rounded "
            "regression anchor (which may disagree with what the prototype was "
            "actually pushed to)."
        ),
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[visualize_prototypes_red] device={device}", flush=True)

    params_dict = load_json(args.param_jsonpath)
    np_cfg = NetworkParams.from_dict(params_dict.get("network_params", {}))

    ppnet = load_ppnet(args.ckpt, np_cfg, device)
    num_proto = int(ppnet.num_prototypes)
    print(f"[visualize_prototypes_red] loaded ckpt, P={num_proto}", flush=True)

    proto_imgs = ppnet.prototype_images.detach().cpu().numpy()
    proto_acts = ppnet.prototype_actmaps.detach().cpu().numpy()
    bb_all = ppnet.proto_bounding_boxes.detach().cpu().numpy()

    proto_classes_buf = None
    if hasattr(ppnet, "proto_classes") and ppnet.proto_classes is not None:
        proto_classes_buf = ppnet.proto_classes.detach().cpu().numpy()

    if proto_imgs.max() == 0:
        raise SystemExit(
            "prototype_images is all zeros — the checkpoint does not contain "
            "pushed prototypes. Use a *_after_protopushing.pth file."
        )

    if args.prototypes is None:
        proto_indices: List[int] = list(range(num_proto))
    else:
        proto_indices = list(dict.fromkeys(args.prototypes))
        bad = [p for p in proto_indices if p < 0 or p >= num_proto]
        if bad:
            raise SystemExit(
                f"Prototype indices {bad} out of range [0, {num_proto - 1}]."
            )

    out_dir = Path(args.output_dir)

    ref_dir: Optional[Path] = None
    if args.ref_dir is not None:
        ref_dir = Path(args.ref_dir)
        if not ref_dir.is_dir():
            raise SystemExit(f"--ref_dir does not exist: {ref_dir}")
    else:
        ref_dir = autodetect_ref_dir(out_dir)
        if ref_dir is not None:
            print(f"[visualize_prototypes_red] auto-detected ref_dir: {ref_dir}", flush=True)

    class_map: Dict[int, int] = {}
    if ref_dir is not None:
        class_map = parse_proto_class_map(ref_dir)
        if class_map:
            print(
                f"[visualize_prototypes_red] loaded class map for "
                f"{len(class_map)} prototypes from {ref_dir}",
                flush=True,
            )
        else:
            print(
                f"[visualize_prototypes_red] WARNING: no prototype_*_class_*.* "
                f"files in {ref_dir}; falling back to anchor rounding.",
                flush=True,
            )

    formats = [f.lower().lstrip(".") for f in args.formats]
    if not formats:
        raise SystemExit("--formats must list at least one image format.")
    fmt_dirs = {fmt: out_dir / fmt for fmt in formats}
    for d in fmt_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(
        f"[visualize_prototypes_red] writing to {out_dir} "
        f"(formats: {', '.join(formats)})",
        flush=True,
    )

    stem_template = args.filename_format
    if "." in Path(stem_template).suffix:
        stem_template = str(Path(stem_template).with_suffix(""))

    n_done, n_skip = 0, 0
    for pno in proto_indices:
        proto_img = proto_imgs[pno]
        if proto_img.max() == 0:
            print(f"  prototype {pno}: empty (no stored image), skipping.")
            n_skip += 1
            continue

        actmap = proto_acts[pno]
        bb = bb_all[pno] if bb_all is not None else None

        cls = class_map.get(pno, None)
        if cls is None:
            if bb is not None and int(bb[-1]) > 0:
                cls = int(bb[-1])
            elif proto_classes_buf is not None:
                # proto_classes is a linspace(min, max, P) of regression anchors.
                # Use floor(a + 0.5) so the boundary 0.5 rounds up to class 1.
                anchor = float(proto_classes_buf[pno])
                cls = int(np.floor(anchor + 0.5))
            else:
                cls = -1
        cls = int(cls)

        stem = stem_template.format(p=pno, c=cls)
        savepaths = [fmt_dirs[fmt] / f"{stem}.{fmt}" for fmt in formats]
        render_prototype_red(
            proto_img_full_bgr=proto_img,
            actmap_lat=actmap,
            bb_info=bb,
            proto_idx=pno,
            savepaths=savepaths,
            dpi=args.dpi,
        )
        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            print(f"  wrote {n_done}/{len(proto_indices)} ...", flush=True)

    print(
        f"[visualize_prototypes_red] DONE  saved={n_done}  skipped={n_skip}  out={out_dir}"
    )


if __name__ == "__main__":
    main()
