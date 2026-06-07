#!/usr/bin/env python3
"""
Two-panel INSightR-Net prototype figure: original image and red attention overlay.

Usage example::

    cd /path/to/CleverHansRegression
    conda activate new_insight_env

    PYTHONUNBUFFERED=1 python3 prototype_visualization/visualize_prototypes_red_2panel.py \
        --ckpt .../saved_models/Epoch_50_after_protopushing.pth \
        --param_jsonpath config/params_example_ordinal.json \
        --output_dir .../img/prototypes/red_activation_clean/epoch_50 \
        --prototypes 0 7 13 24 31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent

for _p in (_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from define_parameters import NetworkParams  # noqa: E402
from helpers import find_high_activation_crop, load_json  # noqa: E402
from visualize_prototypes_red import (  # noqa: E402
    _create_red_attention_overlay,
    _plot_rectangle,
    autodetect_ref_dir,
    load_ppnet,
    parse_proto_class_map,
)


THESIS_DEFAULT_FIGSIZE = (6.0, 3.6)
THESIS_DEFAULT_DPI = 120
THESIS_DEFAULT_MAX_WIDTH_PX = 1400


def _effective_save_dpi(
    figsize: tuple[float, float],
    requested_dpi: int,
    max_width_px: Optional[int],
) -> int:
    """Cap DPI so the saved PNG width does not exceed *max_width_px*."""
    if max_width_px is None or max_width_px <= 0:
        return requested_dpi
    # Tight bbox is slightly narrower than figsize[0]; 0.92 is a safe estimate.
    width_in = max(figsize[0] * 0.92, 1.0)
    cap = int(max_width_px / width_in)
    return max(72, min(requested_dpi, cap))


def render_prototype_red_2panel(
    proto_img_full_bgr: np.ndarray,
    actmap_lat: np.ndarray,
    bb_info: np.ndarray,
    proto_idx: int,
    savepaths: Union[Path, str, Iterable[Union[Path, str]]],
    figsize=(6.0, 3.6),
    dpi: int = 120,
    max_width_px: Optional[int] = THESIS_DEFAULT_MAX_WIDTH_PX,
    panel_wspace: float = 0.05,
) -> None:
    """Render the 2-panel red-overlay figure for one prototype."""
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
    red_overlay = _create_red_attention_overlay(orig_rgb, upsampled_act)

    fig, ax = plt.subplots(
        1,
        2,
        figsize=figsize,
        facecolor="white",
        gridspec_kw={"wspace": panel_wspace},
    )

    ax[0].imshow(orig_rgb, vmin=0, vmax=1)
    ax[0].axis("off")
    ax[0].set_title("Original Image")
    ax[0].add_patch(_plot_rectangle([lo_y, up_y, lo_x, up_x], edgecolor="b"))

    ax[1].imshow(red_overlay, vmin=0, vmax=1)
    ax[1].axis("off")
    ax[1].set_title("Red Attention Overlay")
    ax[1].add_patch(_plot_rectangle([lo_y, up_y, lo_x, up_x], edgecolor="b"))

    fig.suptitle(f"Prototype: {proto_idx}", fontsize=14, fontweight="normal", y=0.98)
    # Small positive wspace (~5% of axes width) = thin white gap, no overlap.
    fig.subplots_adjust(top=0.84, bottom=0.06, left=0.02, right=0.98, wspace=panel_wspace)

    save_dpi = _effective_save_dpi(figsize, dpi, max_width_px)
    save_kw: dict = dict(dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    # PNG compression (requires Pillow); ignored for PDF.
    save_kw["pil_kwargs"] = {"optimize": True}

    if isinstance(savepaths, (str, Path)):
        savepaths = [savepaths]
    for sp in savepaths:
        sp = Path(sp)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, **save_kw)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render 2-panel 'Prototype: N' red-attention overlays from a "
            "trained INSightR-Net checkpoint (PNG + PDF under output_dir)."
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
        help="Root folder; writes png/ and pdf/ subdirectories.",
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
            "Output filename pattern (without extension). Placeholders: "
            "'{p}' = prototype index, '{c}' = prototype class."
        ),
    )
    p.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        help=(
            "Image formats to write under <output_dir>/<fmt>/. Default: png pdf."
        ),
    )
    p.add_argument(
        "--ref_dir",
        type=str,
        default=None,
        help=(
            "Optional path to training output with prototype_*_class_* filenames "
            "for class lookup. Auto-detected from a sibling red_activation/ folder "
            "when omitted."
        ),
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument(
        "--thesis",
        action="store_true",
        help=(
            "Thesis-friendly export preset (default even without this flag): "
            f"figsize={THESIS_DEFAULT_FIGSIZE}, dpi={THESIS_DEFAULT_DPI}, "
            f"max-width-px={THESIS_DEFAULT_MAX_WIDTH_PX}."
        ),
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=list(THESIS_DEFAULT_FIGSIZE),
        help=f"Figure size in inches (default: {THESIS_DEFAULT_FIGSIZE[0]} {THESIS_DEFAULT_FIGSIZE[1]}).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=THESIS_DEFAULT_DPI,
        help=(
            f"Requested save DPI (default: {THESIS_DEFAULT_DPI}). "
            "Capped by --max-width-px so PNGs stay thesis-sized."
        ),
    )
    p.add_argument(
        "--max-width-px",
        type=int,
        default=THESIS_DEFAULT_MAX_WIDTH_PX,
        help=(
            f"Maximum output PNG width in pixels (default: {THESIS_DEFAULT_MAX_WIDTH_PX}; "
            "0 = no cap). Prevents accidental multi-MB files from high --dpi."
        ),
    )
    p.add_argument(
        "--panel_wspace",
        type=float,
        default=0.05,
        help=(
            "Gap between panels as a fraction of subplot width (matplotlib wspace). "
            "Default 0.05 = small white strip; increase (e.g. 0.08) for wider gap, "
            "decrease (e.g. 0.02) for tighter."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[visualize_prototypes_red_2panel] device={device}", flush=True)

    params_dict = load_json(args.param_jsonpath)
    np_cfg = NetworkParams.from_dict(params_dict.get("network_params", {}))

    ppnet = load_ppnet(args.ckpt, np_cfg, device)
    num_proto = int(ppnet.num_prototypes)
    print(f"[visualize_prototypes_red_2panel] loaded ckpt, P={num_proto}", flush=True)

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
            print(
                f"[visualize_prototypes_red_2panel] auto-detected ref_dir: {ref_dir}",
                flush=True,
            )

    class_map: Dict[int, int] = {}
    if ref_dir is not None:
        class_map = parse_proto_class_map(ref_dir)
        if class_map:
            print(
                f"[visualize_prototypes_red_2panel] loaded class map for "
                f"{len(class_map)} prototypes from {ref_dir}",
                flush=True,
            )
        else:
            print(
                f"[visualize_prototypes_red_2panel] WARNING: no prototype_*_class_*.* "
                f"files in {ref_dir}; falling back to anchor rounding.",
                flush=True,
            )

    formats = [f.lower().lstrip(".") for f in args.formats]
    if not formats:
        raise SystemExit("--formats must list at least one image format.")
    fmt_dirs = {fmt: out_dir / fmt for fmt in formats}
    for d in fmt_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    figsize = tuple(args.figsize)
    max_width_px = int(args.max_width_px) if int(args.max_width_px) > 0 else None
    eff_dpi = _effective_save_dpi(figsize, int(args.dpi), max_width_px)
    print(
        f"[visualize_prototypes_red_2panel] writing to {out_dir} "
        f"(formats: {', '.join(formats)})",
        flush=True,
    )
    print(
        f"[visualize_prototypes_red_2panel] export: figsize={figsize} in, "
        f"dpi={args.dpi} (effective save dpi={eff_dpi}), "
        f"max_width_px={max_width_px or 'none'} "
        f"(~{int(figsize[0] * eff_dpi)} px wide)",
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
                anchor = float(proto_classes_buf[pno])
                cls = int(np.floor(anchor + 0.5))
            else:
                cls = -1
        cls = int(cls)

        stem = stem_template.format(p=pno, c=cls)
        savepaths = [fmt_dirs[fmt] / f"{stem}.{fmt}" for fmt in formats]
        render_prototype_red_2panel(
            proto_img_full_bgr=proto_img,
            actmap_lat=actmap,
            bb_info=bb,
            proto_idx=pno,
            savepaths=savepaths,
            figsize=figsize,
            dpi=int(args.dpi),
            max_width_px=max_width_px,
            panel_wspace=args.panel_wspace,
        )
        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            print(f"  wrote {n_done}/{len(proto_indices)} ...", flush=True)

    print(
        f"[visualize_prototypes_red_2panel] DONE  saved={n_done}  skipped={n_skip}  out={out_dir}"
    )


if __name__ == "__main__":
    main()
