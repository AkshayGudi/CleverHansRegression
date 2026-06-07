# small PNGs for thesis report

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

# Shown as one LaTeX subfigure (full multi-panel strip at ~0.47 textwidth).
THESIS_MAX_WIDTH_PX = 1400
THESIS_DPI = 120

# 3-panel: Original | Activation | PRP  (img/prp in thesis)
THESIS_3PANEL_FIGSIZE = (10.5, 3.5)

# 4-panel: Original | Activation | PRP | Focus PRP  (img/focus_prp in thesis)
THESIS_4PANEL_FIGSIZE = (14.0, 3.5)


@dataclass
class ThesisFigureExport:
    figsize: tuple[float, float]
    dpi: int = THESIS_DPI
    max_width_px: int = THESIS_MAX_WIDTH_PX
    panel_title_fontsize: int = 12
    suptitle_fontsize: int = 14

    @classmethod
    def three_panel(cls, **kwargs) -> "ThesisFigureExport":
        return cls(figsize=THESIS_3PANEL_FIGSIZE, **kwargs)

    @classmethod
    def four_panel(cls, **kwargs) -> "ThesisFigureExport":
        return cls(figsize=THESIS_4PANEL_FIGSIZE, **kwargs)


THESIS_3PANEL = ThesisFigureExport.three_panel()
THESIS_4PANEL = ThesisFigureExport.four_panel()


def effective_save_dpi(
    figsize: tuple[float, float],
    requested_dpi: int,
    max_width_px: Optional[int],
) -> int:
    if max_width_px is None or max_width_px <= 0:
        return requested_dpi
    width_in = max(figsize[0] * 0.92, 1.0)
    cap = int(max_width_px / width_in)
    return max(72, min(requested_dpi, cap))


def add_thesis_export_args(parser, *, panel_count: int = 3) -> None:
    """Register CLI flags shared by PRP / PLRP / compare scripts."""
    default_fs = THESIS_3PANEL_FIGSIZE if panel_count == 3 else THESIS_4PANEL_FIGSIZE
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=list(default_fs),
        help=f"Figure size in inches (default: {default_fs[0]} {default_fs[1]}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=THESIS_DPI,
        help=f"Requested save DPI, capped for thesis (default: {THESIS_DPI}).",
    )
    parser.add_argument(
        "--max-width-px",
        type=int,
        default=THESIS_MAX_WIDTH_PX,
        help=f"Max output PNG width in pixels (default: {THESIS_MAX_WIDTH_PX}; 0 = no cap).",
    )


def export_config_from_args(args, *, panel_count: int = 3) -> ThesisFigureExport:
    fs = tuple(getattr(args, "figsize", THESIS_3PANEL_FIGSIZE if panel_count == 3 else THESIS_4PANEL_FIGSIZE))
    dpi = int(getattr(args, "dpi", THESIS_DPI))
    mw = int(getattr(args, "max_width_px", THESIS_MAX_WIDTH_PX))
    if mw <= 0:
        mw = 0
    if panel_count == 3:
        return ThesisFigureExport(figsize=fs, dpi=dpi, max_width_px=mw or THESIS_MAX_WIDTH_PX)
    return ThesisFigureExport(figsize=fs, dpi=dpi, max_width_px=mw or THESIS_MAX_WIDTH_PX)


def save_figure(
    fig,
    path: Union[str, Path],
    cfg: ThesisFigureExport,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
) -> int:
    """Save one figure; returns effective DPI used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dpi = effective_save_dpi(cfg.figsize, cfg.dpi, cfg.max_width_px)
    kw = dict(dpi=save_dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    if path.suffix.lower() == ".png":
        kw["pil_kwargs"] = {"optimize": True}
    fig.savefig(str(path), **kw)
    return save_dpi


def save_figure_multi(
    fig,
    out_dir: Union[str, Path],
    basename: str,
    cfg: ThesisFigureExport,
    formats: Sequence[str] = ("png", "svg"),
) -> int:
    """Save under ``<out_dir>/<fmt>/<basename>.<fmt>``; returns effective DPI."""
    out_dir = Path(out_dir)
    save_dpi = effective_save_dpi(cfg.figsize, cfg.dpi, cfg.max_width_px)
    for fmt in formats:
        sub = out_dir / fmt
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{basename}.{fmt.lstrip('.')}"
        kw = dict(dpi=save_dpi, bbox_inches="tight")
        if fmt.lower() == "png":
            kw["pil_kwargs"] = {"optimize": True}
        fig.savefig(str(path), **kw)
    return save_dpi


def log_export_settings(cfg: ThesisFigureExport, label: str = "figure export") -> None:
    eff = effective_save_dpi(cfg.figsize, cfg.dpi, cfg.max_width_px)
    approx_w = int(cfg.figsize[0] * eff * 0.92)
    print(
        f"[{label}] figsize={cfg.figsize} in, dpi={cfg.dpi} "
        f"(effective save dpi={eff}), max_width_px={cfg.max_width_px} "
        f"(~{approx_w} px wide)",
        flush=True,
    )
