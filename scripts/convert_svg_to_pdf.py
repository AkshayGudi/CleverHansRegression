#!/usr/bin/env python3
"""
Convert SVG files in a folder to PDF (vector output for LaTeX / Overleaf).

Uses svglib + reportlab (install in conda env new_insight_env):
  conda activate new_insight_env
  pip install reportlab lxml tinycss2 cssselect2
  pip install svglib --no-deps

Examples
--------
  cd /sc/home/akshay.gudi/code/CleverHansRegression
  conda activate new_insight_env

  # All *.svg in a folder → PDFs with the same basename
  python3 scripts/convert_svg_to_pdf.py \\
    --input_dir bld_art_25_Apr/class3_v14_fix_43/exp1/DR_25_Jan_2026_1/Fold0_DR_03_May_3/img/prototypes/red_activation_clean_2/epoch_50/svg \\
    --output_dir /sc/home/akshay.gudi/proto_images/class3_v14_fix

  # Only class-3 prototype SVGs
  python3 scripts/convert_svg_to_pdf.py \\
    --input_dir /path/to/svg \\
    --output_dir /path/to/pdf_out \\
    --glob '*_class_3.svg'

  # Specific prototype indices (matches prototype_{id}_class_3.svg)
  python3 scripts/convert_svg_to_pdf.py \\
    --input_dir /path/to/svg \\
    --output_dir /path/to/pdf_out \\
    --prototype_ids 20 21 23 24 25 26 27 28 36 39
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert_one(svg_path: Path, pdf_path: Path) -> None:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    drawing = svg2rlg(str(svg_path))
    if drawing is None:
        raise RuntimeError(f"svglib could not parse: {svg_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(pdf_path))


def discover_svgs(
    input_dir: Path,
    glob_pattern: str,
    prototype_ids: list[int] | None,
) -> list[Path]:
    if prototype_ids:
        paths: list[Path] = []
        for pid in prototype_ids:
            matches = sorted(input_dir.glob(f"prototype_{pid}*.svg"))
            if not matches:
                raise FileNotFoundError(
                    f"No SVG matching prototype_{pid}*.svg in {input_dir}"
                )
            if len(matches) > 1:
                # Prefer exact prototype_{id}_class_3.svg if present
                preferred = input_dir / f"prototype_{pid}_class_3.svg"
                paths.append(preferred if preferred.is_file() else matches[0])
            else:
                paths.append(matches[0])
        return paths

    svgs = sorted(input_dir.glob(glob_pattern))
    if not svgs:
        raise FileNotFoundError(
            f"No files matching {glob_pattern!r} in {input_dir}"
        )
    return svgs


def output_name(svg_path: Path, strip_class_suffix: bool) -> str:
    stem = svg_path.stem
    if strip_class_suffix and stem.endswith("_class_3"):
        stem = stem[: -len("_class_3")]
    return f"{stem}.pdf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SVG files in a folder to PDF (vector, for LaTeX).",
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .svg files",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory where .pdf files are written",
    )
    p.add_argument(
        "--glob",
        default="*.svg",
        dest="glob_pattern",
        help="Glob pattern within input_dir (default: *.svg)",
    )
    p.add_argument(
        "--prototype_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional: convert only prototype_{id}*.svg (e.g. 20 21 23)",
    )
    p.add_argument(
        "--strip_class_suffix",
        action="store_true",
        help="Write prototype_20.pdf instead of prototype_20_class_3.pdf",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: input_dir is not a directory: {input_dir}", file=sys.stderr)
        return 1

    try:
        from svglib.svglib import svg2rlg  # noqa: F401
        from reportlab.graphics import renderPDF  # noqa: F401
    except ImportError:
        print(
            "ERROR: svglib and reportlab are required.\n"
            "  conda activate new_insight_env\n"
            "  pip install reportlab lxml tinycss2 cssselect2\n"
            "  pip install svglib --no-deps",
            file=sys.stderr,
        )
        return 1

    try:
        svgs = discover_svgs(input_dir, args.glob_pattern, args.prototype_ids)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ok = 0
    failed: list[str] = []
    for svg in svgs:
        pdf = output_dir / output_name(svg, args.strip_class_suffix)
        try:
            convert_one(svg, pdf)
            print(f"OK  {svg.name} -> {pdf}")
            ok += 1
        except Exception as exc:
            failed.append(f"{svg.name}: {exc}")
            print(f"FAIL {svg.name}: {exc}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(svgs)} converted -> {output_dir}")
    if failed:
        print(f"Failed ({len(failed)}):", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
