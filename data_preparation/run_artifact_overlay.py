#!/usr/bin/env python3
"""
Run artifact overlay for a Clever-Hans DR experiment configuration.

Wraps ``overlay_patches_conditionally.run_overlay`` with sensible defaults for
the four thesis experiments and copies static config files into the output tree.

Example
-------
  python data_preparation/run_artifact_overlay.py \\
      --experiment DR-100-fixed \\
      --original_data_dir original_data \\
      --output_dir data/DR-100-fixed
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from overlay_patches_conditionally import run_overlay  # noqa: E402
_REPO_ROOT = _SCRIPT_DIR.parent

EXPERIMENTS = (
    "DR-100-fixed",
    "DR-100-random",
    "DR-50-fixed",
    "DR-50-random",
)

# Config CSVs / JSON to copy into the release layout.
_CONFIG_FILES_TO_COPY = (
    "data_info.json",
    "train_labeled_data.csv",
    "test_labeled_data.csv",
    "artifact_pos_train.csv",
    "artifact_pos_test.csv",
    "DR_train_data_patch.csv",
    "DR_test_data_patch.csv",
    "train_yellow_patch.csv",
    "test_yellow_patch.csv",
    "info.txt",
)


def copy_static_configs(config_dir: Path, dest_details_dir: Path) -> None:
    dest_details_dir.mkdir(parents=True, exist_ok=True)
    for name in _CONFIG_FILES_TO_COPY:
        src = config_dir / name
        if src.is_file():
            shutil.copy2(src, dest_details_dir / name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay artifacts for a DR Clever-Hans experiment.",
    )
    p.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        required=True,
        help="Experiment name (maps to data_preparation/<name>/ config folder).",
    )
    p.add_argument(
        "--original_data_dir",
        type=Path,
        default=_REPO_ROOT / "original_data",
        help="Root with original_train/ and original_test/ (default: repo/original_data).",
    )
    p.add_argument(
        "--train_input_dir",
        type=Path,
        default=None,
        help="Override training images dir (default: <original_data_dir>/original_train).",
    )
    p.add_argument(
        "--test_input_dir",
        type=Path,
        default=None,
        help="Override test images dir (default: <original_data_dir>/original_test).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root (default: repo/data/<experiment>).",
    )
    p.add_argument(
        "--config_dir",
        type=Path,
        default=None,
        help="Config folder override (default: data_preparation/<experiment>).",
    )
    p.add_argument(
        "--artifact_patch_dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Directory containing art16.png (default: data_preparation/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_dir = (args.config_dir or (_SCRIPT_DIR / args.experiment)).resolve()
    output_dir = (args.output_dir or (_REPO_ROOT / "data" / args.experiment)).resolve()
    original_data_dir = args.original_data_dir.expanduser().resolve()

    train_input_dir = (
        args.train_input_dir.expanduser().resolve()
        if args.train_input_dir
        else original_data_dir / "original_train"
    )
    test_input_dir = (
        args.test_input_dir.expanduser().resolve()
        if args.test_input_dir
        else original_data_dir / "original_test"
    )

    details_out = output_dir / "artifact" / "data_details_class3"
    copy_static_configs(config_dir, details_out)

    run_overlay(
        config_dir=config_dir,
        train_input_dir=train_input_dir,
        test_input_dir=test_input_dir,
        output_dir=output_dir,
        artifact_patch_dir=args.artifact_patch_dir,
    )

    print(f"\nExperiment layout ready under: {output_dir}")
    print(f"  train/  test/  artifact/data_details_class3/")


if __name__ == "__main__":
    main()
