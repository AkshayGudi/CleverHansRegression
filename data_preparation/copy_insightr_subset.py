#!/usr/bin/env python3
"""
Copy the INSightR-Net train/test image subset from a local Kaggle EyePACS download.

The full Kaggle Diabetic Retinopathy Detection release is much larger than the
balanced subset used in the INSightR-Net paper and in this repository. Image
lists are defined by:

  - data_preparation/DR_train_data.csv  (8,908 training images)
  - data_preparation/DR_test_data.csv   (6,030 test images)

Each CSV has column ``image_name`` (stem without extension, e.g. ``792_right``).
Files are expected as ``<stem>.jpeg`` under the Kaggle ``train/`` and ``test/``
folders respectively.

Outputs (created under ``--output_dir``):

  original_train/   copies of training images
  original_test/    copies of test images

Example
-------
  # After downloading and extracting the Kaggle competition data:
  #
  #   kaggle competitions download -c diabetic-retinopathy-detection
  #   unzip train.zip && unzip test.zip   # yields train/ and test/

  cd /path/to/CleverHansRegression
  python data_preparation/copy_insightr_subset.py \
      --kaggle_dir /path/to/diabetic-retinopathy-detection \
      --output_dir original_data
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def _read_image_names(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "image_name" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must have an 'image_name' column")
        return [row["image_name"].strip() for row in reader if row["image_name"].strip()]


def _resolve_source(kaggle_dir: Path, split: str, stem: str) -> Path | None:
    """Return path to source JPEG if it exists under kaggle_dir."""
    candidates = [
        kaggle_dir / split / f"{stem}.jpeg",
        kaggle_dir / split / f"{stem}.jpg",
        kaggle_dir / f"{split}_images" / f"{stem}.jpeg",
        kaggle_dir / f"{split}_images" / f"{stem}.jpg",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _copy_split(
    *,
    names: list[str],
    kaggle_dir: Path,
    kaggle_split: str,
    dest_dir: Path,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: list[str] = []
    for stem in names:
        src = _resolve_source(kaggle_dir, kaggle_split, stem)
        dst = dest_dir / f"{stem}.jpeg"
        if src is None:
            missing.append(stem)
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        copied += 1
    return copied, len(names), missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy INSightR-Net train/test subset from a Kaggle DR download.",
    )
    p.add_argument(
        "--kaggle_dir",
        type=Path,
        required=True,
        help="Root folder of the extracted Kaggle download (contains train/ and test/).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=_REPO_ROOT / "original_data",
        help="Output root; writes original_train/ and original_test/ (default: repo/original_data).",
    )
    p.add_argument(
        "--train_csv",
        type=Path,
        default=_SCRIPT_DIR / "DR_train_data.csv",
        help="CSV listing training image stems.",
    )
    p.add_argument(
        "--test_csv",
        type=Path,
        default=_SCRIPT_DIR / "DR_test_data.csv",
        help="CSV listing test image stems.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print counts only; do not copy files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kaggle_dir = args.kaggle_dir.expanduser().resolve()
    if not kaggle_dir.is_dir():
        raise SystemExit(f"Kaggle directory not found: {kaggle_dir}")

    for csv_path in (args.train_csv, args.test_csv):
        if not csv_path.is_file():
            raise SystemExit(f"Missing CSV: {csv_path}")

    train_names = _read_image_names(args.train_csv)
    test_names = _read_image_names(args.test_csv)

    out = args.output_dir.expanduser().resolve()
    train_dest = out / "original_train"
    test_dest = out / "original_test"

    print(f"Kaggle source:  {kaggle_dir}")
    print(f"Output:         {out}")
    print(f"Train list:     {args.train_csv} ({len(train_names)} images)")
    print(f"Test list:      {args.test_csv} ({len(test_names)} images)")
    if args.dry_run:
        print("DRY RUN — no files will be copied.\n")

    train_copied, train_total, train_missing = _copy_split(
        names=train_names,
        kaggle_dir=kaggle_dir,
        kaggle_split="train",
        dest_dir=train_dest,
        dry_run=args.dry_run,
    )
    test_copied, test_total, test_missing = _copy_split(
        names=test_names,
        kaggle_dir=kaggle_dir,
        kaggle_split="test",
        dest_dir=test_dest,
        dry_run=args.dry_run,
    )

    print(f"Train: copied {train_copied}/{train_total} -> {train_dest}")
    print(f"Test:  copied {test_copied}/{test_total} -> {test_dest}")

    if train_missing:
        print(f"\nMissing train images ({len(train_missing)}), first 10:", file=sys.stderr)
        for stem in train_missing[:10]:
            print(f"  {stem}", file=sys.stderr)
    if test_missing:
        print(f"\nMissing test images ({len(test_missing)}), first 10:", file=sys.stderr)
        for stem in test_missing[:10]:
            print(f"  {stem}", file=sys.stderr)

    if train_missing or test_missing:
        raise SystemExit(
            "Some images were not found under --kaggle_dir. "
            "Check that train/ and test/ are extracted and paths are correct."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
