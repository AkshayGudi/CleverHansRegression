#!/usr/bin/env python3
"""
Overlay artifact patches onto fundus images according to per-experiment CSV configs.

Reads ``data_info.json``, ``*_labeled_data.csv``, and ``artifact_pos_*.csv`` from
a config directory (e.g. ``data_preparation/DR-100-fixed``). Images with
``artifact_label == 1`` receive the patch at the center and size recorded in the
position CSV; others are copied through unchanged.

Config files (per experiment folder)
------------------------------------
data_info.json
    artifact_name        Patch filename, e.g. art16.png

train_labeled_data.csv / test_labeled_data.csv
    image_name, artifact_label   0 = no patch, 1 = overlay patch

artifact_pos_train.csv / artifact_pos_test.csv
    image_name, center_x, center_y, target_size, image_w, image_h
    One row per image that receives a patch (placement source of truth).

Other CSVs in the config folder (DR_*_patch.csv, *_yellow_patch.csv) are
selection manifests from data preparation; this script does not read them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


class ArtifactPosition(TypedDict):
    center_x: int
    center_y: int
    target_size: int


def create_feathered_mask(size: tuple[int, int], feather_amount: int = 15) -> np.ndarray:
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - feather_amount, 255, -1)
    mask = cv2.GaussianBlur(mask, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
    return cv2.merge([mask, mask, mask])


def load_position_csv(position_csv: Path) -> dict[str, ArtifactPosition]:
    """Load per-image artifact placement keyed by image stem (no extension)."""
    positions: dict[str, ArtifactPosition] = {}
    with position_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "center_x", "center_y", "target_size"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{position_csv} must have columns: {', '.join(sorted(required))}"
            )
        for row in reader:
            stem = row["image_name"].strip()
            if not stem:
                continue
            positions[stem] = {
                "center_x": int(row["center_x"]),
                "center_y": int(row["center_y"]),
                "target_size": int(row["target_size"]),
            }
    return positions


def add_artifact(
    input_dir: str | Path,
    output_dir: str | Path,
    artifact_selection_csv: str | Path,
    artifact_path: str | Path,
    position_csv: str | Path,
) -> tuple[int, int]:
    """
    Overlay artifact patches onto retina images listed in ``artifact_selection_csv``.

    Placement for ``artifact_label == 1`` images is taken from ``position_csv``.

    Returns (images_written, artifacts_placed).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    artifact_selection_csv = Path(artifact_selection_csv)
    artifact_path = Path(artifact_path)
    position_csv = Path(position_csv)

    if not position_csv.is_file():
        raise FileNotFoundError(f"Position CSV not found: {position_csv}")

    positions = load_position_csv(position_csv)
    retina_images = sorted(glob(str(input_dir / "*.jpeg")))
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_selection_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        image_flags = {
            row["image_name"] + ".jpeg": int(row["artifact_label"])
            for row in reader
        }

    artifact = cv2.imread(str(artifact_path))
    if artifact is None:
        raise FileNotFoundError(f"Could not read artifact patch: {artifact_path}")

    written = 0
    placed = 0
    missing_positions: list[str] = []

    for retina_path in retina_images:
        retina = cv2.imread(retina_path)
        if retina is None:
            print(f"Warning: could not read {retina_path}", file=sys.stderr)
            continue

        retina_name = os.path.basename(retina_path)
        if retina_name not in image_flags:
            continue

        has_artifact = image_flags[retina_name]
        if has_artifact == 1:
            stem = os.path.splitext(retina_name)[0]
            pos = positions.get(stem)
            if pos is None:
                missing_positions.append(stem)
                continue

            target_size = pos["target_size"]
            center = (pos["center_x"], pos["center_y"])
            artifact_resized = cv2.resize(artifact, (target_size, target_size))
            feathered_mask = create_feathered_mask(artifact_resized.shape[:2])
            output_image = cv2.seamlessClone(
                artifact_resized, retina, feathered_mask, center, cv2.NORMAL_CLONE
            )
            placed += 1
        else:
            output_image = retina

        cv2.imwrite(str(output_dir / retina_name), output_image)
        written += 1

    if missing_positions:
        print(
            f"\nMissing positions for {len(missing_positions)} artifact images "
            f"in {position_csv.name}, first 10:",
            file=sys.stderr,
        )
        for stem in missing_positions[:10]:
            print(f"  {stem}", file=sys.stderr)
        raise SystemExit(
            "Some images with artifact_label=1 have no row in the position CSV."
        )

    return written, placed


def load_overlay_settings(config_dir: Path, artifact_patch_dir: Path) -> dict:
    config_dir = config_dir.resolve()
    data_json_path = config_dir / "data_info.json"
    with data_json_path.open(encoding="utf-8") as f:
        data_info = json.load(f)

    artifact_path = artifact_patch_dir / str(data_info["artifact_name"]).strip()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact patch not found: {artifact_path}")

    return {
        "artifact_path": artifact_path,
        "data_info": data_info,
    }


def run_overlay(
    *,
    config_dir: Path,
    train_input_dir: Path,
    test_input_dir: Path,
    output_dir: Path,
    artifact_patch_dir: Path | None = None,
) -> None:
    config_dir = config_dir.resolve()
    artifact_patch_dir = (artifact_patch_dir or _SCRIPT_DIR).resolve()
    output_dir = output_dir.resolve()

    settings = load_overlay_settings(config_dir, artifact_patch_dir)

    train_out = output_dir / "train"
    test_out = output_dir / "test"
    train_pos_csv = config_dir / "artifact_pos_train.csv"
    test_pos_csv = config_dir / "artifact_pos_test.csv"

    print(f"Config:         {config_dir}")
    print(f"Artifact patch: {settings['artifact_path']}")
    print(f"Train positions:{train_pos_csv}")
    print(f"Test positions: {test_pos_csv}")
    print(f"Train input:    {train_input_dir}")
    print(f"Test input:     {test_input_dir}")
    print(f"Output:         {output_dir}\n")

    train_written, train_placed = add_artifact(
        train_input_dir,
        train_out,
        config_dir / "train_labeled_data.csv",
        settings["artifact_path"],
        train_pos_csv,
    )
    print(f"Train: wrote {train_written} images ({train_placed} with artifact) -> {train_out}")

    test_written, test_placed = add_artifact(
        test_input_dir,
        test_out,
        config_dir / "test_labeled_data.csv",
        settings["artifact_path"],
        test_pos_csv,
    )
    print(f"Test:  wrote {test_written} images ({test_placed} with artifact) -> {test_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay artifact patches using per-experiment config CSVs and data_info.json.",
    )
    p.add_argument(
        "--config_dir",
        type=Path,
        required=True,
        help="Experiment config folder (e.g. data_preparation/DR-100-fixed).",
    )
    p.add_argument(
        "--train_input_dir",
        type=Path,
        required=True,
        help="Directory of clean training JPEGs (e.g. original_data/original_train).",
    )
    p.add_argument(
        "--test_input_dir",
        type=Path,
        required=True,
        help="Directory of clean test JPEGs (e.g. original_data/original_test).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root; writes train/ and test/ subdirectories.",
    )
    p.add_argument(
        "--artifact_patch_dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Directory containing the artifact PNG named in data_info.json (default: data_preparation/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.config_dir, "config_dir"),
        (args.train_input_dir, "train_input_dir"),
        (args.test_input_dir, "test_input_dir"),
    ):
        if not path.is_dir():
            raise SystemExit(f"{label} not found: {path}")

    for csv_name in (
        "train_labeled_data.csv",
        "test_labeled_data.csv",
        "artifact_pos_train.csv",
        "artifact_pos_test.csv",
        "data_info.json",
    ):
        if not (args.config_dir / csv_name).is_file():
            raise SystemExit(f"Missing {csv_name} in {args.config_dir}")

    run_overlay(
        config_dir=args.config_dir,
        train_input_dir=args.train_input_dir,
        test_input_dir=args.test_input_dir,
        output_dir=args.output_dir,
        artifact_patch_dir=args.artifact_patch_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
