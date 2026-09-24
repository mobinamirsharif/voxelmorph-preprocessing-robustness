#!/usr/bin/env python3
"""Measure cross-session consistency after subject-to-atlas registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voxelmorph_pipeline.io_utils import (
    assert_same_geometry,
    load_nifti_float,
    load_warp_npz,
)
from voxelmorph_pipeline.metrics import dice_coefficient, robust_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--first-image", type=Path, required=True)
    parser.add_argument("--second-image", type=Path, required=True)
    parser.add_argument("--first-warp", type=Path, required=True)
    parser.add_argument("--second-warp", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atlas_image, atlas = load_nifti_float(args.atlas)
    first_image, first = load_nifti_float(args.first_image)
    second_image, second = load_nifti_float(args.second_image)
    assert_same_geometry(atlas_image, first_image)
    assert_same_geometry(atlas_image, second_image)

    first_warp = load_warp_npz(args.first_warp)
    second_warp = load_warp_npz(args.second_warp)
    if first_warp.shape != second_warp.shape or first_warp.shape[:3] != atlas.shape:
        raise ValueError("Warp shapes do not match the atlas")

    atlas_mask = robust_mask(atlas)
    first_values = first[atlas_mask]
    second_values = second[atlas_mask]
    warp_difference = np.linalg.norm(second_warp - first_warp, axis=-1)[atlas_mask]

    summary = {
        "shape": list(atlas.shape),
        "geometry_matches_atlas": True,
        "finite": bool(
            np.isfinite(first).all()
            and np.isfinite(second).all()
            and np.isfinite(first_warp).all()
            and np.isfinite(second_warp).all()
        ),
        "image_correlation": float(np.corrcoef(first_values, second_values)[0, 1]),
        "image_mse": float(np.mean((first_values - second_values) ** 2)),
        "image_mae": float(np.mean(np.abs(first_values - second_values))),
        "foreground_mask_dice": dice_coefficient(
            robust_mask(first), robust_mask(second)
        ),
        "warp_difference_voxel_percentiles_0_50_95_99_100": [
            float(value)
            for value in np.percentile(warp_difference, [0, 50, 95, 99, 100])
        ],
        "interpretation_warning": (
            "Cross-session warp differences are consistency metrics and must not be "
            "interpreted as atrophy or biological change."
        ),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

