#!/usr/bin/env python3
"""Compare affine and VoxelMorph outputs against the fixed atlas."""

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
from voxelmorph_pipeline.metrics import (
    image_metrics,
    jacobian_summary,
    robust_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--moved", type=Path, required=True)
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atlas_image, atlas = load_nifti_float(args.atlas)
    affine_image, affine = load_nifti_float(args.affine)
    moved_image, moved = load_nifti_float(args.moved)
    assert_same_geometry(atlas_image, affine_image)
    assert_same_geometry(atlas_image, moved_image)

    warp = load_warp_npz(args.warp)
    if warp.shape[:3] != atlas.shape:
        raise ValueError("Warp and atlas shapes do not match")

    before = image_metrics(atlas, affine)
    after = image_metrics(atlas, moved)
    brain_mask = robust_mask(atlas)
    jacobian = jacobian_summary(warp, brain_mask)

    summary = {
        "shape": list(atlas.shape),
        "geometry_matches_atlas": True,
        "finite": bool(
            np.isfinite(atlas).all()
            and np.isfinite(affine).all()
            and np.isfinite(moved).all()
            and np.isfinite(warp).all()
        ),
        "before_voxelmorph": before,
        "after_voxelmorph": after,
        "relative_mse_reduction_percentage": float(
            100 * (before["mse"] - after["mse"]) / before["mse"]
        ),
        "jacobian": jacobian,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

