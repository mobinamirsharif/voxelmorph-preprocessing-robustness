#!/usr/bin/env python3
"""Compare atlas and registered-image intensity distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voxelmorph_pipeline.io_utils import assert_same_geometry, load_nifti_float
from voxelmorph_pipeline.metrics import robust_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.images):
        raise ValueError("--labels must have the same length as --images")

    atlas_image, atlas = load_nifti_float(args.atlas)
    atlas_mask = robust_mask(atlas)
    labels = args.labels or [path.stem for path in args.images]
    summary: dict[str, object] = {}

    arrays = []
    for label, path in zip(labels, args.images):
        image, data = load_nifti_float(path)
        assert_same_geometry(atlas_image, image)
        values = data[atlas_mask]
        arrays.append(data)
        summary[label] = {
            "range": [float(data.min()), float(data.max())],
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "percentiles_1_50_99": [
                float(value) for value in np.percentile(values, [1, 50, 99])
            ],
            "finite": bool(np.isfinite(data).all()),
        }

    if len(arrays) == 2:
        summary["pairwise_correlation_in_atlas_mask"] = float(
            np.corrcoef(arrays[0][atlas_mask], arrays[1][atlas_mask])[0, 1]
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

