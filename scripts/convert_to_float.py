#!/usr/bin/env python3
"""Convert a 3D NIfTI image to float32 without changing data or geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voxelmorph_pipeline.io_utils import load_nifti_float, save_float_nifti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, data = load_nifti_float(args.input)
    save_float_nifti(data, image, args.output)

    reloaded, reloaded_data = load_nifti_float(args.output)
    if not np.allclose(reloaded.affine, image.affine):
        raise RuntimeError("Output affine differs from the input")
    if not np.array_equal(reloaded_data, data):
        raise RuntimeError("Output data differs from the input")

    print("Output:", args.output)
    print("Datatype:", reloaded.header.get_data_dtype())
    print("Shape:", reloaded.shape)
    print("Finite:", bool(np.isfinite(reloaded_data).all()))


if __name__ == "__main__":
    main()

