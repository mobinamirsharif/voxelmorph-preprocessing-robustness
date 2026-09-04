#!/usr/bin/env python3
"""Convert the upstream VoxelMorph atlas NPZ to geometry-aware NIfTI files."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-npz", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-t1", type=Path, required=True)
    parser.add_argument("--output-seg", type=Path, required=True)
    parser.add_argument(
        "--crop-start",
        type=int,
        nargs=3,
        default=(48, 31, 3),
        metavar=("I", "J", "K"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = nib.load(str(args.reference))
    crop_start = np.asarray(args.crop_start, dtype=float)

    with np.load(args.atlas_npz) as archive:
        if not {"vol", "seg"}.issubset(archive.files):
            raise KeyError("Atlas archive must contain 'vol' and 'seg'")
        volume = archive["vol"].astype(np.float32)
        segmentation = archive["seg"].astype(np.int16)

    if volume.shape != segmentation.shape:
        raise ValueError("Atlas intensity and segmentation shapes differ")
    if volume.shape != (160, 192, 224):
        raise ValueError(f"Unexpected pretrained atlas shape: {volume.shape}")

    affine = reference.affine.copy()
    affine[:3, 3] = reference.affine[:3, :3] @ crop_start + reference.affine[:3, 3]

    args.output_t1.parent.mkdir(parents=True, exist_ok=True)
    args.output_seg.parent.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(volume, affine), str(args.output_t1))
    nib.save(nib.Nifti1Image(segmentation, affine), str(args.output_seg))

    orientation = "".join(nib.aff2axcodes(affine))
    print("Shape:", volume.shape)
    print("Voxel sizes:", nib.affines.voxel_sizes(affine))
    print("Orientation:", orientation)
    print("T1 output:", args.output_t1)
    print("Segmentation output:", args.output_seg)


if __name__ == "__main__":
    main()

