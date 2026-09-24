#!/usr/bin/env python3
"""Run SynthStrip followed by rigid and affine FreeSurfer registration."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from voxelmorph_pipeline.io_utils import load_nifti_float, save_float_nifti
from voxelmorph_pipeline.naming import image_stem
from voxelmorph_pipeline.quality import require_float_storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--preproc-dir", type=Path, required=True)
    parser.add_argument("--registration-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    for executable in ("mri_synthstrip", "mri_robust_register"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not available on PATH")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.atlas.is_file():
        raise FileNotFoundError(args.atlas)

    stem = image_stem(args.subject, args.session)
    args.preproc_dir.mkdir(parents=True, exist_ok=True)
    args.registration_dir.mkdir(parents=True, exist_ok=True)

    brain = args.preproc_dir / f"{stem}_desc-brain_T1w.nii.gz"
    mask = args.preproc_dir / f"{stem}_desc-brain_mask.nii.gz"
    brain_float = args.preproc_dir / f"{stem}_desc-brainFloat_T1w.nii.gz"
    rigid = args.registration_dir / f"{stem}_space-vxm-atlas_desc-rigid_T1w.nii.gz"
    rigid_weights = args.registration_dir / (
        f"{stem}_space-vxm-atlas_desc-rigid_weights.nii.gz"
    )
    rigid_scale = args.registration_dir / f"{stem}_mode-rigid_iscale.txt"
    rigid_lta = args.registration_dir / f"{stem}_from-native_to-vxm-atlas_mode-rigid.lta"
    affine = args.registration_dir / f"{stem}_space-vxm-atlas_desc-affine_T1w.nii.gz"
    affine_weights = args.registration_dir / (
        f"{stem}_space-vxm-atlas_desc-affine_weights.nii.gz"
    )
    affine_scale = args.registration_dir / f"{stem}_mode-affine_iscale.txt"
    affine_lta = args.registration_dir / f"{stem}_from-native_to-vxm-atlas_mode-affine.lta"

    run(
        [
            "mri_synthstrip",
            "-t",
            str(args.threads),
            "-i",
            str(args.input),
            "-o",
            str(brain),
            "-m",
            str(mask),
        ]
    )

    brain_image, brain_data = load_nifti_float(brain)
    save_float_nifti(brain_data, brain_image, brain_float)
    audit = require_float_storage(
        brain_float,
        purpose="Intensity-scaled FreeSurfer registration",
    )
    print("Pre-registration intensity audit:", audit.to_dict())

    common = [
        "--mov",
        str(brain_float),
        "--dst",
        str(args.atlas),
        "--iscale",
        "--satit",
        "--maxit",
        "20",
        "--highit",
        "20",
    ]
    run(
        [
            "mri_robust_register",
            *common,
            "--initorient",
            "--lta",
            str(rigid_lta),
            "--mapmov",
            str(rigid),
            "--weights",
            str(rigid_weights),
            "--iscaleout",
            str(rigid_scale),
        ]
    )
    run(
        [
            "mri_robust_register",
            *common,
            "--affine",
            "--ixform",
            str(rigid_lta),
            "--iscalein",
            str(rigid_scale),
            "--iscaleout",
            str(affine_scale),
            "--lta",
            str(affine_lta),
            "--mapmov",
            str(affine),
            "--weights",
            str(affine_weights),
        ]
    )

    print("Brain mask:", mask)
    print("Float brain:", brain_float)
    print("Rigid output:", rigid)
    print("Affine output:", affine)


if __name__ == "__main__":
    main()
