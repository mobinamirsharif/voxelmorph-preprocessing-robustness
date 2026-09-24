#!/usr/bin/env python3
"""Reproduce the int16 failure and compare corrected registration branches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from voxelmorph_pipeline.io_utils import (
    assert_same_geometry,
    load_nifti_float,
    save_float_nifti,
)
from voxelmorph_pipeline.metrics import image_metrics
from voxelmorph_pipeline.quality import audit_nifti, require_float_storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-int16", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--maxit", type=int, default=20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing branch outputs instead of stopping.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_new_or_resume(path: Path, resume: bool) -> bool:
    if path.exists():
        if resume:
            print("Reusing:", path)
            return False
        raise FileExistsError(
            f"Output already exists: {path}. Use --resume or a new output directory."
        )
    return True


def register_branch(
    *,
    moving: Path,
    atlas: Path,
    branch_dir: Path,
    maxit: int,
    resume: bool,
) -> tuple[Path, Path]:
    branch_dir.mkdir(parents=True, exist_ok=True)
    rigid = branch_dir / "rigid_T1w.nii.gz"
    rigid_lta = branch_dir / "rigid.lta"
    rigid_scale = branch_dir / "rigid_iscale.txt"
    rigid_weights = branch_dir / "rigid_weights.nii.gz"
    affine = branch_dir / "affine_T1w.nii.gz"
    affine_lta = branch_dir / "affine.lta"
    affine_scale = branch_dir / "affine_iscale.txt"
    affine_weights = branch_dir / "affine_weights.nii.gz"

    common = [
        "--mov",
        str(moving),
        "--dst",
        str(atlas),
        "--iscale",
        "--satit",
        "--maxit",
        str(maxit),
        "--highit",
        str(maxit),
    ]

    if require_new_or_resume(rigid, resume):
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

    if require_new_or_resume(affine, resume):
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
    return rigid, affine


def stage_summary(atlas_path: Path, image_path: Path) -> dict[str, object]:
    atlas_image, atlas = load_nifti_float(atlas_path)
    image, data = load_nifti_float(image_path)
    assert_same_geometry(atlas_image, image)
    return {
        "path": str(image_path),
        "intensity_audit": audit_nifti(image_path).to_dict(),
        **image_metrics(atlas, data),
    }


def main() -> None:
    args = parse_args()
    if shutil.which("mri_robust_register") is None:
        raise RuntimeError("mri_robust_register is not available on PATH")
    for path in (args.brain_int16, args.atlas):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_audit = audit_nifti(args.brain_int16)
    if "integer_storage" not in source_audit.reasons:
        raise ValueError(
            "--brain-int16 must use integer storage to reproduce the failure branch"
        )

    integer_rigid, integer_affine = register_branch(
        moving=args.brain_int16,
        atlas=args.atlas,
        branch_dir=args.output_dir / "branch-a-int16",
        maxit=args.maxit,
        resume=args.resume,
    )

    float_dir = args.output_dir / "branch-b-float32"
    float_dir.mkdir(parents=True, exist_ok=True)
    float_brain = float_dir / "brain_float32_T1w.nii.gz"
    if require_new_or_resume(float_brain, args.resume):
        source_image, source_data = load_nifti_float(args.brain_int16)
        save_float_nifti(source_data, source_image, float_brain)
    require_float_storage(float_brain, purpose="Corrected registration branch")

    float_rigid, float_affine = register_branch(
        moving=float_brain,
        atlas=args.atlas,
        branch_dir=float_dir,
        maxit=args.maxit,
        resume=args.resume,
    )

    stages = {
        "int16_rigid": stage_summary(args.atlas, integer_rigid),
        "int16_affine": stage_summary(args.atlas, integer_affine),
        "float32_rigid": stage_summary(args.atlas, float_rigid),
        "float32_affine": stage_summary(args.atlas, float_affine),
    }

    if args.model:
        if not args.model.is_file():
            raise FileNotFoundError(args.model)
        vxm_dir = args.output_dir / "branch-c-voxelmorph"
        vxm_dir.mkdir(parents=True, exist_ok=True)
        moved = vxm_dir / "vxm_T1w.nii.gz"
        warp = vxm_dir / "vxm_warp.npz"
        runtime = vxm_dir / "vxm_runtime.json"
        if require_new_or_resume(moved, args.resume):
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_voxelmorph.py")),
                "--moving",
                str(float_affine),
                "--fixed",
                str(args.atlas),
                "--model",
                str(args.model),
                "--output-image",
                str(moved),
                "--output-warp",
                str(warp),
                "--output-json",
                str(runtime),
                "--gpu",
                args.gpu,
            ]
            if args.cpu:
                command.append("--cpu")
            run(command)
        stages["float32_affine_voxelmorph"] = stage_summary(args.atlas, moved)

    summary = {
        "experiment": "silent_dtype_quantization_ablation",
        "source": source_audit.to_dict(),
        "atlas": str(args.atlas),
        "stages": stages,
        "interpretation": (
            "The int16 branch intentionally reproduces a failure. The float32 branch "
            "is the production-safe path. Similarity metrics are QC signals, not "
            "anatomical ground truth."
        ),
    }
    output_json = args.output_dir / "dtype_ablation.json"
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Summary:", output_json)


if __name__ == "__main__":
    main()
