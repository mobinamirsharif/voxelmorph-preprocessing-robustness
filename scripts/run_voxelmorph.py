#!/usr/bin/env python3
"""Run pretrained VoxelMorph inference and save the moved image and warp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from voxelmorph_pipeline.io_utils import (
    assert_same_geometry,
    load_nifti_float,
    save_float_nifti,
)
from voxelmorph_pipeline.inference import predict_moved_and_warp
from voxelmorph_pipeline.metrics import displacement_summary
from voxelmorph_pipeline.quality import require_float_storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-warp", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else args.gpu

    import tensorflow as tf
    import voxelmorph as vxm

    if not args.cpu:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("No TensorFlow GPU detected; pass --cpu for CPU inference")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    fixed_image, fixed = load_nifti_float(args.fixed)
    moving_image, moving = load_nifti_float(args.moving)
    require_float_storage(args.fixed, purpose="VoxelMorph fixed image")
    require_float_storage(args.moving, purpose="VoxelMorph moving image")
    assert_same_geometry(fixed_image, moving_image)

    if fixed.shape != (160, 192, 224):
        raise ValueError(f"Expected model input shape (160, 192, 224), got {fixed.shape}")

    moving_batch = moving[None, ..., None]
    fixed_batch = fixed[None, ..., None]
    model = vxm.networks.VxmDense.load(str(args.model))

    start = time.perf_counter()
    moved, warp = predict_moved_and_warp(
        model,
        moving_batch,
        fixed_batch,
        model_factory=tf.keras.Model,
    )
    elapsed = time.perf_counter() - start

    if warp.shape != fixed.shape + (3,):
        raise ValueError(f"Unexpected warp shape: {warp.shape}")
    if not np.isfinite(moved).all() or not np.isfinite(warp).all():
        raise ValueError("Inference produced NaN or Inf values")

    save_float_nifti(moved, fixed_image, args.output_image)
    args.output_warp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_warp, warp=warp)

    summary = {
        "moving": str(args.moving),
        "fixed": str(args.fixed),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "output_image": str(args.output_image),
        "output_warp": str(args.output_warp),
        "shape": list(moved.shape),
        "warp_shape": list(warp.shape),
        "moved_range": [float(moved.min()), float(moved.max())],
        "warp_finite": bool(np.isfinite(warp).all()),
        "displacement": displacement_summary(warp),
        "inference_seconds": float(elapsed),
        "timing_scope": "single neural-network forward pass only",
        "tensorflow_version": tf.__version__,
        "voxelmorph_version": getattr(vxm, "__version__", "unknown"),
        "device": "CPU" if args.cpu else f"GPU:{args.gpu}",
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
