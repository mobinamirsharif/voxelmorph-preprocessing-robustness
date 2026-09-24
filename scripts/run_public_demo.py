#!/usr/bin/env python3
"""Run a public VoxelMorph smoke demo with upstream atlas and test-scan NPZs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from voxelmorph_pipeline.inference import predict_moved_and_warp
from voxelmorph_pipeline.metrics import mean_label_dice


KNOWN_MODEL_SHA256 = "8e5fe6bcbca68b4fa867864460315fdaa7e00139cd522379cea79db5e63a9e3c"
EXPECTED_SHAPE = (160, 192, 224)
ATLAS_URL = "https://raw.githubusercontent.com/voxelmorph/voxelmorph/75ac2a2cd7298af3b3d563a3f0cfa000e410d099/data/atlas.npz"
TEST_SCAN_URL = "https://raw.githubusercontent.com/voxelmorph/voxelmorph/75ac2a2cd7298af3b3d563a3f0cfa000e410d099/data/test_scan.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--atlas-npz", type=Path, required=True)
    parser.add_argument("--test-scan-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_public_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if not {"vol", "seg"}.issubset(archive.files):
            raise KeyError(f"Public NPZ must contain 'vol' and 'seg': {path}")
        volume = np.asarray(archive["vol"], dtype=np.float32)
        segmentation = np.asarray(archive["seg"], dtype=np.int16)
    if volume.shape != EXPECTED_SHAPE or segmentation.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected public arrays with shape {EXPECTED_SHAPE}")
    if not np.isfinite(volume).all():
        raise ValueError(f"Volume contains NaN or Inf: {path}")
    return volume, segmentation


def main() -> None:
    args = parse_args()
    model_digest = sha256(args.model)
    if model_digest != KNOWN_MODEL_SHA256:
        raise ValueError(
            f"Unexpected model SHA-256: {model_digest}; expected {KNOWN_MODEL_SHA256}"
        )

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else args.gpu
    import tensorflow as tf
    import voxelmorph as vxm

    if not args.cpu:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("No TensorFlow GPU detected; pass --cpu for CPU inference")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    fixed, fixed_seg = load_public_npz(args.atlas_npz)
    moving, moving_seg = load_public_npz(args.test_scan_npz)
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
    inference_seconds = time.perf_counter() - start

    moved_seg_batch = vxm.layers.SpatialTransformer(interp_method="nearest")(
        [moving_seg[None, ..., None].astype(np.float32), warp[None, ...]]
    )
    moved_seg = np.asarray(moved_seg_batch)[0, ..., 0].astype(np.int16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "public_demo_outputs.npz"
    report_path = args.output_dir / "public_demo_metrics.json"
    np.savez_compressed(arrays_path, moved=moved, warp=warp, moved_seg=moved_seg)
    report = {
        "provenance": {
            "classification": "public",
            "atlas": ATLAS_URL,
            "test_scan": TEST_SCAN_URL,
            "model_filename": args.model.name,
            "model_sha256": model_digest,
        },
        "shape": list(EXPECTED_SHAPE),
        "mse_before": float(np.mean((fixed - moving) ** 2)),
        "mse_after": float(np.mean((fixed - moved) ** 2)),
        "mean_nonzero_label_dice_before": mean_label_dice(fixed_seg, moving_seg),
        "mean_nonzero_label_dice_after": mean_label_dice(fixed_seg, moved_seg),
        "inference_seconds": float(inference_seconds),
        "timing_scope": "single neural-network forward pass only",
        "limitations": (
            "This public smoke test checks software execution on upstream example "
            "assets; it is not clinical validation or a population benchmark."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Arrays:", arrays_path)
    print("Metrics:", report_path)


if __name__ == "__main__":
    main()
