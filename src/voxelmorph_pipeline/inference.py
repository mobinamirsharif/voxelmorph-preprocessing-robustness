"""Single-pass inference helpers for pinned TensorFlow VoxelMorph models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def predict_moved_and_warp(
    model: object,
    moving_batch: np.ndarray,
    fixed_batch: np.ndarray,
    *,
    model_factory: Callable[..., object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unbatched moved image and final warp from one model prediction.

    The pinned VxmDense implementation caches the moved source as ``y_source``
    and the integrated, full-resolution deformation as ``pos_flow``. Building a
    two-output view of that graph keeps both arrays from the same forward pass.
    ``model_factory`` is injected so the behavior can be tested without loading
    TensorFlow or model weights in the lightweight test suite.
    """
    references = getattr(model, "references", None)
    if references is None or not hasattr(references, "y_source"):
        raise TypeError("Model does not expose VxmDense y_source reference")
    if not hasattr(references, "pos_flow"):
        raise TypeError("Model does not expose VxmDense pos_flow reference")

    inference_model = model_factory(
        inputs=model.inputs,
        outputs=[references.y_source, references.pos_flow],
    )
    prediction = inference_model.predict([moving_batch, fixed_batch], verbose=0)
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 2:
        raise ValueError("Expected moved image and final warp from inference model")

    moved_batch = np.asarray(prediction[0], dtype=np.float32)
    warp_batch = np.asarray(prediction[1], dtype=np.float32)
    expected_warp_shape = moving_batch.shape[:-1] + (3,)
    if moved_batch.shape != moving_batch.shape:
        raise ValueError(f"Unexpected moved-image batch shape: {moved_batch.shape}")
    if warp_batch.shape != expected_warp_shape:
        raise ValueError(f"Unexpected warp batch shape: {warp_batch.shape}")
    if not np.isfinite(moved_batch).all() or not np.isfinite(warp_batch).all():
        raise ValueError("Inference produced NaN or Inf values")

    return moved_batch[0, ..., 0], warp_batch[0]
