"""Registration and deformation-field quality-control metrics."""

from __future__ import annotations

import numpy as np


def robust_mask(image: np.ndarray, fraction: float = 0.01) -> np.ndarray:
    """Create a foreground mask relative to the image maximum."""
    array = np.asarray(image)
    maximum = float(np.max(array))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Image maximum must be finite and positive")
    return array > fraction * maximum


def dice_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the Dice coefficient between two boolean masks."""
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    if first_mask.shape != second_mask.shape:
        raise ValueError("Dice masks must have identical shapes")
    denominator = np.count_nonzero(first_mask) + np.count_nonzero(second_mask)
    if denominator == 0:
        return 1.0
    intersection = np.count_nonzero(first_mask & second_mask)
    return float(2 * intersection / denominator)


def mean_label_dice(first: np.ndarray, second: np.ndarray) -> float:
    """Average Dice across all nonzero labels present in either array."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("Label maps must have identical shapes")
    labels = np.union1d(np.unique(first), np.unique(second))
    labels = labels[labels != 0]
    if labels.size == 0:
        raise ValueError("Label maps contain no nonzero labels")
    return float(
        np.mean([dice_coefficient(first == label, second == label) for label in labels])
    )


def image_metrics(reference: np.ndarray, moving: np.ndarray) -> dict[str, float]:
    """Compute foreground Dice, MSE, and correlation on the mask union."""
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    if reference.shape != moving.shape:
        raise ValueError("Images must have identical shapes")

    reference_mask = robust_mask(reference)
    moving_mask = robust_mask(moving)
    union = reference_mask | moving_mask
    reference_values = reference[union]
    moving_values = moving[union]

    mse = float(np.mean((reference_values - moving_values) ** 2))
    correlation = float(np.corrcoef(reference_values, moving_values)[0, 1])
    dice = dice_coefficient(reference_mask, moving_mask)
    return {"mse": mse, "correlation": correlation, "mask_dice": dice}


def jacobian_determinant(displacement: np.ndarray) -> np.ndarray:
    """Compute det(I + grad(u)) for an ij-indexed displacement field."""
    field = np.asarray(displacement, dtype=np.float32)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("Expected displacement shape (X, Y, Z, 3)")

    du_dx, du_dy, du_dz = np.gradient(field[..., 0])
    dv_dx, dv_dy, dv_dz = np.gradient(field[..., 1])
    dw_dx, dw_dy, dw_dz = np.gradient(field[..., 2])

    j11, j12, j13 = 1 + du_dx, du_dy, du_dz
    j21, j22, j23 = dv_dx, 1 + dv_dy, dv_dz
    j31, j32, j33 = dw_dx, dw_dy, 1 + dw_dz

    determinant = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return determinant.astype(np.float32, copy=False)


def jacobian_summary(
    displacement: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize finite values, folding percentage, and percentiles."""
    determinant = jacobian_determinant(displacement)
    values = determinant if mask is None else determinant[np.asarray(mask, dtype=bool)]
    return {
        "finite": bool(np.isfinite(values).all()),
        "non_positive_percentage": float(100 * np.mean(values <= 0)),
        "percentiles_0_1_50_99_100": [
            float(value) for value in np.percentile(values, [0, 1, 50, 99, 100])
        ],
    }


def displacement_summary(displacement: np.ndarray) -> dict[str, float]:
    """Summarize displacement-vector magnitudes in voxel units."""
    magnitude = np.linalg.norm(np.asarray(displacement, dtype=np.float32), axis=-1)
    return {
        "mean_voxels": float(np.mean(magnitude)),
        "maximum_voxels": float(np.max(magnitude)),
    }
