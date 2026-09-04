"""NIfTI and deformation-field input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_nifti_float(path: Path | str) -> Tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Load a NIfTI-compatible image as finite float32 data."""
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Image contains NaN or Inf values: {path}")
    return image, data


def save_float_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path | str,
) -> Path:
    """Save float32 data using the geometry of a reference image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(array, reference.affine, header), str(output_path))
    return output_path


def load_warp_npz(path: Path | str) -> np.ndarray:
    """Load a compressed VoxelMorph displacement field."""
    with np.load(path) as archive:
        if "warp" not in archive.files:
            raise KeyError(f"Missing 'warp' array in {path}")
        warp = archive["warp"].astype(np.float32)
    if warp.ndim != 4 or warp.shape[-1] != 3:
        raise ValueError(f"Expected warp shape (X, Y, Z, 3), got {warp.shape}")
    if not np.isfinite(warp).all():
        raise ValueError(f"Warp contains NaN or Inf values: {path}")
    return warp


def assert_same_geometry(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    *,
    atol: float = 1e-5,
) -> None:
    """Require matching shape and voxel-to-world affine matrices."""
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} versus {second.shape}")
    if not np.allclose(first.affine, second.affine, atol=atol):
        raise ValueError("Affine matrices do not match")

