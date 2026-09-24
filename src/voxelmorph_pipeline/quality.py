"""Intensity-encoding audits and fail-fast quality gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class IntensityAudit:
    """Serializable summary of a NIfTI intensity encoding."""

    path: str
    stored_dtype: str
    shape: tuple[int, ...]
    finite: bool
    minimum: float
    maximum: float
    nonzero_fraction: float
    sampled_unique_values: int
    sampled_fractional_percentage: float
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["shape"] = list(self.shape)
        result["reasons"] = list(self.reasons)
        return result


def _deterministic_sample(array: np.ndarray, maximum_values: int) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if flat.size <= maximum_values:
        return flat
    step = max(1, flat.size // maximum_values)
    return flat[::step][:maximum_values]


def audit_intensity_array(
    data: np.ndarray,
    *,
    stored_dtype: np.dtype | str,
    path: Path | str = "<array>",
    maximum_sample_values: int = 250_000,
) -> IntensityAudit:
    """Detect storage and value patterns associated with silent quantization."""
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {array.shape}")

    sample = _deterministic_sample(array, maximum_sample_values)
    finite = bool(np.isfinite(array).all())
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    nonzero_fraction = float(np.mean(array != 0))
    unique_count = int(np.unique(sample).size)
    fractional_percentage = float(
        100 * np.mean(np.abs(sample - np.rint(sample)) > 1e-6)
    )

    dtype = np.dtype(stored_dtype)
    reasons: list[str] = []
    failures: list[str] = []

    if not finite:
        failures.append("non_finite_values")
    if maximum <= minimum:
        failures.append("constant_image")
    if np.issubdtype(dtype, np.integer):
        reasons.append("integer_storage")
    if unique_count <= 16:
        failures.append("low_unique_value_count")
    if nonzero_fraction < 0.01:
        failures.append("extremely_sparse_foreground")

    reasons.extend(failures)
    status = "fail" if failures else ("warn" if reasons else "pass")
    return IntensityAudit(
        path=str(path),
        stored_dtype=str(dtype),
        shape=tuple(int(value) for value in array.shape),
        finite=finite,
        minimum=minimum,
        maximum=maximum,
        nonzero_fraction=nonzero_fraction,
        sampled_unique_values=unique_count,
        sampled_fractional_percentage=fractional_percentage,
        status=status,
        reasons=tuple(reasons),
    )


def audit_nifti(path: Path | str) -> IntensityAudit:
    """Load and audit a NIfTI-compatible image."""
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return audit_intensity_array(
        data,
        stored_dtype=image.header.get_data_dtype(),
        path=path,
    )


def require_float_storage(path: Path | str, *, purpose: str) -> IntensityAudit:
    """Reject integer-stored images before intensity-scaled registration."""
    audit = audit_nifti(path)
    dtype = np.dtype(audit.stored_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(
            f"{purpose} requires floating-point NIfTI storage; got "
            f"{audit.stored_dtype}: {path}"
        )
    if audit.status == "fail":
        raise ValueError(
            f"{purpose} input failed intensity QC: {', '.join(audit.reasons)}"
        )
    return audit
