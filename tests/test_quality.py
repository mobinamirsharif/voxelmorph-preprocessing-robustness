from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from voxelmorph_pipeline.quality import (
    audit_intensity_array,
    audit_nifti,
    require_float_storage,
)


def test_integer_quantization_is_detected():
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[2:18, 2:18, 2:18] = 1
    data[8:12, 8:12, 8:12] = 2

    audit = audit_intensity_array(data, stored_dtype=np.int16)

    assert audit.status == "fail"
    assert "integer_storage" in audit.reasons
    assert "low_unique_value_count" in audit.reasons


def test_healthy_float_image_passes():
    data = np.linspace(0, 1, 30 * 31 * 32, dtype=np.float32).reshape(30, 31, 32)

    audit = audit_intensity_array(data, stored_dtype=np.float32)

    assert audit.status == "pass"
    assert audit.sampled_unique_values > 16
    assert audit.sampled_fractional_percentage > 99


def test_integer_image_with_dynamic_range_warns():
    data = np.arange(20 * 20 * 20, dtype=np.int16).reshape(20, 20, 20)

    audit = audit_intensity_array(data, stored_dtype=np.int16)

    assert audit.status == "warn"
    assert audit.reasons == ("integer_storage",)


def test_sparse_float_output_fails():
    data = np.zeros((30, 30, 30), dtype=np.float32)
    data[0, 0, :20] = np.linspace(0.1, 1.0, 20)

    audit = audit_intensity_array(data, stored_dtype=np.float32)

    assert audit.status == "fail"
    assert "extremely_sparse_foreground" in audit.reasons


def test_preflight_rejects_integer_nifti(tmp_path: Path):
    path = tmp_path / "integer.nii.gz"
    data = np.arange(24 * 24 * 24, dtype=np.int16).reshape(24, 24, 24)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    with pytest.raises(ValueError, match="floating-point NIfTI storage"):
        require_float_storage(path, purpose="registration")


def test_preflight_accepts_float_nifti(tmp_path: Path):
    path = tmp_path / "float.nii.gz"
    data = np.linspace(0, 1, 24 * 24 * 24, dtype=np.float32).reshape(24, 24, 24)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    audit = require_float_storage(path, purpose="registration")

    assert audit_nifti(path).stored_dtype == "float32"
    assert audit.status == "pass"
