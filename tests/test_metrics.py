import numpy as np
import pytest

from voxelmorph_pipeline.metrics import (
    dice_coefficient,
    displacement_summary,
    image_metrics,
    jacobian_determinant,
    jacobian_summary,
    mean_label_dice,
    robust_mask,
)


def test_identical_masks_have_unit_dice():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    assert dice_coefficient(mask, mask) == 1.0


def test_empty_masks_have_unit_dice():
    mask = np.zeros((3, 3, 3), dtype=bool)
    assert dice_coefficient(mask, mask) == 1.0


def test_mean_label_dice_uses_nonzero_union():
    first = np.array([0, 1, 1, 2])
    second = np.array([0, 1, 2, 3])
    expected = np.mean([2 / 3, 0, 0])
    assert mean_label_dice(first, second) == pytest.approx(expected)


def test_robust_mask_rejects_empty_image():
    with pytest.raises(ValueError):
        robust_mask(np.zeros((3, 3, 3), dtype=np.float32))


def test_identical_images_have_expected_metrics():
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image[1:3, 1:3, 1:3] = np.arange(1, 9).reshape(2, 2, 2)
    metrics = image_metrics(image, image)
    assert metrics["mse"] == 0.0
    assert metrics["correlation"] == pytest.approx(1.0)
    assert metrics["mask_dice"] == 1.0


def test_zero_displacement_has_unit_jacobian():
    displacement = np.zeros((5, 6, 7, 3), dtype=np.float32)
    determinant = jacobian_determinant(displacement)
    assert determinant.shape == (5, 6, 7)
    assert np.allclose(determinant, 1.0)


def test_translation_has_unit_jacobian():
    displacement = np.zeros((5, 6, 7, 3), dtype=np.float32)
    displacement[..., 0] = 2.0
    displacement[..., 1] = -1.0
    summary = jacobian_summary(displacement)
    assert summary["finite"] is True
    assert summary["non_positive_percentage"] == 0.0
    assert np.allclose(summary["percentiles_0_1_50_99_100"], 1.0)


def test_displacement_magnitude_summary():
    displacement = np.zeros((2, 2, 2, 3), dtype=np.float32)
    displacement[..., 0] = 3.0
    displacement[..., 1] = 4.0
    summary = displacement_summary(displacement)
    assert summary["mean_voxels"] == pytest.approx(5.0)
    assert summary["maximum_voxels"] == pytest.approx(5.0)
