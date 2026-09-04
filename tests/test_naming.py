import pytest

from voxelmorph_pipeline.naming import image_stem, validate_label


def test_image_stem():
    assert image_stem("SAMPLE_001", "m06") == "sub-SAMPLE_001_ses-m06"


@pytest.mark.parametrize("value", ["../secret", "m06/test", "", "space label"])
def test_unsafe_labels_are_rejected(value):
    with pytest.raises(ValueError):
        validate_label(value, "test")
