import pytest

from training.validation.openpi_smoke import _expected_image_shape


def test_openpi_smoke_uses_framework_specific_image_layout():
    assert _expected_image_shape(2, "pytorch") == [2, 3, 224, 224]
    assert _expected_image_shape(2, "jax") == [2, 224, 224, 3]


def test_openpi_smoke_rejects_unknown_framework():
    with pytest.raises(ValueError, match="Unsupported framework"):
        _expected_image_shape(2, "unknown")
