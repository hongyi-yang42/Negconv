"""Synthetic test image generation + validation.

Creates a 64x64 image with three horizontal bands matching the verification trace:
- Top band:    pixel = 1.0 * Dmin  (film base / darkest)
- Middle band: pixel = 0.1 * Dmin  (dense negative / midtone)
- Bottom band: pixel = 0.01 * Dmin (very dense / highlight)

Runs the full pipeline and checks that the center pixel of each band
lands in the expected range after inversion.
"""

import numpy as np
import pytest

from negconv.params import NegconvParams
from negconv.pipeline import invert


@pytest.fixture
def synthetic_image():
    params = NegconvParams.color_film()
    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.float32)

    band_h = h // 3

    # Top band: film base (1.0 * Dmin)
    img[:band_h, :, :] = params.dmin * 1.0

    # Middle band: dense negative (0.1 * Dmin)
    img[band_h : 2 * band_h, :, :] = params.dmin * 0.1

    # Bottom band: very dense (0.01 * Dmin)
    img[2 * band_h :, :, :] = params.dmin * 0.01

    return img, params, band_h


def test_synthetic_film_base_band(synthetic_image):
    """Top band (1.0 * Dmin) -> near black after inversion."""
    img, params, band_h = synthetic_image
    result = invert(img, params)
    center = result[band_h // 2, 32, :]
    # Should be near black (~0.001)
    assert np.all(center < 0.01), f"Film base band too bright: {center}"


def test_synthetic_dense_band(synthetic_image):
    """Middle band (0.1 * Dmin) -> midtone after inversion."""
    img, params, band_h = synthetic_image
    result = invert(img, params)
    center = result[band_h + band_h // 2, 32, :]
    # Should be midtone (~0.408)
    assert np.all(center > 0.2), f"Dense band too dark: {center}"
    assert np.all(center < 0.6), f"Dense band too bright: {center}"


def test_synthetic_highlight_band(synthetic_image):
    """Bottom band (0.01 * Dmin) -> highlight after inversion."""
    img, params, band_h = synthetic_image
    result = invert(img, params)
    center = result[2 * band_h + band_h // 2, 32, :]
    # Should be highlight (~0.808)
    assert np.all(center > 0.6), f"Highlight band too dark: {center}"


def test_synthetic_shape_preserved(synthetic_image):
    """Output shape matches input shape."""
    img, params, _ = synthetic_image
    result = invert(img, params)
    assert result.shape == img.shape


def test_synthetic_no_infs_or_nans(synthetic_image):
    """No inf or nan values in output."""
    img, params, _ = synthetic_image
    result = invert(img, params)
    assert np.all(np.isfinite(result))
