"""Tests for auto white balance."""

import numpy as np
import pytest

from negconv.params import auto_wb, NegconvParams


class TestAutoWB:
    def test_neutral_scene(self):
        """Equal-density channels → wb_high ≈ [1, 1, 1]."""
        # Image with equal per-channel density relative to dmin
        dmin = np.array([0.5, 0.4, 0.3], dtype=np.float32)
        img = np.zeros((100, 100, 3), dtype=np.float32)
        # All channels at 50% of dmin → equal log density
        img[:, :, 0] = dmin[0] * 0.3
        img[:, :, 1] = dmin[1] * 0.3
        img[:, :, 2] = dmin[2] * 0.3

        wb = auto_wb(img, dmin, d_max=1.5)
        assert np.allclose(wb, [1.0, 1.0, 1.0], atol=0.05)

    def test_orange_cast(self):
        """R>G>B density (simulating orange mask) → wb_high corrects toward neutral."""
        dmin = np.array([0.5, 0.4, 0.3], dtype=np.float32)
        img = np.zeros((100, 100, 3), dtype=np.float32)
        # R channel darker (denser), B brighter (thinner)
        img[:, :, 0] = dmin[0] * 0.15  # very dense R
        img[:, :, 1] = dmin[1] * 0.30  # medium G
        img[:, :, 2] = dmin[2] * 0.50  # thin B

        wb = auto_wb(img, dmin, d_max=1.5)
        # R is over-represented → wb_high[R] should be < 1 to compensate
        assert wb[0] < 0.95
        # B is under-represented → wb_high[B] should be > 1 to compensate
        assert wb[2] > 1.05

    def test_single_color_dominant(self):
        """Scene dominated by one color → wb_high stays within [0.25, 4.0]."""
        dmin = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = np.full((100, 100, 3), 0.25, dtype=np.float32)
        # Red-dominant scene: R much denser
        img[:, :, 0] = dmin[0] * 0.02  # extreme R density

        wb = auto_wb(img, dmin, d_max=2.5)
        assert np.all(wb >= 0.25)
        assert np.all(wb <= 4.0)

    def test_underexposed_fallback(self):
        """Near-black image → auto_wb returns [1,1,1] (no correction)."""
        dmin = np.array([0.5, 0.4, 0.3], dtype=np.float32)
        img = np.full((100, 100, 3), 0.001, dtype=np.float32)  # almost black

        wb = auto_wb(img, dmin, d_max=1.5)
        np.testing.assert_array_equal(wb, [1.0, 1.0, 1.0])
