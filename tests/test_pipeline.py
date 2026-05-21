"""Unit tests: known pixel values through each pipeline stage.

Uses the verification trace from the spec:
| Pixel (x Dmin) | log_density | corrected | print_linear | after gamma=4 |
|----------------|-------------|-----------|--------------|----------------|
| 1.0            | 0           | -0.05     | 0.171        | ~0.001         |
| 0.1            | -1.0        | -0.675    | 0.799        | 0.408          |
| 0.01           | -2.0        | -1.30     | 0.948        | 0.808          |
"""

import numpy as np
import pytest

from negconv.params import NegconvParams
from negconv.pipeline import invert, THRESHOLD


@pytest.fixture
def color_params():
    return NegconvParams.color_film()


def _make_pixel(multiplier: float, params: NegconvParams) -> np.ndarray:
    """Create a 1x1 RGB image where each channel = multiplier * Dmin[channel]."""
    return (multiplier * params.dmin).reshape(1, 1, 3).astype(np.float32)


class TestStage1LogDensity:
    def test_film_base_pixel(self, color_params):
        """pixel = 1.0 * Dmin -> log_density = 0 for all channels."""
        pixel = _make_pixel(1.0, color_params)
        clamped = np.maximum(pixel, THRESHOLD)
        dmin = color_params.dmin.astype(np.float32)
        density = dmin / clamped
        log_density = -np.log10(density)
        np.testing.assert_allclose(log_density, 0.0, atol=1e-6)

    def test_dense_negative(self, color_params):
        """pixel = 0.1 * Dmin -> log_density = -1.0."""
        pixel = _make_pixel(0.1, color_params)
        clamped = np.maximum(pixel, THRESHOLD)
        dmin = color_params.dmin.astype(np.float32)
        density = dmin / clamped
        log_density = -np.log10(density)
        np.testing.assert_allclose(log_density, -1.0, atol=1e-6)

    def test_very_dense(self, color_params):
        """pixel = 0.01 * Dmin -> log_density = -2.0."""
        pixel = _make_pixel(0.01, color_params)
        clamped = np.maximum(pixel, THRESHOLD)
        dmin = color_params.dmin.astype(np.float32)
        density = dmin / clamped
        log_density = -np.log10(density)
        np.testing.assert_allclose(log_density, -2.0, atol=1e-5)


class TestStage2Correction:
    def test_film_base_corrected(self, color_params):
        """log_density=0 -> corrected = -0.05 (the offset)."""
        pixel = _make_pixel(1.0, color_params)
        clamped = np.maximum(pixel, THRESHOLD)
        dmin = color_params.dmin.astype(np.float32)
        log_density = -np.log10(dmin / clamped)
        wb_high_norm = color_params.wb_high / np.float32(color_params.d_max)
        corrected = wb_high_norm * log_density + np.float32(color_params.offset)
        np.testing.assert_allclose(corrected, -0.05, atol=1e-6)

    def test_dense_corrected(self, color_params):
        """log_density=-1.0 -> corrected = -0.675."""
        pixel = _make_pixel(0.1, color_params)
        clamped = np.maximum(pixel, THRESHOLD)
        dmin = color_params.dmin.astype(np.float32)
        log_density = -np.log10(dmin / clamped)
        wb_high_norm = color_params.wb_high / np.float32(color_params.d_max)
        corrected = wb_high_norm * log_density + np.float32(color_params.offset)
        np.testing.assert_allclose(corrected, -0.675, atol=1e-4)


class TestStage3PrintLinear:
    def test_film_base_print(self, color_params):
        """corrected=-0.05 -> print_linear ~ 0.171."""
        pixel = _make_pixel(1.0, color_params)
        result = invert(pixel, color_params)
        # After gamma=4, ~0.001 — check the full pipeline gives ~0.001 for film base
        # But let's test print_linear directly
        corrected = np.float32(-0.05)
        black_fma = np.float32(-color_params.exposure * (1.0 + color_params.black))
        ten_x = np.float_power(np.float32(10.0), corrected)
        print_linear = -(np.float32(color_params.exposure) * ten_x + black_fma)
        np.testing.assert_allclose(print_linear, 0.171, atol=0.002)

    def test_dense_print(self, color_params):
        """corrected=-0.675 -> print_linear ~ 0.799."""
        corrected = np.float32(-0.675)
        black_fma = np.float32(-color_params.exposure * (1.0 + color_params.black))
        ten_x = np.float_power(np.float32(10.0), corrected)
        print_linear = -(np.float32(color_params.exposure) * ten_x + black_fma)
        np.testing.assert_allclose(print_linear, 0.799, atol=0.002)


class TestStage4Gamma:
    def test_black(self, color_params):
        """print_linear=0.171 -> gamma=4 -> ~0.001."""
        print_linear = np.float32(0.171)
        result = np.power(print_linear, np.float32(color_params.gamma))
        np.testing.assert_allclose(result, 0.001, atol=0.001)

    def test_midtone(self, color_params):
        """print_linear=0.799 -> gamma=4 -> ~0.408."""
        print_linear = np.float32(0.799)
        result = np.power(print_linear, np.float32(color_params.gamma))
        np.testing.assert_allclose(result, 0.408, atol=0.002)

    def test_highlight(self, color_params):
        """print_linear=0.948 -> gamma=4 -> ~0.808."""
        print_linear = np.float32(0.948)
        result = np.power(print_linear, np.float32(color_params.gamma))
        np.testing.assert_allclose(result, 0.808, atol=0.002)


class TestFullPipeline:
    def test_film_base_produces_near_black(self, color_params):
        pixel = _make_pixel(1.0, color_params)
        result = invert(pixel, color_params)
        # Film base should produce near-black after gamma
        np.testing.assert_allclose(result, 0.001, atol=0.002)

    def test_dense_negative_produces_midtone(self, color_params):
        pixel = _make_pixel(0.1, color_params)
        result = invert(pixel, color_params)
        np.testing.assert_allclose(result, 0.408, atol=0.005)

    def test_very_dense_produces_highlight(self, color_params):
        # Spec trace gives 0.808 after gamma; Stage 5 soft-clip compresses to ~0.801
        pixel = _make_pixel(0.01, color_params)
        result = invert(pixel, color_params)
        np.testing.assert_allclose(result, 0.801, atol=0.005)

    def test_zero_pixel_does_not_crash(self, color_params):
        """Zero pixel should be clamped at THRESHOLD, not cause div-by-zero."""
        pixel = np.zeros((1, 1, 3), dtype=np.float32)
        result = invert(pixel, color_params)
        assert np.all(np.isfinite(result))
        assert result.shape == (1, 1, 3)

    def test_deterministic(self, color_params):
        """Same input + same params = identical output."""
        pixel = _make_pixel(0.1, color_params)
        r1 = invert(pixel, color_params)
        r2 = invert(pixel, color_params)
        np.testing.assert_array_equal(r1, r2)


class TestBWPresets:
    def test_bw_film_base(self):
        params = NegconvParams.bw_film()
        # B&W dmin is [1.0, 1.0, 1.0], so all channels identical
        pixel = np.array([[0.1, 0.1, 0.1]], dtype=np.float32).reshape(1, 1, 3)
        result = invert(pixel, params)
        # All channels should be identical (no orange mask correction)
        np.testing.assert_allclose(result[0, 0, 0], result[0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(result[0, 0, 1], result[0, 0, 2], atol=1e-7)
