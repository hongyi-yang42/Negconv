"""Sprint 2 tests: auto-detect Dmin/Dmax, RAW support, CLI flags."""

import numpy as np
import pytest

from negconv.params import detect_dmin, detect_dmax, auto_detect, NegconvParams


def _make_negative_with_border(
    h=200, w=200, border_px=10,
    border_rgb=(0.15, 0.04, 0.01),
    interior_rgb=(0.05, 0.01, 0.003),
):
    """Create a synthetic negative with a bright film base border."""
    img = np.full((h, w, 3), interior_rgb, dtype=np.float32)
    border = np.array(border_rgb, dtype=np.float32)
    img[:border_px, :] = border
    img[-border_px:, :] = border
    img[:, :border_px] = border
    img[:, -border_px:] = border
    return img


class TestDetectDmin:
    def test_detects_border_values(self):
        """Auto Dmin should match the bright border pixel values."""
        border_rgb = (0.15, 0.04, 0.01)
        img = _make_negative_with_border(border_rgb=border_rgb)
        dmin = detect_dmin(img)
        assert dmin is not None
        np.testing.assert_allclose(dmin, border_rgb, atol=0.01)

    def test_returns_none_for_no_border(self):
        """Image cropped to frame (no film border) → None."""
        # Uniform image — border and interior identical
        img = np.full((200, 200, 3), 0.05, dtype=np.float32)
        dmin = detect_dmin(img)
        assert dmin is None

    def test_ignores_uniform_image(self):
        """Uniform image (R=G=B, no orange mask) → None."""
        img = np.full((200, 200, 3), 0.05, dtype=np.float32)
        dmin = detect_dmin(img)
        assert dmin is None

    def test_detects_border_even_if_darker_than_interior(self):
        """Camera scans: border darker overall but still valid film base (above 0.05 max)."""
        img = _make_negative_with_border(
            border_rgb=(0.10, 0.03, 0.01),
            interior_rgb=(0.15, 0.04, 0.01),
        )
        dmin = detect_dmin(img)
        assert dmin is not None
        np.testing.assert_allclose(dmin, [0.10, 0.03, 0.01], atol=0.002)

    def test_returns_none_for_film_holder_black_border(self):
        """Camera scans with film holder (black edges) → None with warning."""
        img = _make_negative_with_border(
            border_rgb=(0.01, 0.005, 0.002),
            interior_rgb=(0.10, 0.03, 0.01),
        )
        dmin = detect_dmin(img)
        assert dmin is None

    def test_different_border_width(self):
        """Should work with different border fractions."""
        img = _make_negative_with_border(h=500, w=500, border_px=25)
        dmin = detect_dmin(img, border_frac=0.05)
        assert dmin is not None
        np.testing.assert_allclose(dmin[0], 0.15, atol=0.01)


class TestDetectDmax:
    def test_computes_density_range(self):
        """Dmax = log10(Dmin / min_pixel) for the densest channel."""
        img = _make_negative_with_border(
            border_rgb=(0.15, 0.04, 0.01),
            interior_rgb=(0.001, 0.001, 0.001),  # very dark interior
        )
        dmin = np.array([0.15, 0.04, 0.01], dtype=np.float32)
        dmax = detect_dmax(img, dmin)
        # log10(0.15 / 0.001) = log10(150) ≈ 2.18
        assert 1.5 < dmax < 3.0

    def test_small_range(self):
        """Low contrast image → small Dmax, floored at 0.5."""
        img = _make_negative_with_border(
            border_rgb=(0.15, 0.04, 0.01),
            interior_rgb=(0.10, 0.03, 0.008),
        )
        dmin = np.array([0.15, 0.04, 0.01], dtype=np.float32)
        dmax = detect_dmax(img, dmin)
        assert 0.5 <= dmax < 1.5


class TestAutoDetect:
    def test_with_border(self):
        """Full auto-detect with film border present."""
        img = _make_negative_with_border()
        params = auto_detect(img)
        # Should detect Dmin close to border values
        np.testing.assert_allclose(params.dmin[0], 0.15, atol=0.01)

    def test_without_border_fallback(self):
        """No film border → percentile fallback gives image statistics."""
        img = np.full((200, 200, 3), 0.05, dtype=np.float32)
        params = auto_detect(img, fallback_preset="color")
        # Percentile of uniform 0.05 image = 0.05
        np.testing.assert_allclose(params.dmin, [0.05, 0.05, 0.05], atol=0.01)

    def test_without_border_manual_mode(self):
        """Manual mode skips detection, returns preset defaults."""
        img = np.full((200, 200, 3), 0.05, dtype=np.float32)
        params = auto_detect(img, fallback_preset="color", dmin_mode="manual")
        np.testing.assert_allclose(params.dmin, [1.13, 0.49, 0.27], atol=0.01)

    def test_bw_fallback(self):
        """Can fall back to B&W preset in manual mode."""
        img = np.full((200, 200, 3), 0.05, dtype=np.float32)
        params = auto_detect(img, fallback_preset="bw", dmin_mode="manual")
        np.testing.assert_allclose(params.dmin, [1.0, 1.0, 1.0], atol=0.01)

    def test_returns_params_object(self):
        img = _make_negative_with_border()
        params = auto_detect(img)
        assert isinstance(params, NegconvParams)


class TestIO:
    def test_is_raw_extensions(self):
        from negconv.io import is_raw
        assert is_raw("scan.CR3")
        assert is_raw("scan.arw")
        assert is_raw("photo.NEF")
        assert is_raw("image.dng")
        assert not is_raw("scan.tif")
        assert not is_raw("photo.jpg")

    def test_read_tiff_linearized(self):
        """Sprint 1 test still works with new read_tiff function."""
        from negconv.io import read_tiff
        import tempfile, os
        import tifffile

        # Create a small test TIFF
        data = np.full((10, 10, 3), 32768, dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tifffile.imwrite(f.name, data, photometric="rgb")
            img = read_tiff(f.name, linearize=False)
            assert img.shape == (10, 10, 3)
            assert img.dtype == np.float32
            np.testing.assert_allclose(img[0, 0], 32768 / 65535.0, atol=0.001)
            os.unlink(f.name)


class TestCLIFlags:
    def test_dmin_override(self):
        """Manual --dmin-r/g/b overrides auto-detect."""
        import numpy as np
        from negconv.params import NegconvParams

        # Simulate what the CLI does
        params = NegconvParams.color_film()
        params.dmin = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        np.testing.assert_allclose(params.dmin, [0.5, 0.3, 0.1], atol=0.001)

    def test_dmax_override(self):
        params = NegconvParams.color_film()
        params.d_max = 2.5
        assert params.d_max == 2.5
