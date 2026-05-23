"""Sprint 7 Phase A tests: Rec.2020 color space + percentile Dmin fallback."""
import numpy as np
import pytest
import tifffile

from negconv.color import (
    MATRIX_REC2020_TO_SRGB,
    MATRIX_SRGB_TO_REC2020,
    convert_color_space,
    rec2020_to_srgb,
    srgb_to_rec2020,
)
from negconv.params import auto_detect, detect_dmin_percentile


class TestColorMatrix:
    def test_srgb_to_rec2020_roundtrip(self):
        """Convert sRGB→Rec.2020→sRGB, verify <1e-5 error."""
        img = np.random.rand(10, 10, 3).astype(np.float32)
        roundtrip = rec2020_to_srgb(srgb_to_rec2020(img))
        np.testing.assert_allclose(roundtrip, img, atol=1e-5)

    def test_matrix_inverse_correct(self):
        """M * M_inv ≈ I."""
        product = MATRIX_SRGB_TO_REC2020 @ MATRIX_REC2020_TO_SRGB
        np.testing.assert_allclose(product, np.eye(3), atol=1e-5)

    def test_convert_preserves_shape(self):
        img = np.random.rand(50, 100, 3).astype(np.float32)
        out = srgb_to_rec2020(img)
        assert out.shape == img.shape

    def test_convert_does_not_clip(self):
        """Matrix conversion doesn't clip — values can exceed [0,1]."""
        img = np.array([[[2.0, 0.1, 0.5]]], dtype=np.float32)
        out = srgb_to_rec2020(img)
        assert out.max() > 1.0  # Out-of-gamut values preserved


class TestRec2020Pipeline:
    def test_pipeline_in_rec2020_no_clipping(self):
        """Full pipeline in Rec.2020 produces valid output for typical negatives."""
        from negconv.pipeline import invert
        from negconv.params import NegconvParams

        # Synthetic negative in Rec.2020 space
        img = np.random.uniform(0.2, 0.8, (50, 50, 3)).astype(np.float32)
        params = NegconvParams(
            dmin=np.array([0.8, 0.5, 0.3], dtype=np.float32),
            d_max=1.5,
        )
        result = invert(img, params)
        assert result.shape == img.shape
        assert np.all(np.isfinite(result))
        # Result should be in reasonable range (no extreme values from color shift)
        assert np.max(np.abs(result)) < 100

    def test_rec2020_hue_preservation(self):
        """Rec.2020 conversion preserves hue better than doing nothing for saturated colors."""
        # Create a highly saturated red in sRGB
        srgb_red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        rec_red = srgb_to_rec2020(srgb_red)

        # After Rec.2020 conversion, red should still be dominant
        assert rec_red[0, 0, 0] > rec_red[0, 0, 1]  # R > G
        assert rec_red[0, 0, 0] > rec_red[0, 0, 2]  # R > B


class TestPercentileDmin:
    def test_percentile_on_bordered_image(self):
        """Image with orange border → percentile estimate close to border values."""
        img = np.full((200, 200, 3), 0.05, dtype=np.float32)
        # Orange border in top 5%
        img[:10, :] = [0.8, 0.5, 0.3]
        dmin = detect_dmin_percentile(img)
        # 99.5th percentile should be near the border values (they're 5% of pixels)
        assert dmin[0] > 0.05  # Should detect the bright border
        assert dmin[0] > dmin[2]  # R > B (orange mask)

    def test_percentile_on_uniform_image(self):
        """Uniform image → percentile = uniform value."""
        img = np.full((100, 100, 3), 0.42, dtype=np.float32)
        dmin = detect_dmin_percentile(img)
        np.testing.assert_allclose(dmin, [0.42, 0.42, 0.42], atol=0.01)

    def test_percentile_cli_flag(self):
        """--dmin-mode percentile bypasses border detection."""
        from negconv.main import _build_parser, _resolve_params
        img = np.full((200, 200, 3), 0.3, dtype=np.float32)
        parser = _build_parser()
        args = parser.parse_args(["test.tif", "-o", "out.tif", "--dmin-mode", "percentile"])
        params = _resolve_params(args, img)
        # Percentile should give 0.3, not the preset default
        np.testing.assert_allclose(params.dmin, [0.3, 0.3, 0.3], atol=0.01)
