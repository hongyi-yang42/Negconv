"""Tests for highlight recovery."""

import numpy as np
import pytest

from negconv.color import recover_highlights


class TestHighlightRecovery:
    def test_reconstructs_clipped_channel(self):
        """R clipped to 1.0, G/B at 0.5 → R should be ≈ 0.5."""
        h, w = 32, 32
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        # Clip R in center 4x4
        img[14:18, 14:18, 0] = 1.0

        result = recover_highlights(img, threshold=0.99)
        # Center pixel R should be recovered toward 0.5
        assert result[16, 16, 0] < 0.7
        # G and B unchanged
        assert result[16, 16, 1] == pytest.approx(0.5, abs=0.01)
        assert result[16, 16, 2] == pytest.approx(0.5, abs=0.01)

    def test_no_change_unclipped(self):
        """All channels below threshold → output == input."""
        img = np.random.rand(16, 16, 3).astype(np.float32) * 0.5
        result = recover_highlights(img, threshold=0.99)
        np.testing.assert_array_equal(result, img)

    def test_all_clipped_unchanged(self):
        """All 3 channels clipped → no information to recover, unchanged."""
        img = np.random.rand(16, 16, 3).astype(np.float32) * 0.5
        img[8, 8, :] = 1.0  # all channels clipped at one pixel
        result = recover_highlights(img, threshold=0.99)
        # The all-clipped pixel should remain unchanged
        assert result[8, 8, 0] == 1.0
        assert result[8, 8, 1] == 1.0
        assert result[8, 8, 2] == 1.0
