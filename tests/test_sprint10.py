"""Sprint 10 tests: roll analysis and border buffer."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from negconv.params import (
    detect_dmin, detect_dmin_percentile, auto_detect, NegconvParams,
    analyze_roll, RollProfile, detect_border_region,
)


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


class TestRollAnalysis:
    def test_roll_analysis_consistent_dmin(self):
        """5 synthetic images with consistent Dmin: roll Dmin within 5% of means."""
        border_rgb = (0.15, 0.04, 0.01)
        images = [_make_negative_with_border(border_rgb=border_rgb) for _ in range(5)]
        profile = analyze_roll(images)

        mean_dmin = np.mean([detect_dmin_percentile(img) for img in images], axis=0)
        np.testing.assert_allclose(profile.roll_dmin, mean_dmin, rtol=0.05)
        assert profile.num_frames == 5
        assert len(profile.outlier_indices) == 0

    def test_roll_analysis_outlier_detection(self):
        """4 normal + 1 outlier frame: outlier should be flagged."""
        border_rgb = (0.15, 0.04, 0.01)
        images = [_make_negative_with_border(border_rgb=border_rgb) for _ in range(4)]

        # Outlier: much brighter Dmin (different film stock)
        outlier = _make_negative_with_border(border_rgb=(0.40, 0.20, 0.10))
        images.append(outlier)

        profile = analyze_roll(images)
        assert profile.num_frames == 5
        assert 4 in profile.outlier_indices, f"Expected frame 4 flagged, got {profile.outlier_indices}"

    def test_roll_profile_save_load(self):
        """RollProfile roundtrip through JSON."""
        profile = RollProfile(
            roll_dmin=np.array([0.15, 0.04, 0.01], dtype=np.float32),
            roll_wb_high=np.array([1.0, 1.0, 1.5], dtype=np.float32),
            roll_exposure_offset=0.0,
            num_frames=10,
            outlier_indices=[3, 7],
            per_frame_dmin=[[0.15, 0.04, 0.01]] * 10,
            per_frame_wb=[[1.0, 1.0, 1.5]] * 10,
            per_frame_exposure=[1.0] * 10,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roll_profile.json"
            profile.save(path)
            loaded = RollProfile.load(path)

            np.testing.assert_array_equal(profile.roll_dmin, loaded.roll_dmin)
            np.testing.assert_array_equal(profile.roll_wb_high, loaded.roll_wb_high)
            assert loaded.num_frames == 10
            assert loaded.outlier_indices == [3, 7]

    def test_match_params_across_files(self):
        """Match: apply current params sidecars to multiple files."""
        import json
        from negconv.gui.app import create_app

        app = create_app()
        client = app.test_client()

        # Create temp TIFFs
        import tifffile
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                p = Path(tmp) / f"frame_{i}.tif"
                data = np.full((100, 100, 3), 30000, dtype=np.uint16)
                tifffile.imwrite(str(p), data, photometric="rgb")
                paths.append(str(p))

            # Load first file
            resp = client.post("/api/load", json={"path": paths[0]})
            assert resp.status_code == 200

            # Match to all files
            resp = client.post("/api/match-params", json={"indices": [0, 1, 2]})
            data = resp.get_json()
            assert data["count"] == 3

            # Verify sidecars were created
            for p in paths:
                sp = Path(p).with_suffix(".tif.negconv.json")
                assert sp.is_file(), f"Sidecar missing for {p}"
                with open(sp) as f:
                    sidecar = json.load(f)
                assert "dmin" in sidecar


class TestBorderBuffer:
    def test_border_buffer_excludes_holder(self):
        """Synthetic image with bright border: detect_border_region should find content rect."""
        # Create image with bright unexposed border and darker interior
        h, w = 200, 200
        img = np.full((h, w, 3), 0.05, dtype=np.float32)
        # Bright border (film base)
        border = np.array([0.15, 0.04, 0.01], dtype=np.float32)
        img[:15, :] = border
        img[-15:, :] = border
        img[:, :15] = border
        img[:, -15:] = border

        rect = detect_border_region(img)
        # Content rect should be inset from edges
        assert rect["x"] >= 0
        assert rect["y"] >= 0
        assert rect["w"] > 0
        assert rect["h"] > 0
        # Should exclude at least some of the border
        assert rect["x"] > 0 or rect["w"] < w

    def test_border_buffer_improves_dmin(self):
        """Dmin from content-excluded border should be more accurate."""
        h, w = 200, 200
        # Film base border at known value
        border_rgb = np.array([0.15, 0.04, 0.01], dtype=np.float32)
        interior_rgb = np.array([0.05, 0.01, 0.003], dtype=np.float32)
        img = np.full((h, w, 3), interior_rgb, dtype=np.float32)
        img[:10, :] = border_rgb
        img[-10:, :] = border_rgb
        img[:, :10] = border_rgb
        img[:, -10:] = border_rgb

        # Dmin from full image should detect border
        dmin_full = detect_dmin(img)
        assert dmin_full is not None

        # With border buffer, sampling the interior region should NOT
        # return the border values
        rect = detect_border_region(img, border_px=12)
        inner = img[rect["y"]:rect["y"]+rect["h"], rect["x"]:rect["x"]+rect["w"]]
        dmin_inner = detect_dmin(inner)
        # Inner region has no film base border, so should fail or differ
        if dmin_inner is not None:
            # If it detects something, it should NOT match the border
            assert not np.allclose(dmin_inner, border_rgb, atol=0.01)
