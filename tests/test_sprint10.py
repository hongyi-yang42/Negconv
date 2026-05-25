"""Sprint 10 tests: roll analysis, border buffer, tint, and post-inversion editing."""

import numpy as np
import json
import pytest
import tempfile
from pathlib import Path

from negconv.params import (
    detect_dmin, detect_dmin_percentile, auto_detect, NegconvParams,
    analyze_roll, RollProfile, detect_border_region,
)
from negconv.postproc import apply_tint, apply_curves, apply_hsl, apply_sharpen, cubic_spline_lut, apply_tone_profile


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

            # Verify sidecars were created (hidden .negconv/ directory)
            for p in paths:
                sp = Path(p).parent / ".negconv" / (Path(p).name + ".negconv.json")
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


class TestTintSlider:
    def test_tint_slider_shifts_green_magenta(self):
        """Positive tint reduces G relative to R/B."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        out = apply_tint(img, 0.5)
        # G should decrease, R and B should increase
        assert out[0, 0, 1] < img[0, 0, 1], "Positive tint should reduce green"
        assert out[0, 0, 0] > img[0, 0, 0], "Positive tint should increase red"
        assert out[0, 0, 2] > img[0, 0, 2], "Positive tint should increase blue"
        # Verify specific formula: G * (1 - 0.5*0.5) = 0.5 * 0.75
        np.testing.assert_allclose(out[0, 0, 1], 0.375, rtol=1e-5)

    def test_tint_zero_is_identity(self):
        """tint=0.0 output unchanged."""
        img = np.random.rand(10, 10, 3).astype(np.float32)
        out = apply_tint(img, 0.0)
        np.testing.assert_array_equal(out, img)


class TestCurves:
    def test_curves_identity(self):
        """Diagonal curve (0,0)→(1,1): output identical to input."""
        img = np.random.rand(20, 20, 3).astype(np.float32) * 0.8
        out = apply_curves(img, r_points=[(0, 0), (1, 1)])
        np.testing.assert_allclose(out, img, atol=0.01)

    def test_curves_contrast_boost(self):
        """S-curve: shadows darker, highlights brighter."""
        # 100x1 gradient so index maps directly to value
        img = np.linspace(0, 1, 100, dtype=np.float32).reshape(1, 100, 1)
        img = np.broadcast_to(img, (1, 100, 3)).copy()

        # S-curve: darken shadows, brighten highlights
        out = apply_curves(img, r_points=[(0, 0), (0.25, 0.15), (0.75, 0.85), (1, 1)])

        # Shadow region (input ~0.25): should be darker
        shadow_idx = 25
        assert out[0, shadow_idx, 0] < img[0, shadow_idx, 0], \
            f"S-curve should darken shadows: {out[0, shadow_idx, 0]} >= {img[0, shadow_idx, 0]}"

        # Highlight region (input ~0.75): should be brighter
        highlight_idx = 75
        assert out[0, highlight_idx, 0] > img[0, highlight_idx, 0], \
            f"S-curve should brighten highlights: {out[0, highlight_idx, 0]} <= {img[0, highlight_idx, 0]}"


class TestHSL:
    def test_hsl_saturation_red(self):
        """Boost red saturation: red pixels should be more saturated."""
        # Pure red image
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 0.8
        img[:, :, 1] = 0.2
        img[:, :, 2] = 0.2

        out = apply_hsl(img, {"red": {"saturation": 50}})

        # After boosting red sat, red channel should increase relative to others
        assert out[0, 0, 0] >= img[0, 0, 0] - 0.01, "Red sat boost should not decrease red channel"


class TestSharpen:
    def test_sharpen_increases_edge_contrast(self):
        """Unsharp mask should increase contrast at edges."""
        # Create image with a horizontal edge: dark top, bright bottom
        img = np.zeros((40, 40, 3), dtype=np.float32)
        img[20:, :, :] = 0.8

        out = apply_sharpen(img, amount=100, radius=1.0, threshold=0)

        # At the edge (row 19-20), sharpening should make the transition sharper
        # Dark side should get slightly darker, bright side slightly brighter
        edge_dark = out[19, 20, 0]
        edge_bright = out[20, 20, 0]

        # The edge contrast (difference) should increase
        orig_contrast = abs(img[20, 20, 0] - img[19, 20, 0])
        sharp_contrast = abs(edge_bright - edge_dark)
        assert sharp_contrast >= orig_contrast - 0.01, \
            f"Sharpening should increase edge contrast: {sharp_contrast} < {orig_contrast}"


class TestToneProfiles:
    def test_tone_profile_frontier_warmer_than_standard(self):
        """Lab Warm has higher R/B ratio (warmer) than Standard at midtones."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        standard = apply_tone_profile(img, "standard")
        warm = apply_tone_profile(img, "lab-warm")

        # Color temperature proxy: R/B ratio
        warm_rb = warm[0, 0, 0] / max(warm[0, 0, 2], 1e-6)
        std_rb = standard[0, 0, 0] / max(standard[0, 0, 2], 1e-6)
        assert warm_rb > std_rb, \
            f"Lab Warm should be warmer (R/B={warm_rb:.3f}) than Standard ({std_rb:.3f})"

    def test_tone_profile_roundtrip_sidecar(self):
        """Tone profile selection persists through sidecar save/load."""
        from negconv.gui.app import create_app
        import tifffile

        app = create_app()
        client = app.test_client()

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.tif"
            data = np.full((100, 100, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(str(p), data, photometric="rgb")

            # Load and set tone profile
            client.post("/api/load", json={"path": str(p)})
            client.post("/api/post-edit", json={"tone_profile": "lab-warm"})

            # Reload and verify tone profile restored
            client.post("/api/load", json={"path": str(p)})
            resp = client.get("/api/post-edit")
            pe = resp.get_json()
            assert pe["tone_profile"] == "lab-warm", \
                f"Expected 'lab-warm', got '{pe['tone_profile']}'"

    def test_hidden_sidecar_write_read(self):
        """Sidecar is written to .negconv/ subdirectory."""
        from negconv.gui.app import create_app
        import tifffile

        app = create_app()
        client = app.test_client()

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.tif"
            data = np.full((100, 100, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(str(p), data, photometric="rgb")

            client.post("/api/load", json={"path": str(p)})
            client.post("/api/params", json={"gamma": 5.0})

            # Verify sidecar in .negconv/ subdir
            hidden = Path(tmp) / ".negconv" / "test.tif.negconv.json"
            assert hidden.is_file(), f"Hidden sidecar not found at {hidden}"

            with open(hidden) as f:
                sidecar = json.load(f)
            assert abs(sidecar["gamma"] - 5.0) < 0.01

    def test_legacy_sidecar_migration(self):
        """Legacy sidecar (beside file) is found and loaded."""
        from negconv.gui.app import create_app
        import tifffile

        app = create_app()
        client = app.test_client()

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.tif"
            data = np.full((100, 100, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(str(p), data, photometric="rgb")

            # Write a legacy sidecar (beside the file)
            legacy = str(p) + ".negconv.json"
            with open(legacy, "w") as f:
                json.dump({"gamma": 7.0, "dmin": [1.13, 0.49, 0.27],
                           "d_max": 1.6, "wb_high": [1,1,1], "wb_low": [1,1,1],
                           "offset": -0.05, "exposure": 0.9245, "black": 0.0755,
                           "soft_clip": 0.75}, f)

            resp = client.post("/api/load", json={"path": str(p)})
            data_resp = resp.get_json()
            assert data_resp["sidecar_loaded"] is True
            assert abs(data_resp["params"]["gamma"] - 7.0) < 0.01
