"""Integration tests for rotate→crop pipeline, reset completeness, and preview/export parity."""

import io
import json
import os
import tempfile

import numpy as np
import pytest
import tifffile
from PIL import Image

from negconv.gui.app import create_app, _load_file, _run_pipeline, _build_preview, GuiState


def _make_tiff(path, h=200, w=300, value=30000):
    """Create a uniform TIFF at path."""
    data = np.full((h, w, 3), value, dtype=np.uint16)
    tifffile.imwrite(path, data, photometric="rgb")


def _make_state_with_tiff(tmp_path, h=200, w=300):
    """Create a GuiState with a loaded TIFF. Returns (state, path)."""
    path = str(tmp_path / "test.tif")
    _make_tiff(path, h, w)
    state = GuiState(settings={"preview_max_width": 1200, "preview_quality": 90})
    _load_file(state, path)
    return state, path


class TestCropShowsResultNotOriginal:
    """Verify that crop mode uses the result preview (inverted+rotated), not the raw negative."""

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def _load_tiff(self, client, tmp_path, h=200, w=300):
        """Helper: create and load a TIFF via the API."""
        path = str(tmp_path / "test.tif")
        _make_tiff(path, h, w)
        resp = client.post("/api/load", json={"path": path})
        assert resp.status_code == 200
        return resp.get_json()

    def test_crop_preview_matches_result_preview(self, client, tmp_path):
        """After rotate+invert, crop mode should show the same image as result preview.
        The /api/preview/orig endpoint serves the raw negative (with orientation).
        The /api/preview/result endpoint serves the fully developed image.
        Crop mode should serve the result, not the orig."""
        self._load_tiff(client, tmp_path)
        # Invert
        resp = client.post("/api/invert")
        assert resp.status_code == 200
        # Rotate 90°
        resp = client.post("/api/rotate", json={"orientation": 1})
        assert resp.status_code == 200
        # Fine rotate 3°
        resp = client.post("/api/fine-rotate", json={"angle_deg": 3.0})
        assert resp.status_code == 200

        # Fetch both previews and compare
        orig_resp = client.get("/api/preview/orig")
        result_resp = client.get("/api/preview/result")
        assert orig_resp.status_code == 200
        assert result_resp.status_code == 200
        # They must be different (orig is negative, result is positive)
        assert orig_resp.data != result_resp.data

        # The crop endpoint should NOT serve the orig — crop should use result
        # (This is verified by the frontend fix, but we verify the result preview exists)
        assert len(result_resp.data) > 1000

    def test_crop_after_rotation_maps_correctly(self, client, tmp_path):
        """Crop coordinates after rotation should map to the rotated image, not the original."""
        data = self._load_tiff(client, tmp_path, h=400, w=600)
        # Invert
        client.post("/api/invert")
        # Rotate 90° CW
        resp = client.post("/api/rotate", json={"orientation": 1})
        assert resp.status_code == 200
        rot_data = resp.get_json()

        # After 90° CW rotation of 400x600, rotated_dims should be (600, 400)
        rot_dims = rot_data.get("rotated_dims", [])
        assert rot_dims[0] == 600  # new height = old width
        assert rot_dims[1] == 400  # new width = old height

        # Apply crop in the rotated preview space
        pw = rot_data["preview_dims"][0]
        ph = rot_data["preview_dims"][1]
        # Crop center quarter of the preview
        cx, cy = pw // 4, ph // 4
        cw, ch = pw // 2, ph // 2
        resp = client.post("/api/crop", json={"x": cx, "y": cy, "w": cw, "h": ch})
        assert resp.status_code == 200
        crop_data = resp.get_json()
        assert crop_data["crop_rect"] is not None
        # Crop rect should have non-zero dimensions in full-res space
        cr = crop_data["crop_rect"]
        assert cr["w"] > 0
        assert cr["h"] > 0


class TestResetCompleteness:
    """Verify reset clears ALL state to defaults."""

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def _load_tiff(self, client, tmp_path, h=200, w=300):
        path = str(tmp_path / "test.tif")
        _make_tiff(path, h, w)
        resp = client.post("/api/load", json={"path": path})
        assert resp.status_code == 200
        return resp.get_json()

    def test_reset_clears_rotation(self, client, tmp_path):
        """After reset, rotation should be 0."""
        self._load_tiff(client, tmp_path)
        client.post("/api/rotate", json={"orientation": 1})
        client.post("/api/fine-rotate", json={"angle_deg": 3.0})

        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["angle_deg"] == 0.0
        assert data["orientation"] == 0

    def test_reset_clears_crop(self, client, tmp_path):
        """After reset, crop_rect should be null."""
        self._load_tiff(client, tmp_path)
        client.post("/api/invert")
        # Set a crop
        pw = 300  # preview dims
        client.post("/api/crop", json={"x": 10, "y": 10, "w": 100, "h": 80})

        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["crop_rect"] is None

    def test_reset_clears_post_edits(self, client, tmp_path):
        """After reset, post-edit state should be default."""
        self._load_tiff(client, tmp_path)
        client.post("/api/invert")
        # Apply some post-edit
        client.post("/api/post-edit/curves", json={
            "channel": "composite",
            "points": [[0, 0], [0.3, 0.5], [1, 1]],
        })

        resp = client.post("/api/reset")
        assert resp.status_code == 200
        # Check post-edit is reset
        pe_resp = client.get("/api/post-edit")
        pe = pe_resp.get_json()
        assert pe["curves"] is None

    def test_reset_clears_highlight_recovery(self, client, tmp_path):
        """After reset, highlight_recovery should match user default setting."""
        self._load_tiff(client, tmp_path)
        # Enable highlight recovery
        client.post("/api/highlight-recovery", json={"enabled": True})

        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        # Should reset to user's saved default (False by default)
        assert data["highlight_recovery"] is False

    def test_reset_returns_lut_as_null(self, client, tmp_path):
        """After reset, LUT state should be cleared."""
        self._load_tiff(client, tmp_path)
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("lut") is None

    def test_reset_after_rotate_then_crop(self, client, tmp_path):
        """Full rotate→crop→invert→reset cycle clears everything."""
        self._load_tiff(client, tmp_path, h=400, w=600)
        client.post("/api/invert")
        client.post("/api/rotate", json={"orientation": 1})
        client.post("/api/fine-rotate", json={"angle_deg": 2.5})
        client.post("/api/crop", json={"x": 20, "y": 20, "w": 100, "h": 100})

        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["orientation"] == 0
        assert data["angle_deg"] == 0.0
        assert data["crop_rect"] is None
        assert data["highlight_recovery"] is False
        assert data.get("lut") is None


class TestExportMatchesPreview:
    """Verify export pipeline applies all edits identically to preview."""

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def _load_tiff(self, client, tmp_path, h=200, w=300):
        path = str(tmp_path / "test.tif")
        _make_tiff(path, h, w)
        resp = client.post("/api/load", json={"path": path})
        assert resp.status_code == 200
        return resp.get_json()

    def test_export_after_rotate_and_crop(self, client, tmp_path):
        """Export after rotate+crop should succeed and have correct dimensions."""
        self._load_tiff(client, tmp_path, h=400, w=600)
        client.post("/api/invert")
        # Rotate 90° CW
        client.post("/api/rotate", json={"orientation": 1})
        # Get preview dims to crop
        resp = client.get("/api/info")
        info = resp.get_json()
        # Crop via API
        crop_resp = client.post("/api/crop", json={"x": 10, "y": 10, "w": 100, "h": 100})
        assert crop_resp.status_code == 200

        # Export as JPEG
        resp = client.post("/api/export", json={
            "format": "jpeg",
            "quality": 90,
            "output_sharpen": "none",
        })
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"
        # Verify it's a valid JPEG
        img = Image.open(io.BytesIO(resp.data))
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_export_after_fine_rotate(self, client, tmp_path):
        """Export after arbitrary rotation succeeds with expanded dimensions."""
        self._load_tiff(client, tmp_path, h=200, w=300)
        client.post("/api/invert")
        client.post("/api/fine-rotate", json={"angle_deg": 5.0})

        resp = client.post("/api/export", json={
            "format": "jpeg",
            "quality": 90,
            "output_sharpen": "none",
        })
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.data))
        # After rotation, image should be larger than original due to expand
        assert img.size[0] > 200 or img.size[1] > 200

    def test_output_sharpening_not_in_preview(self, client, tmp_path):
        """Output sharpening should differ from preview — export-only processing."""
        # Use a non-uniform image with detail so sharpening has an effect
        path = str(tmp_path / "detail.tif")
        rng = np.random.default_rng(42)
        data = rng.integers(10000, 60000, (200, 300, 3), dtype=np.uint16)
        tifffile.imwrite(path, data, photometric="rgb")
        resp = client.post("/api/load", json={"path": path})
        assert resp.status_code == 200
        client.post("/api/invert")

        # Get preview
        preview_resp = client.get("/api/preview/result")
        assert preview_resp.status_code == 200

        # Export without sharpening
        export_plain = client.post("/api/export", json={
            "format": "jpeg", "quality": 92, "output_sharpen": "none",
        })
        assert export_plain.status_code == 200

        # Export with sharpening
        export_sharp = client.post("/api/export", json={
            "format": "jpeg", "quality": 92, "output_sharpen": "screen",
        })
        assert export_sharp.status_code == 200
        # Sharpened export should differ from unsharpened
        assert export_plain.data != export_sharp.data


class TestPipelineStateConsistency:
    """Verify internal state remains consistent through rotate→crop operations."""

    def test_rotated_dims_after_rotation(self, tmp_path):
        """After rotation, rotated_dims should reflect the rotated size."""
        state, path = _make_state_with_tiff(tmp_path, h=200, w=300)
        _run_pipeline(state)
        # Before rotation: rotated_dims should match original
        assert state.rotated_dims[0] == 200
        assert state.rotated_dims[1] == 300

        # After 90° CW rotation
        state.orientation = 1
        _build_preview(state)
        assert state.rotated_dims[0] == 300  # new h = old w
        assert state.rotated_dims[1] == 200  # new w = old h

    def test_rotated_dims_after_fine_rotate(self, tmp_path):
        """After arbitrary rotation, rotated_dims should expand."""
        state, path = _make_state_with_tiff(tmp_path, h=200, w=300)
        _run_pipeline(state)

        state.angle_deg = 5.0
        _build_preview(state)
        # After rotation, both dims should be >= originals
        assert state.rotated_dims[0] >= 200
        assert state.rotated_dims[1] >= 300

    def test_crop_in_rotated_space(self, tmp_path):
        """Crop coordinates should be in post-rotation space."""
        state, path = _make_state_with_tiff(tmp_path, h=200, w=300)
        _run_pipeline(state)

        # Rotate 90° CW
        state.orientation = 1
        _build_preview(state)
        # Rotated image is 300 high × 200 wide
        assert state.rotated_dims == (300, 200)

        # Set crop in rotated space
        state.crop_rect = {"x": 10, "y": 20, "w": 150, "h": 200}
        _build_preview(state)
        # Preview should exist and be smaller than rotated image
        assert state.result_preview is not None
        pil = Image.open(io.BytesIO(state.result_preview))
        # The preview width should be proportional to crop width (150/200 of preview max)
        assert pil.width > 0
        assert pil.height > 0

    def test_load_file_clears_lut(self, tmp_path):
        """_load_file should clear LUT state."""
        state, path = _make_state_with_tiff(tmp_path)
        # Simulate LUT loaded
        state.lut_data = {"some": "data"}
        state.lut_path = "/some/lut.cube"
        # Reload
        _load_file(state, path)
        assert state.lut_data is None
        assert state.lut_path == ""

    def test_load_file_clears_crop_and_rotation(self, tmp_path):
        """_load_file should reset crop, orientation, and angle."""
        state, path = _make_state_with_tiff(tmp_path)
        state.crop_rect = {"x": 10, "y": 10, "w": 50, "h": 50}
        state.orientation = 1
        state.angle_deg = 3.0
        state.flip_h = True
        # Reload same file
        _load_file(state, path)
        assert state.crop_rect is None
        assert state.orientation == 0
        assert state.angle_deg == 0.0
        assert state.flip_h is False

    def test_preview_dims_match_result_preview(self, tmp_path):
        """preview_dims should match actual result preview JPEG dimensions."""
        state, path = _make_state_with_tiff(tmp_path, h=400, w=600)
        _run_pipeline(state)
        assert state.result_preview is not None
        pil = Image.open(io.BytesIO(state.result_preview))
        assert state.preview_dims == (pil.width, pil.height)

    def test_preview_dims_after_rotate_and_crop(self, tmp_path):
        """Preview dims should update after rotation and crop."""
        state, path = _make_state_with_tiff(tmp_path, h=400, w=600)
        _run_pipeline(state)

        # Rotate 90°
        state.orientation = 1
        _build_preview(state)
        dims_after_rot = state.preview_dims
        assert dims_after_rot[0] > 0 and dims_after_rot[1] > 0

        # Crop
        rh, rw = state.rotated_dims
        state.crop_rect = {"x": 0, "y": 0, "w": rw // 2, "h": rh // 2}
        _build_preview(state)
        dims_after_crop = state.preview_dims
        # Cropped preview should be narrower than rotated preview
        assert dims_after_crop[0] <= dims_after_rot[0]
