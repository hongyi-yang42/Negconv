"""Sprint 4 tests: GUI viewer conversion, Flask routes, eyedropper, export."""

import io
import json
import os
import tempfile

import numpy as np
import pytest
import tifffile
from PIL import Image

from negconv.gui.app import create_app, _sample_dmin, _preview_to_orig_coords
from negconv.gui.viewer import linear_to_srgb, make_preview


class TestViewerSRGB:
    def test_midgray(self):
        """0.5 linear → ~188 sRGB."""
        result = linear_to_srgb(np.array([0.5], dtype=np.float32))
        val = int(result[0] * 255 + 0.5)
        assert 186 <= val <= 190

    def test_black(self):
        """0.0 linear → 0 sRGB."""
        result = linear_to_srgb(np.array([0.0], dtype=np.float32))
        assert result[0] == 0.0

    def test_white(self):
        """1.0 linear → ~1.0 sRGB (float32 precision)."""
        result = linear_to_srgb(np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(result[0], 1.0, atol=1e-6)

    def test_shape_preserved(self):
        """Output shape matches input."""
        img = np.random.rand(10, 20, 3).astype(np.float32)
        result = linear_to_srgb(img)
        assert result.shape == img.shape


class TestMakePreview:
    def test_returns_jpeg_bytes(self):
        """Output starts with JPEG magic bytes."""
        img = np.random.rand(100, 100, 3).astype(np.float32) * 0.5
        data = make_preview(img)
        assert data[:2] == b'\xff\xd8'  # JPEG SOI marker

    def test_respects_max_width(self):
        """Preview width does not exceed max_width."""
        img = np.random.rand(100, 3000, 3).astype(np.float32)
        data = make_preview(img, max_width=800)
        pil = Image.open(io.BytesIO(data))
        assert pil.width <= 800

    def test_no_resize_when_small(self):
        """Small images are not resized."""
        img = np.random.rand(50, 100, 3).astype(np.float32)
        data = make_preview(img, max_width=1200)
        pil = Image.open(io.BytesIO(data))
        assert pil.width == 100


class TestSampleDmin:
    def test_uniform_patch(self):
        """Sampling a uniform image returns the uniform value."""
        img = np.full((100, 100, 3), [0.2, 0.3, 0.4], dtype=np.float32)
        dmin = _sample_dmin(img, 50, 50)
        np.testing.assert_allclose(dmin, [0.2, 0.3, 0.4], atol=0.001)

    def test_edge_coords(self):
        """Sampling near edge doesn't crash."""
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        dmin = _sample_dmin(img, 0, 0)
        assert dmin.shape == (3,)
        assert np.all(np.isfinite(dmin))


class TestCoordMapping:
    def test_center_pixel(self):
        """Center of preview maps to center of original."""
        ox, oy = _preview_to_orig_coords(600, 400, 1200, 800, 4000, 3000)
        assert ox == 2000
        assert oy == 1500

    def test_top_left(self):
        ox, oy = _preview_to_orig_coords(0, 0, 1200, 800, 4000, 3000)
        assert ox == 0
        assert oy == 0

    def test_bottom_right(self):
        ox, oy = _preview_to_orig_coords(1199, 799, 1200, 800, 4000, 3000)
        assert ox >= 3990
        assert oy >= 2990


class TestFlaskRoutes:
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def test_index(self, client):
        """GET / returns 200 with HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"negconv" in resp.data

    def test_load_missing_file(self, client):
        """POST /api/load with bad path returns error."""
        resp = client.post("/api/load",
                           json={"path": "/nonexistent/file.ARW"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_load_tiff(self, client):
        """POST /api/load with valid TIFF returns preview info."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.full((200, 200, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(f.name, data, photometric="rgb")
            path = f.name

        try:
            resp = client.post("/api/load", json={"path": path})
            assert resp.status_code == 200
            result = resp.get_json()
            assert "preview" in result
            assert "params" in result
            assert result["dims"] == [200, 200]
        finally:
            os.unlink(path)

    def test_preview_orig(self, client):
        """GET /api/preview/orig returns JPEG after loading."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.full((100, 100, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(f.name, data, photometric="rgb")
            path = f.name

        try:
            client.post("/api/load", json={"path": path})
            resp = client.get("/api/preview/orig")
            assert resp.status_code == 200
            assert resp.content_type == "image/jpeg"
        finally:
            os.unlink(path)

    def test_invert_without_load(self, client):
        """POST /api/invert without loading first returns error."""
        resp = client.post("/api/invert")
        assert resp.status_code == 400

    def test_invert_after_load(self, client):
        """POST /api/invert after load returns result preview."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.full((100, 100, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(f.name, data, photometric="rgb")
            path = f.name

        try:
            client.post("/api/load", json={"path": path})
            resp = client.post("/api/invert")
            assert resp.status_code == 200
            result = resp.get_json()
            assert result["preview"] == "/api/preview/result"
        finally:
            os.unlink(path)

    def test_pick_dmin(self, client):
        """POST /api/pick-dmin samples the image and returns Dmin."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.full((200, 200, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(f.name, data, photometric="rgb")
            path = f.name

        try:
            client.post("/api/load", json={"path": path})
            resp = client.post("/api/pick-dmin", json={"x": 50, "y": 50})
            assert resp.status_code == 200
            result = resp.get_json()
            assert "dmin" in result
            assert len(result["dmin"]) == 3
            # Should be close to the uniform value 30000/65535 ≈ 0.458
            assert all(v > 0.1 for v in result["dmin"])
        finally:
            os.unlink(path)

    def test_params_round_trip(self, client):
        """GET params → POST modified → GET matches."""
        resp = client.get("/api/params")
        params = resp.get_json()

        params["gamma"] = 5.5
        params["exposure"] = 1.2
        client.post("/api/params",
                    json={"gamma": 5.5, "exposure": 1.2})

        resp2 = client.get("/api/params")
        updated = resp2.get_json()
        assert updated["gamma"] == 5.5
        assert updated["exposure"] == 1.2

    def test_preset_color(self, client):
        """POST /api/preset/color resets to C-41 defaults."""
        resp = client.post("/api/preset/color")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["gamma"] == 4.0

    def test_preset_bw(self, client):
        """POST /api/preset/bw resets to B&W defaults."""
        resp = client.post("/api/preset/bw")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["gamma"] == 5.0

    def test_export(self, client):
        """POST /api/export returns TIFF download."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.full((100, 100, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(f.name, data, photometric="rgb")
            path = f.name

        try:
            client.post("/api/load", json={"path": path})
            client.post("/api/invert")
            resp = client.post("/api/export", json={"dtype": "uint16"})
            assert resp.status_code == 200
            assert "tiff" in resp.content_type or "octet-stream" in resp.content_type
        finally:
            os.unlink(path)


class TestCrop:
    @pytest.fixture
    def client_with_image(self):
        """Client with a loaded 200x200 TIFF."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.full((200, 200, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(f.name, data, photometric="rgb")
            path = f.name
        client.post("/api/load", json={"path": path})
        yield client
        os.unlink(path)

    def test_crop_rect_stored_in_state(self, client_with_image):
        """POST /api/crop stores crop_rect, GET /api/params includes it."""
        resp = client_with_image.post("/api/crop", json={
            "x": 10, "y": 20, "w": 100, "h": 80,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["crop_rect"] == {"x": 10, "y": 20, "w": 100, "h": 80}

        resp2 = client_with_image.get("/api/params")
        params = resp2.get_json()
        assert params["crop_rect"] == {"x": 10, "y": 20, "w": 100, "h": 80}

    def test_pipeline_scoped_to_crop(self, client_with_image):
        """With crop set, invert produces a valid preview."""
        client_with_image.post("/api/crop", json={
            "x": 0, "y": 0, "w": 50, "h": 200,
        })
        resp = client_with_image.post("/api/invert")
        assert resp.status_code == 200
        resp2 = client_with_image.get("/api/preview/result")
        assert resp2.status_code == 200
        assert resp2.content_type == "image/jpeg"

    def test_params_include_crop(self, client_with_image):
        """After setting crop, /api/params includes crop_rect."""
        client_with_image.post("/api/crop", json={
            "x": 10, "y": 10, "w": 180, "h": 180,
        })
        resp = client_with_image.get("/api/params")
        data = resp.get_json()
        assert data["crop_rect"] is not None
        assert data["crop_rect"]["w"] == 180

    def test_eyedropper_outside_crop(self, client_with_image):
        """Eyedropper click outside crop_rect still updates Dmin."""
        client_with_image.post("/api/crop", json={
            "x": 0, "y": 0, "w": 50, "h": 50,
        })
        resp = client_with_image.post("/api/pick-dmin", json={"x": 200, "y": 200})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "dmin" in data
        assert len(data["dmin"]) == 3

    def test_clear_crop_reverts(self, client_with_image):
        """DELETE /api/crop clears crop, pipeline runs on full image."""
        client_with_image.post("/api/crop", json={
            "x": 10, "y": 10, "w": 50, "h": 50,
        })
        resp = client_with_image.delete("/api/crop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["crop_rect"] is None

        resp2 = client_with_image.get("/api/params")
        params = resp2.get_json()
        assert params.get("crop_rect") is None

    def test_crop_coord_mapping(self, client_with_image):
        """Crop coords sent to API are in original image space."""
        resp = client_with_image.post("/api/crop", json={
            "x": 100, "y": 0, "w": 100, "h": 200,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["crop_rect"]["x"] == 100
        assert data["crop_rect"]["w"] == 100
