"""Sprint 8 tests: Gray point selector, LUT loader, highlight recovery."""

import io
import json
import tempfile

import numpy as np
import pytest

from negconv.gui.app import create_app


class TestPickGray:
    """Tests for /api/pick-gray endpoint."""

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def _load_test_image(self, client, img_array):
        """Helper: save array as TIFF, load via API, return response."""
        import tifffile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tifffile.imwrite(f.name, (img_array * 65535).astype(np.uint16))
            return client.post("/api/load", json={"path": f.name})

    def test_pick_gray_neutralizes_cast(self, client):
        """R channel denser than G/B → wb_high[R] corrected relative to green."""
        h, w = 64, 64
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[:, :, 0] = 0.1  # R: dark in negative (dense, bright scene)
        img[:, :, 1] = 0.3  # G: brighter in negative
        img[:, :, 2] = 0.3  # B

        self._load_test_image(client, img)

        # Override dmin so we know exact values
        client.post("/api/params", json={"dmin": [0.6, 0.6, 0.6]})
        client.post("/api/invert", method="POST")

        resp = client.post("/api/pick-gray", json={"x": 32, "y": 32})
        assert resp.status_code == 200
        data = resp.get_json()
        wb = data["params"]["wb_high"]

        # Green is the anchor — should stay at 1.0
        assert abs(wb[1] - 1.0) < 0.01
        # R was denser → wb_high[R] should be < wb_high[G]
        assert wb[0] < wb[1]

    def test_pick_gray_samples_from_original(self, client):
        """Gray picker reads from state.original_img, not result."""
        h, w = 64, 64
        img = np.ones((h, w, 3), dtype=np.float32) * 0.5
        img[30:35, 30:35, 0] = 0.3  # Red patch at center

        self._load_test_image(client, img)
        client.post("/api/params", json={"dmin": [0.6, 0.6, 0.6]})
        client.post("/api/invert", method="POST")

        # Pick on the red patch
        resp = client.post("/api/pick-gray", json={"x": 32, "y": 32})
        assert resp.status_code == 200
        # wb_high[R] should differ from 1.0 since R density differs
        wb = resp.get_json()["params"]["wb_high"]
        assert wb[0] != wb[1]  # R corrected differently from G

    def test_pick_gray_does_not_affect_dmin(self, client):
        """Picking gray should not change dmin."""
        h, w = 64, 64
        img = np.ones((h, w, 3), dtype=np.float32) * 0.5

        self._load_test_image(client, img)
        client.post("/api/params", json={"dmin": [0.5, 0.4, 0.3]})
        client.post("/api/invert", method="POST")

        dmin_before = client.get("/api/params").get_json()["dmin"]
        client.post("/api/pick-gray", json={"x": 32, "y": 32})
        dmin_after = client.get("/api/params").get_json()["dmin"]

        assert dmin_before == dmin_after
