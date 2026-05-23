"""Sprint 7 tests: profiles, WB eyedropper, undo/redo, black level, selective carry."""

import json
import os
import tempfile

import numpy as np
import pytest
import tifffile

from negconv.gui.app import create_app
from negconv.params import NegconvParams
from negconv.profiles import (
    PROFILE_DIR, save_profile, load_profile, list_profiles, delete_profile,
    ensure_profile_dir,
)


# ---- B.5: Profile + WB tests ----


class TestProfiles:
    def test_save_load_profile_roundtrip(self, tmp_path, monkeypatch):
        """Saving and loading a profile preserves all params."""
        monkeypatch.setattr("negconv.profiles.PROFILE_DIR", tmp_path / "profiles")
        params = NegconvParams.color_film()
        params.gamma = 5.5
        params.exposure = 1.1
        params.dmin = np.array([0.5, 0.3, 0.1], dtype=np.float32)

        save_profile("test_stock", params)
        loaded = load_profile("test_stock")

        assert abs(loaded.gamma - 5.5) < 0.01
        assert abs(loaded.exposure - 1.1) < 0.01
        np.testing.assert_allclose(loaded.dmin, [0.5, 0.3, 0.1], atol=0.001)
        # Other fields match defaults
        assert abs(loaded.black - params.black) < 0.01
        assert abs(loaded.soft_clip - params.soft_clip) < 0.01

    def test_profile_directory_created(self, tmp_path, monkeypatch):
        """ensure_profile_dir creates the profile directory."""
        d = tmp_path / "new_profiles"
        monkeypatch.setattr("negconv.profiles.PROFILE_DIR", d)
        ensure_profile_dir()
        assert d.is_dir()

    def test_profile_list_endpoint(self, tmp_path, monkeypatch):
        """GET /api/profiles returns saved profiles."""
        d = tmp_path / "profiles"
        monkeypatch.setattr("negconv.profiles.PROFILE_DIR", d)
        save_profile("Portra 400", NegconvParams.color_film())
        save_profile("HP5", NegconvParams.bw_film())

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/api/profiles")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [p["name"] for p in data]
        assert "Portra 400" in names
        assert "HP5" in names


class TestWBEyedropper:
    def _make_tiff_with_color_cast(self):
        """Create a TIFF with a color cast that WB can correct."""
        f = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        # Simulate a negative with blue cast (blue channel brighter)
        data = np.zeros((200, 200, 3), dtype=np.uint16)
        data[:, :, 0] = 20000   # R
        data[:, :, 1] = 25000   # G
        data[:, :, 2] = 40000   # B (strong blue cast)
        # Add border for Dmin detection
        border = np.array([60000, 30000, 15000], dtype=np.uint16)
        data[:10, :] = border
        data[-10:, :] = border
        data[:, :10] = border
        data[:, -10:] = border
        tifffile.imwrite(f.name, data, photometric="rgb")
        f.close()
        return f.name

    def test_wb_eyedropper_neutral_patch(self):
        """POST /api/pick-wb adjusts wb_high to neutralize a color cast."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        path = self._make_tiff_with_color_cast()

        try:
            client.post("/api/load", json={"path": path})
            client.post("/api/invert")

            # Pick a center point for WB
            resp = client.post("/api/pick-wb", json={"x": 100, "y": 100})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "wb_high" in data
            assert len(data["wb_high"]) == 3
            # WB should have changed from [1,1,1]
            wb = data["wb_high"]
            assert not all(abs(v - 1.0) < 0.01 for v in wb), \
                "WB should have changed from neutral"
            # Preview should be available
            assert "preview" in data
        finally:
            os.unlink(path)
            sidecar = path + ".negconv.json"
            if os.path.isfile(sidecar):
                os.unlink(sidecar)

    def test_wb_eyedropper_does_not_affect_dmin(self):
        """WB eyedropper modifies wb_high but leaves dmin unchanged."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        path = self._make_tiff_with_color_cast()

        try:
            client.post("/api/load", json={"path": path})
            client.post("/api/invert")

            # Record Dmin before WB
            resp_before = client.get("/api/params")
            dmin_before = resp_before.get_json()["dmin"]

            # Pick WB
            resp = client.post("/api/pick-wb", json={"x": 100, "y": 100})
            assert resp.status_code == 200

            # Dmin should be unchanged
            resp_after = client.get("/api/params")
            dmin_after = resp_after.get_json()["dmin"]
            np.testing.assert_allclose(dmin_before, dmin_after, atol=0.001)
        finally:
            os.unlink(path)
            sidecar = path + ".negconv.json"
            if os.path.isfile(sidecar):
                os.unlink(sidecar)


# ---- C.3: Undo/Redo + Black level tests ----


class TestUndoRedo:
    def _make_tiff(self):
        f = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        data = np.full((100, 100, 3), 30000, dtype=np.uint16)
        tifffile.imwrite(f.name, data, photometric="rgb")
        f.close()
        return f.name

    def test_undo_restores_previous_params(self):
        """POST /api/undo restores params to previous state."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        path = self._make_tiff()

        try:
            client.post("/api/load", json={"path": path})
            # Set gamma to a known value
            client.post("/api/params", json={"gamma": 6.0})
            # Change it again
            client.post("/api/params", json={"gamma": 3.0})

            # Undo should restore gamma=6.0
            resp = client.post("/api/undo")
            assert resp.status_code == 200
            data = resp.get_json()
            assert abs(data["params"]["gamma"] - 6.0) < 0.01
        finally:
            os.unlink(path)
            sidecar = path + ".negconv.json"
            if os.path.isfile(sidecar):
                os.unlink(sidecar)

    def test_redo_after_undo(self):
        """POST /api/redo after undo restores the undone state."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        path = self._make_tiff()

        try:
            client.post("/api/load", json={"path": path})
            client.post("/api/params", json={"gamma": 6.0})
            client.post("/api/params", json={"gamma": 3.0})

            # Undo → gamma=6.0
            client.post("/api/undo")
            # Redo → gamma=3.0
            resp = client.post("/api/redo")
            assert resp.status_code == 200
            data = resp.get_json()
            assert abs(data["params"]["gamma"] - 3.0) < 0.01
        finally:
            os.unlink(path)
            sidecar = path + ".negconv.json"
            if os.path.isfile(sidecar):
                os.unlink(sidecar)

    def test_undo_stack_max_depth(self):
        """Undo stack caps at 50 entries; oldest entries are discarded."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        path = self._make_tiff()

        try:
            client.post("/api/load", json={"path": path})
            # Push 55 distinct gamma values (load pushes 1, params push 55 = 56 total, capped to 50)
            for i in range(55):
                client.post("/api/params", json={"gamma": float(i)})

            # Can undo up to 49 times (stack capped at 50, index starts at 49)
            undo_count = 0
            for _ in range(60):
                resp = client.post("/api/undo")
                if resp.status_code != 200:
                    break
                undo_count += 1

            # Should be able to undo 49 times (from index 49 to index 0)
            assert undo_count == 49
        finally:
            os.unlink(path)
            sidecar = path + ".negconv.json"
            if os.path.isfile(sidecar):
                os.unlink(sidecar)


class TestBlackLevel:
    def test_black_level_per_channel_displayed(self):
        """GET /api/info returns black_level_per_channel for TIFF (zeros)."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        f = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        data = np.full((50, 50, 3), 30000, dtype=np.uint16)
        tifffile.imwrite(f.name, data, photometric="rgb")
        f.close()

        try:
            client.post("/api/load", json={"path": f.name})
            resp = client.get("/api/info")
            assert resp.status_code == 200
            info = resp.get_json()
            assert "black_level" in info
            assert len(info["black_level"]) == 4
        finally:
            os.unlink(f.name)
            sidecar = f.name + ".negconv.json"
            if os.path.isfile(sidecar):
                os.unlink(sidecar)


class TestSelectiveCarry:
    def _make_tiff(self, tmp_path, name, value=30000):
        path = str(tmp_path / name)
        data = np.full((100, 100, 3), value, dtype=np.uint16)
        tifffile.imwrite(path, data, photometric="rgb")
        return path

    def test_copy_settings_selective_carry(self, tmp_path):
        """POST /api/copy-settings with only tone carries gamma but not WB."""
        p1 = self._make_tiff(tmp_path, "aaa.tif", value=30000)
        p2 = self._make_tiff(tmp_path, "bbb.tif", value=10000)

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        try:
            client.post("/api/load", json={"path": p1})
            # Set custom gamma and WB
            client.post("/api/params", json={"gamma": 6.0, "wb_high": [1.5, 1.0, 0.8]})

            # Copy only tone to file 2 (no WB, no geometry)
            resp = client.post("/api/copy-settings", json={
                "target_index": 1,
                "categories": {"tone": True, "wb": False, "film_base": False, "geometry": False},
            })
            assert resp.status_code == 200
            data = resp.get_json()

            # Gamma carried
            assert abs(data["params"]["gamma"] - 6.0) < 0.01
            # WB NOT carried — should be defaults
            wb = data["params"]["wb_high"]
            assert all(abs(v - 1.0) < 0.01 for v in wb), f"WB should be neutral, got {wb}"
        finally:
            for p in (p1, p2):
                s = p + ".negconv.json"
                if os.path.isfile(s):
                    os.unlink(s)

    def test_carry_categories_auto_discover(self):
        """CARRY_CATEGORIES covers all NegconvParams fields."""
        from negconv.params import CARRY_CATEGORIES, NegconvParams
        import dataclasses

        all_carry_fields = set()
        for fields in CARRY_CATEGORIES.values():
            all_carry_fields.update(fields)

        for f in dataclasses.fields(NegconvParams):
            assert f.name in all_carry_fields, f"Field '{f.name}' not in any CARRY_CATEGORIES"
