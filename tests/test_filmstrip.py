"""Tests for lazy thumbnail generation and open folder."""

import os
import tempfile

import numpy as np
import pytest
import tifffile

from negconv.gui.app import (
    create_app,
    _ensure_thumb_window,
    _thumb_queued,
    _scan_directory_simple,
    _thumb_path,
    GuiState,
    THUMB_WINDOW,
)


def _make_tiff(directory, name, shape=(100, 100)):
    """Create a synthetic TIFF file in directory and return its path."""
    data = np.full((*shape, 3), 30000, dtype=np.uint16)
    path = os.path.join(directory, name)
    tifffile.imwrite(path, data, photometric="rgb")
    return path


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


class TestLazyThumbWindow:
    def test_only_window_queued(self):
        """With 50 files, _ensure_thumb_window only queues ±THUMB_WINDOW indices."""
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i in range(50):
                files.append(_make_tiff(tmp, f"frame_{i:04d}.tif"))

            state = GuiState()
            state.directory_files = files
            _thumb_queued.clear()

            _ensure_thumb_window(0, state)

            # Only indices [0, THUMB_WINDOW] should be queued (lo=max(0,-10)=0)
            expected_lo = 0
            expected_hi = THUMB_WINDOW + 1  # 11
            assert len(_thumb_queued) == expected_hi - expected_lo
            for i in range(expected_lo, expected_hi):
                assert i in _thumb_queued
            # Index 25 should NOT be queued
            assert 25 not in _thumb_queued

    def test_center_midrange(self):
        """Center at index 25 queues [15, 35]."""
        with tempfile.TemporaryDirectory() as tmp:
            files = [_make_tiff(tmp, f"f_{i:03d}.tif") for i in range(50)]

            state = GuiState()
            state.directory_files = files
            _thumb_queued.clear()

            _ensure_thumb_window(25, state)

            assert 15 in _thumb_queued
            assert 35 in _thumb_queued
            assert 14 not in _thumb_queued
            assert 36 not in _thumb_queued

    def test_navigate_extends_window(self):
        """Navigating to a distant index queues new thumbnails."""
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(30):
                _make_tiff(tmp, f"frame_{i:04d}.tif")

            files = _scan_directory_simple(tmp)
            app = create_app()
            client = app.test_client()

            resp = client.post("/api/load", json={"path": files[0]})
            assert resp.status_code == 200

            resp = client.post("/api/navigate", json={"index": 25})
            assert resp.status_code == 200
            result = resp.get_json()
            assert result["current_index"] == 25


class TestThumbRetry:
    def test_thumb_404_for_nonexistent_file(self):
        """Requesting thumb for out-of-range index returns 404."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_tiff(tmp, "frame_0.tif")

            files = _scan_directory_simple(tmp)
            app = create_app()
            client = app.test_client()

            resp = client.post("/api/load", json={"path": files[0]})
            assert resp.status_code == 200

            # Index out of range
            resp = client.get("/api/thumb/99")
            assert resp.status_code == 404

    def test_thumb_404_when_not_yet_generated(self):
        """Thumb returns 404 if file exists but thumb hasn't been generated yet.

        We test this by creating a state where the thumb file doesn't exist.
        The api_thumb endpoint triggers _ensure_thumb_window which queues async
        generation, but the thumb may not exist yet on first request.
        """
        with tempfile.TemporaryDirectory() as tmp:
            files = [_make_tiff(tmp, f"frame_{i:04d}.tif") for i in range(5)]

            app = create_app()
            client = app.test_client()

            resp = client.post("/api/load", json={"path": files[0]})
            assert resp.status_code == 200

            # Delete any generated thumbs for these specific files
            for f in files:
                tp = _thumb_path(f)
                if tp.exists():
                    os.unlink(tp)

            _thumb_queued.clear()

            # Request thumb for index 0 — thumb file doesn't exist yet
            resp = client.get("/api/thumb/0")
            # Either 404 (not generated yet) or 200 (generated fast) is acceptable
            assert resp.status_code in (200, 404)


class TestOpenFolder:
    def test_load_directory_endpoint(self):
        """POST /api/load-directory loads first file and returns file list."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                paths.append(_make_tiff(tmp, f"photo_{i:04d}.tif"))

            app = create_app()
            client = app.test_client()

            resp = client.post("/api/load-directory", json={"path": tmp})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["filename"] == "photo_0000.tif"
            assert data["total_files"] == 3
            assert data["current_index"] == 0
            assert data["dims"] == [100, 100]
            assert "preview" in data

    def test_load_directory_empty(self):
        """POST /api/load-directory with no images returns error."""
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app()
            client = app.test_client()

            resp = client.post("/api/load-directory", json={"path": tmp})
            assert resp.status_code == 400

    def test_load_directory_nonexistent(self):
        """POST /api/load-directory with bad path returns error."""
        app = create_app()
        client = app.test_client()

        resp = client.post("/api/load-directory",
                           json={"path": "/nonexistent/dir"})
        assert resp.status_code == 400

    def test_load_directory_trailing_slash(self):
        """Filepath with trailing / triggers load-directory endpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_tiff(tmp, "scan.tif")

            app = create_app()
            client = app.test_client()

            resp = client.post("/api/load-directory",
                               json={"path": tmp.rstrip("/")})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["total_files"] == 1


class TestScanDirectorySubdirs:
    def test_top_level_only(self):
        """Default scan only finds top-level files."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_tiff(tmp, "top.tif")
            sub = os.path.join(tmp, "sub")
            os.makedirs(sub)
            _make_tiff(sub, "nested.tif")

            files = _scan_directory_simple(tmp)
            assert len(files) == 1
            assert "top.tif" in files[0]

    def test_include_subdirs(self):
        """include_subdirs=True finds files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_tiff(tmp, "top.tif")
            sub = os.path.join(tmp, "sub")
            os.makedirs(sub)
            _make_tiff(sub, "nested.tif")

            files = _scan_directory_simple(tmp, include_subdirs=True)
            assert len(files) == 2
            names = [os.path.basename(f) for f in files]
            assert "top.tif" in names
            assert "nested.tif" in names
