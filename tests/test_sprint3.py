"""Sprint 3 tests: CLI flags, JSON sidecar, batch mode."""

import json
import os
import tempfile

import numpy as np
import pytest
import tifffile

from negconv.params import NegconvParams, load_params, save_params


class TestJSONSidecar:
    def test_round_trip(self):
        """Save → load produces identical params."""
        original = NegconvParams.color_film()
        original.dmin = np.array([0.85, 0.30, 0.15], dtype=np.float32)
        original.d_max = 2.37
        original.gamma = 3.5

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_params(original, path)
            loaded = load_params(path)
            np.testing.assert_allclose(loaded.dmin, original.dmin)
            assert loaded.d_max == original.d_max
            assert loaded.gamma == original.gamma
            assert loaded.exposure == original.exposure
            assert loaded.black == original.black
            assert loaded.soft_clip == original.soft_clip
            assert loaded.offset == original.offset
        finally:
            os.unlink(path)

    def test_json_format(self):
        """JSON has expected keys and array format."""
        params = NegconvParams.bw_film()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_params(params, path)
            with open(path) as f:
                data = json.load(f)
            assert set(data.keys()) == {
                "dmin", "d_max", "wb_high", "wb_low",
                "offset", "exposure", "black", "gamma", "soft_clip",
            }
            assert len(data["dmin"]) == 3
            assert isinstance(data["d_max"], float)
        finally:
            os.unlink(path)

    def test_overrides_after_load(self):
        """CLI flags override loaded params."""
        import argparse
        from negconv.main import _apply_cli_overrides

        params = NegconvParams.color_film()
        args = argparse.Namespace(
            dmin_r=0.5, dmin_g=0.3, dmin_b=0.1,
            dmax=2.0,
            wb_high_r=None, wb_high_g=None, wb_high_b=None,
            wb_low_r=None, wb_low_g=None, wb_low_b=None,
            exposure=1.0, black=None, gamma=5.0, soft_clip=None, offset=None,
        )
        _apply_cli_overrides(params, args)
        np.testing.assert_allclose(params.dmin, [0.5, 0.3, 0.1])
        assert params.d_max == 2.0
        assert params.exposure == 1.0
        assert params.gamma == 5.0

    def test_partial_overrides(self):
        """Setting only one dmin channel preserves the others."""
        import argparse
        from negconv.main import _apply_cli_overrides

        params = NegconvParams.color_film()
        original_g = params.dmin[1]
        args = argparse.Namespace(
            dmin_r=0.5, dmin_g=None, dmin_b=None,
            dmax=None,
            wb_high_r=None, wb_high_g=None, wb_high_b=None,
            wb_low_r=None, wb_low_g=None, wb_low_b=None,
            exposure=None, black=None, gamma=None, soft_clip=None, offset=None,
        )
        _apply_cli_overrides(params, args)
        assert params.dmin[0] == 0.5
        assert params.dmin[1] == original_g


class TestCLIFlags:
    def test_all_paper_params(self):
        """All paper simulation params can be overridden."""
        import argparse
        from negconv.main import _apply_cli_overrides

        params = NegconvParams.color_film()
        args = argparse.Namespace(
            dmin_r=None, dmin_g=None, dmin_b=None,
            dmax=None,
            wb_high_r=None, wb_high_g=None, wb_high_b=None,
            wb_low_r=None, wb_low_g=None, wb_low_b=None,
            exposure=1.5, black=0.1, gamma=6.0, soft_clip=0.9, offset=-0.1,
        )
        _apply_cli_overrides(params, args)
        assert params.exposure == 1.5
        assert params.black == 0.1
        assert params.gamma == 6.0
        assert params.soft_clip == 0.9
        assert params.offset == -0.1

    def test_wb_overrides(self):
        """White balance overrides work."""
        import argparse
        from negconv.main import _apply_cli_overrides

        params = NegconvParams.color_film()
        args = argparse.Namespace(
            dmin_r=None, dmin_g=None, dmin_b=None,
            dmax=None,
            wb_high_r=1.2, wb_high_g=1.0, wb_high_b=0.8,
            wb_low_r=0.9, wb_low_g=1.0, wb_low_b=1.1,
            exposure=None, black=None, gamma=None, soft_clip=None, offset=None,
        )
        _apply_cli_overrides(params, args)
        np.testing.assert_allclose(params.wb_high, [1.2, 1.0, 0.8])
        np.testing.assert_allclose(params.wb_low, [0.9, 1.0, 1.1])

    def test_no_overrides_preserves(self):
        """All-None args don't change params."""
        import argparse
        from negconv.main import _apply_cli_overrides

        params = NegconvParams.color_film()
        orig_dmin = params.dmin.copy()
        args = argparse.Namespace(
            dmin_r=None, dmin_g=None, dmin_b=None,
            dmax=None,
            wb_high_r=None, wb_high_g=None, wb_high_b=None,
            wb_low_r=None, wb_low_g=None, wb_low_b=None,
            exposure=None, black=None, gamma=None, soft_clip=None, offset=None,
        )
        _apply_cli_overrides(params, args)
        np.testing.assert_array_equal(params.dmin, orig_dmin)
        assert params.d_max == 1.6


class TestBatchMode:
    def test_batch_processes_all(self):
        """Batch mode processes all TIFFs in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "in")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(in_dir)

            # Create 3 small test TIFFs
            for i in range(3):
                data = np.full((32, 32, 3), 30000, dtype=np.uint16)
                tifffile.imwrite(os.path.join(in_dir, f"img_{i}.tif"), data)

            # Run batch
            import subprocess
            result = subprocess.run(
                ["python", "-m", "negconv", in_dir, "-o", out_dir],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert "processing 3 files" in result.stdout

            # Check outputs exist
            for i in range(3):
                out_path = os.path.join(out_dir, f"img_{i}_negconv.tif")
                assert os.path.exists(out_path), f"Missing output: {out_path}"

            # Check params.json was auto-saved
            assert os.path.exists(os.path.join(out_dir, "params.json"))

    def test_batch_params_json_valid(self):
        """Auto-saved params.json is valid and loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "in")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(in_dir)

            data = np.full((32, 32, 3), 30000, dtype=np.uint16)
            tifffile.imwrite(os.path.join(in_dir, "test.tif"), data)

            import subprocess
            subprocess.run(
                ["python", "-m", "negconv", in_dir, "-o", out_dir],
                capture_output=True,
            )

            params = load_params(os.path.join(out_dir, "params.json"))
            assert isinstance(params, NegconvParams)
            assert params.gamma > 0

    def test_batch_with_override(self):
        """Batch mode with --gamma override applies to all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "in")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(in_dir)

            for i in range(2):
                data = np.full((32, 32, 3), 30000, dtype=np.uint16)
                tifffile.imwrite(os.path.join(in_dir, f"img_{i}.tif"), data)

            import subprocess
            result = subprocess.run(
                ["python", "-m", "negconv", in_dir, "-o", out_dir, "--gamma", "3.0"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0

            params = load_params(os.path.join(out_dir, "params.json"))
            assert params.gamma == 3.0


class TestInputCollection:
    def test_single_file(self):
        from negconv.main import _collect_inputs
        files = _collect_inputs("tests/fixtures/Example.tif")
        assert len(files) == 1
        assert "Example.tif" in files[0]

    def test_directory_globs_tiffs(self):
        from negconv.main import _collect_inputs
        files = _collect_inputs("tests/fixtures")
        tifs = [f for f in files if f.endswith(".tif")]
        assert len(tifs) >= 1  # at least Example.tif
