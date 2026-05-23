"""Sprint 8 tests: WB upgrade, LUT loader, highlight recovery."""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from negconv.gui.app import create_app
from negconv.lut import parse_cube, apply_lut


class TestCubeParse:
    def test_parse_1d_identity(self, tmp_path):
        """Parse a 1D identity LUT, verify shape."""
        n = 16
        lines = [f"LUT_1D_SIZE {n}"]
        for i in range(n):
            v = i / (n - 1)
            lines.append(f"{v:.6f} {v:.6f} {v:.6f}")
        cube_file = tmp_path / "identity1d.cube"
        cube_file.write_text("\n".join(lines))

        lut = parse_cube(str(cube_file))
        assert lut["type"] == "1D"
        assert lut["size"] == n
        assert lut["table"].shape == (n, 3)
        assert np.isclose(lut["table"][0, 0], 0.0)
        assert np.isclose(lut["table"][-1, 0], 1.0)

    def test_parse_3d_identity(self, tmp_path):
        """Parse a 3D identity LUT, verify shape."""
        n = 4
        lines = [f"LUT_3D_SIZE {n}"]
        for r in range(n):
            for g in range(n):
                for b in range(n):
                    rv, gv, bv = r / (n - 1), g / (n - 1), b / (n - 1)
                    lines.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")
        cube_file = tmp_path / "identity3d.cube"
        cube_file.write_text("\n".join(lines))

        lut = parse_cube(str(cube_file))
        assert lut["type"] == "3D"
        assert lut["size"] == n
        assert lut["table"].shape == (n ** 3, 3)

    def test_identity_lut_passthrough(self, tmp_path):
        """Apply 3D identity LUT — output ≈ input."""
        n = 8
        lines = [f"LUT_3D_SIZE {n}"]
        for r in range(n):
            for g in range(n):
                for b in range(n):
                    rv, gv, bv = r / (n - 1), g / (n - 1), b / (n - 1)
                    lines.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")
        cube_file = tmp_path / "identity.cube"
        cube_file.write_text("\n".join(lines))

        lut = parse_cube(str(cube_file))
        img = np.random.rand(32, 32, 3).astype(np.float32) * 0.9
        result = apply_lut(img, lut)
        assert np.allclose(result, img, atol=0.02)
