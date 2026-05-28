"""Sprint 11 tests: arbitrary rotation (straighten + fine slider)."""
import numpy as np
import pytest

from negconv.geometry import (
    compute_straighten_angle,
    rotated_dimensions,
    rotate_arbitrary,
)


class TestStraightenAngle:
    def test_horizontal_line_gives_zero(self):
        angle = compute_straighten_angle(0, 0, 100, 0)
        assert abs(angle) < 0.001

    def test_straighten_computes_correct_angle(self):
        # Line tilted 5° below horizontal (dy positive in Y-down screen coords)
        dx, dy = 100.0, 100.0 * np.tan(np.radians(5))
        angle = compute_straighten_angle(0, 0, dx, dy)
        expected = np.degrees(np.arctan2(dy, dx))
        assert abs(angle - expected) < 0.01

    def test_negative_tilt(self):
        dx, dy = 100.0, -100.0 * np.tan(np.radians(3))
        angle = compute_straighten_angle(0, 0, dx, dy)
        assert angle < 0
        expected = np.degrees(np.arctan2(dy, dx))
        assert abs(angle - expected) < 0.01

    def test_clamp_to_15(self):
        # Nearly vertical line — angle should clamp
        angle = compute_straighten_angle(0, 0, 10, 1000)
        assert -15.0 <= angle <= 15.0

    def test_identical_points(self):
        angle = compute_straighten_angle(50, 50, 50, 50)
        assert abs(angle) < 0.001


class TestRotatedDimensions:
    def test_zero_angle(self):
        assert rotated_dimensions(100, 200, 0.0) == (100, 200)

    def test_arbitrary_rotation_dimensions(self):
        h, w = 100, 200
        new_h, new_w = rotated_dimensions(h, w, 3.0)
        assert new_w > w
        assert new_h > h
        theta = np.radians(3.0)
        exp_w = int(np.ceil(w * abs(np.cos(theta)) + h * abs(np.sin(theta))))
        exp_h = int(np.ceil(w * abs(np.sin(theta)) + h * abs(np.cos(theta))))
        assert new_w == exp_w
        assert new_h == exp_h

    def test_90_degrees(self):
        # At exactly 90°, sin/cos give exact values but ceil may round up
        new_h, new_w = rotated_dimensions(100, 200, 90.0)
        assert abs(new_w - 100) <= 1
        assert abs(new_h - 200) <= 1


class TestRotateArbitrary:
    def test_zero_returns_same_image(self):
        img = np.random.rand(50, 80, 3).astype(np.float32)
        result = rotate_arbitrary(img, 0.0)
        np.testing.assert_array_equal(result, img)

    def test_output_dtype_is_float32(self):
        img = np.zeros((30, 40, 3), dtype=np.float32)
        result = rotate_arbitrary(img, 5.0)
        assert result.dtype == np.float32

    def test_rotation_fill_value_scalar(self):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        result = rotate_arbitrary(img, 10.0, fill_value=0.5)
        # Corner should be fill value
        assert abs(result[0, 0, 0] - 0.5) < 0.05

    def test_rotation_fill_value_array(self):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        fill = np.array([1.0, 0.5, 0.2], dtype=np.float32)
        result = rotate_arbitrary(img, 10.0, fill_value=fill)
        corner = result[0, 0, :]
        np.testing.assert_allclose(corner, fill, atol=0.05)

    def test_center_preserved(self):
        # A white center pixel should stay approximately white after small rotation
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[48:52, 48:52, :] = 1.0
        result = rotate_arbitrary(img, 2.0)
        cy, cx = result.shape[0] // 2, result.shape[1] // 2
        assert result[cy, cx, 0] > 0.9

    def test_grayscale_image(self):
        img = np.random.rand(50, 80).astype(np.float32)
        result = rotate_arbitrary(img, 5.0)
        assert result.ndim == 2
        assert result.dtype == np.float32

    def test_dimensions_match_rotated_dimensions(self):
        img = np.random.rand(60, 80, 3).astype(np.float32)
        result = rotate_arbitrary(img, 7.0)
        exp_h, exp_w = rotated_dimensions(60, 80, 7.0)
        assert result.shape[0] == exp_h
        assert result.shape[1] == exp_w


class TestRotationSidecar:
    def test_rotation_persisted_in_sidecar(self, tmp_path):
        import json
        from negconv.gui.app import _auto_save, _load_sidecar, _apply_sidecar, GuiState

        state = GuiState()
        state.angle_deg = 3.5
        state.file_path = str(tmp_path / "test.tif")

        # Write a minimal TIF so sidecar parent exists
        import tifffile
        tifffile.imwrite(state.file_path, np.zeros((10, 10, 3), dtype=np.uint16))

        _auto_save(state)
        loaded = _load_sidecar(state.file_path)
        assert loaded is not None
        assert abs(loaded.get("angle_deg", 0.0) - 3.5) < 0.001

        # Apply to new state
        state2 = GuiState()
        _apply_sidecar(state2, loaded)
        assert abs(state2.angle_deg - 3.5) < 0.001

    def test_angle_deg_default_zero(self, tmp_path):
        from negconv.gui.app import _auto_save, _load_sidecar, _apply_sidecar, GuiState

        state = GuiState()
        state.file_path = str(tmp_path / "test.tif")
        import tifffile
        tifffile.imwrite(state.file_path, np.zeros((10, 10, 3), dtype=np.uint16))

        _auto_save(state)
        loaded = _load_sidecar(state.file_path)
        # angle_deg == 0 should NOT be in the sidecar (conditional save)
        assert "angle_deg" not in loaded

        state2 = GuiState()
        _apply_sidecar(state2, loaded)
        assert state2.angle_deg == 0.0
