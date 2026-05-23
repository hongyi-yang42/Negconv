"""Color space conversions for the Negconv pipeline.

All matrices operate on linear light (no gamma).
Computed from IEC 61966-2-1 (sRGB) and ITU-R BT.2020 chromaticity coordinates, D65 white point.
"""
from __future__ import annotations

import numpy as np

# sRGB primaries: R(0.64,0.33) G(0.30,0.60) B(0.15,0.06), D65 white(0.3127,0.3290)
# Rec.2020 primaries: R(0.708,0.292) G(0.170,0.797) B(0.131,0.046), D65

MATRIX_SRGB_TO_REC2020 = np.array([
    [0.6274, 0.3293, 0.0433],
    [0.0691, 0.9195, 0.0114],
    [0.0164, 0.0880, 0.8956],
], dtype=np.float32)

MATRIX_REC2020_TO_SRGB = np.linalg.inv(MATRIX_SRGB_TO_REC2020).astype(np.float32)


def convert_color_space(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color matrix to an HxWx3 float32 image (vectorized)."""
    return np.einsum('ij,...j->...i', matrix, image)


def srgb_to_rec2020(image: np.ndarray) -> np.ndarray:
    return convert_color_space(image, MATRIX_SRGB_TO_REC2020)


def rec2020_to_srgb(image: np.ndarray) -> np.ndarray:
    return convert_color_space(image, MATRIX_REC2020_TO_SRGB)
