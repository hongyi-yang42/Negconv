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


def recover_highlights(img: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """Reconstruct clipped channels from unclipped neighbors.

    For pixels with 1-2 channels clipped (>= threshold), estimates the
    clipped value using local 5x5 mean of fully-unclipped neighbor pixels
    to preserve channel ratios. All-3-clipped pixels are left unchanged.
    """
    h, w, _ = img.shape
    clipped = img >= threshold
    n_clipped = clipped.sum(axis=2)

    needs_fix = (n_clipped >= 1) & (n_clipped <= 2)
    if not np.any(needs_fix):
        return img

    result = img.copy()
    pad = 2

    # Use cumulative sums for fast 5x5 box average over fully-unclipped pixels
    good_mask = (n_clipped == 0).astype(np.float32)  # (H, W)
    # Weighted sum: pixel values * good_mask for each channel, plus count
    padded_g = np.pad(good_mask, ((pad, pad), (pad, pad)), mode="constant")
    padded_v = np.pad(img * good_mask[:, :, np.newaxis],
                      ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    # Cumulative sums on padded array
    cs_v = padded_v.cumsum(axis=0).cumsum(axis=1)
    cs_g = padded_g.cumsum(axis=0).cumsum(axis=1)

    def _box_sum(cs, y0, y1, x0, x1):
        return cs[y1, x1] - cs[y0, x1] - cs[y1, x0] + cs[y0, x0]

    fix_y, fix_x = np.where(needs_fix)
    for i in range(len(fix_y)):
        fy, fx = fix_y[i], fix_x[i]
        # Box in padded coords (centered on pixel)
        r0, r1 = fy + pad - 2, fy + pad + 3
        c0, c1 = fx + pad - 2, fx + pad + 3

        sum_v = _box_sum(cs_v, r0, r1, c0, c1)  # (3,)
        count = _box_sum(cs_g, r0, r1, c0, c1)

        if count < 1.0:
            continue

        local_mean = sum_v / count
        if local_mean.max() < 1e-6:
            continue

        pixel = img[fy, fx].copy()
        unclipped = ~clipped[fy, fx]

        # Scale local mean so unclipped channels match the pixel's actual values
        local_unclipped_mean = local_mean[unclipped].mean()
        if local_unclipped_mean < 1e-6:
            continue
        scale = pixel[unclipped].mean() / local_unclipped_mean

        for c in range(3):
            if clipped[fy, fx, c]:
                pixel[c] = local_mean[c] * scale

        result[fy, fx] = pixel

    return result
