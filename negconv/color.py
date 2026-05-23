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
    Fully vectorized via cumulative-sum box filter.
    """
    h, w, _ = img.shape
    clipped = img >= threshold
    n_clipped = clipped.sum(axis=2)

    needs_fix = (n_clipped >= 1) & (n_clipped <= 2)
    if not np.any(needs_fix):
        return img

    result = img.copy()
    pad = 2

    # Mask of fully-unclipped pixels (all 3 channels below threshold)
    good_mask = (n_clipped == 0).astype(np.float32)  # (H, W)

    # Pad and compute cumulative sums for 5x5 box average
    padded_g = np.pad(good_mask, ((pad, pad), (pad, pad)), mode="constant")
    padded_v = np.pad(img * good_mask[:, :, np.newaxis],
                      ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    cs_v = padded_v.cumsum(axis=0).cumsum(axis=1)  # (H+4, W+4, 3)
    cs_g = padded_g.cumsum(axis=0).cumsum(axis=1)  # (H+4, W+4)

    # Compute box sums for all pixels at once (vectorized)
    # Box: [y+pad-2, y+pad+3) x [x+pad-2, x+pad+3) in padded coords
    r1 = cs_v[4:, 4:]        # bottom-right corner (y+pad+3-1, x+pad+3-1)
    r0 = cs_v[:-4, 4:]       # top edge
    c0 = cs_v[4:, :-4]       # left edge
    rc00 = cs_v[:-4, :-4]    # top-left corner
    box_v = r1 - r0 - c0 + rc00  # (H, W, 3)

    g1 = cs_g[4:, 4:]
    g0 = cs_g[:-4, 4:]
    gc0 = cs_g[4:, :-4]
    gc00 = cs_g[:-4, :-4]
    box_g = g1 - g0 - gc0 + gc00  # (H, W)

    # Local mean of good neighbors: (H, W, 3)
    safe_count = np.maximum(box_g, 1.0)
    local_mean = box_v / safe_count[:, :, np.newaxis]

    # Per-pixel unclipped channel mask and scale factor
    # For each pixel needing fix, scale local_mean so unclipped channels
    # match actual pixel values
    unclipped = ~clipped  # (H, W, 3)
    # Sum of unclipped channels: actual pixel and local_mean
    pixel_unclipped_sum = (img * unclipped).sum(axis=2)       # (H, W)
    local_unclipped_sum = (local_mean * unclipped).sum(axis=2)  # (H, W)
    n_unclipped = unclipped.sum(axis=2).astype(np.float32)      # (H, W)

    # Scale = (pixel unclipped mean) / (local unclipped mean)
    safe_local = np.maximum(local_unclipped_sum / np.maximum(n_unclipped, 1), 1e-6)
    pixel_mean = pixel_unclipped_sum / np.maximum(n_unclipped, 1)
    scale = pixel_mean / safe_local  # (H, W)

    # Estimated values: local_mean * scale, only for clipped channels
    estimated = local_mean * scale[:, :, np.newaxis]

    # Apply: replace clipped channels with estimate, keep unclipped as-is
    fix_mask = needs_fix & (box_g >= 1.0) & (pixel_mean > 1e-6)  # (H, W)
    # Only overwrite clipped channels where fix is valid
    apply_mask = fix_mask[:, :, np.newaxis] & clipped  # (H, W, 3)
    result = np.where(apply_mask, estimated, result)

    return result
