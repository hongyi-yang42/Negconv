"""Arbitrary rotation utilities for film scan straightening.

Pure NumPy bilinear interpolation — no scipy or OpenCV.
"""
from __future__ import annotations

import numpy as np


def compute_straighten_angle(x1: float, y1: float,
                             x2: float, y2: float) -> float:
    """Compute CCW rotation angle to make the line (x1,y1)->(x2,y2) horizontal.

    Points are in screen coordinates (Y-down).  Positive result = CCW correction.
    Clamped to [-15, +15] degrees.
    """
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    return max(-15.0, min(15.0, round(angle, 2)))


def rotated_dimensions(h: int, w: int, angle_deg: float) -> tuple[int, int]:
    """Return (new_h, new_w) after rotating an (h, w) image by *angle_deg*."""
    if angle_deg == 0.0:
        return h, w
    theta = np.radians(angle_deg)
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    new_w = int(np.ceil(w * c + h * s))
    new_h = int(np.ceil(w * s + h * c))
    return new_h, new_w


def rotate_arbitrary(img: np.ndarray, angle_deg: float,
                     fill_value: float | np.ndarray = 0.0) -> np.ndarray:
    """Rotate *img* by *angle_deg* (positive = CCW) using bilinear interpolation.

    Canvas expands to fit the rotated image; out-of-bounds areas filled with
    *fill_value* (scalar or per-channel array).

    Returns a new float32 ndarray with expanded dimensions.
    """
    if angle_deg == 0.0:
        return img.copy()

    h, w = img.shape[:2]
    has_channels = img.ndim == 3
    new_h, new_w = rotated_dimensions(h, w, angle_deg)

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Build output coordinate grid (in pixel units, origin at top-left)
    ys, xs = np.mgrid[0:new_h, 0:new_w].astype(np.float64)

    # Shift to rotation centre, apply inverse rotation, shift back
    cx_src, cy_src = (w - 1) / 2.0, (h - 1) / 2.0
    cx_dst, cy_dst = (new_w - 1) / 2.0, (new_h - 1) / 2.0

    dx = xs - cx_dst
    dy = ys - cy_dst
    # Inverse rotation: rotate by -theta
    src_x = cos_t * dx + sin_t * dy + cx_src
    src_y = -sin_t * dx + cos_t * dy + cy_src

    # Bilinear interpolation
    x0 = np.floor(src_x).astype(np.intp)
    y0 = np.floor(src_y).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (src_x - x0).astype(np.float32)
    fy = (src_y - y0).astype(np.float32)

    # Bounds mask
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)

    # Clamp for safe indexing (masked later)
    x0c = np.clip(x0, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    x1c = np.clip(x1, 0, w - 1)
    y1c = np.clip(y1, 0, h - 1)

    if has_channels:
        out = np.empty((new_h, new_w, img.shape[2]), dtype=np.float32)
        fill = np.broadcast_to(np.asarray(fill_value, dtype=np.float32),
                               (img.shape[2],)).copy()
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            tl = ch[y0c, x0c].astype(np.float32)
            tr = ch[y0c, x1c].astype(np.float32)
            bl = ch[y1c, x0c].astype(np.float32)
            br = ch[y1c, x1c].astype(np.float32)
            top = tl * (1 - fx) + tr * fx
            bot = bl * (1 - fx) + br * fx
            val = top * (1 - fy) + bot * fy
            out[:, :, c] = np.where(valid, val, fill[c])
    else:
        tl = img[y0c, x0c].astype(np.float32)
        tr = img[y0c, x1c].astype(np.float32)
        bl = img[y1c, x0c].astype(np.float32)
        br = img[y1c, x1c].astype(np.float32)
        top = tl * (1 - fx) + tr * fx
        bot = bl * (1 - fx) + br * fx
        val = top * (1 - fy) + bot * fy
        out = np.where(valid, val, np.float32(fill_value))

    return np.ascontiguousarray(out)
