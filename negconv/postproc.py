"""Post-inversion editing applied to cached pipeline output.

All functions operate on Rec.2020 float32 images (the pipeline cache).
None of these re-run the Cineon inversion — they modify the positive result.
"""
from __future__ import annotations

import numpy as np


def apply_tint(img: np.ndarray, tint: float) -> np.ndarray:
    """Green-magenta axis correction in output space.

    Args:
        img: Pipeline output, float32 (H, W, 3).
        tint: -1.0 (green) to +1.0 (magenta). 0.0 = identity.

    Returns:
        Corrected image (same shape, float32).
    """
    if abs(tint) < 1e-6:
        return img
    result = img.copy()
    result[:, :, 0] *= np.float32(1.0 + tint * 0.25)   # R
    result[:, :, 1] *= np.float32(1.0 - tint * 0.5)     # G
    result[:, :, 2] *= np.float32(1.0 + tint * 0.25)    # B
    return result


def cubic_spline_lut(control_points: list[tuple[float, float]],
                     size: int = 256) -> np.ndarray:
    """Build a 1D LUT from control points using natural cubic spline.

    Pure NumPy — no scipy dependency.

    Args:
        control_points: List of (input, output) pairs, sorted by input.
            Must include (0,0) and (1,1) for full range.
        size: LUT entries (default 256).

    Returns:
        1D float32 array of length `size` with values in [0, 1].
    """
    if len(control_points) < 2:
        return np.linspace(0, 1, size, dtype=np.float32)

    pts = sorted(control_points, key=lambda p: p[0])
    n = len(pts) - 1
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)

    # Natural cubic spline: solve tridiagonal system for second derivatives
    h = np.diff(x)
    if np.any(h <= 0):
        # Degenerate: linear interp
        lut_x = np.linspace(0, 1, size, dtype=np.float32)
        return np.interp(lut_x, x, y).astype(np.float32)

    # Tridiagonal system for natural cubic spline (c[0] = c[n] = 0)
    a_lo = np.zeros(n + 1)
    b_di = np.zeros(n + 1)
    c_up = np.zeros(n + 1)
    d_rh = np.zeros(n + 1)

    b_di[0] = 1.0
    b_di[n] = 1.0
    for i in range(1, n):
        a_lo[i] = h[i - 1]
        b_di[i] = 2.0 * (h[i - 1] + h[i])
        c_up[i] = h[i]
        d_rh[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i - 1] - y[i]) / h[i - 1])

    # Thomas algorithm
    for i in range(1, n):
        w = a_lo[i] / b_di[i - 1]
        b_di[i] -= w * c_up[i - 1]
        d_rh[i] -= w * d_rh[i - 1]

    c2 = np.zeros(n + 1)
    for i in range(n, -1, -1):
        if i == n:
            c2[i] = d_rh[i] / b_di[i]
        else:
            c2[i] = (d_rh[i] - c_up[i] * c2[i + 1]) / b_di[i]

    # Spline coefficients per interval
    a_co = y[:-1].copy()
    b_co = np.zeros(n)
    d_co = np.zeros(n)
    for i in range(n):
        d_co[i] = (c2[i + 1] - c2[i]) / (3.0 * h[i])
        b_co[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c2[i] + c2[i + 1]) / 3.0

    # Evaluate LUT via vectorized search + eval
    lut_x = np.linspace(0.0, 1.0, size, dtype=np.float64)
    idx = np.searchsorted(x, lut_x, side='right') - 1
    idx = np.clip(idx, 0, n - 1)
    dx = lut_x - x[idx]

    lut = a_co[idx] + b_co[idx] * dx + c2[idx] * dx * dx + d_co[idx] * dx * dx * dx
    lut = np.clip(lut, 0, 1)

    # Enforce monotonicity (cubic spline can overshoot at steep transitions)
    for i in range(1, size):
        lut[i] = max(lut[i], lut[i - 1])
    return lut.astype(np.float32)


def apply_curves(img: np.ndarray,
                 composite_points: list[tuple[float, float]] | None = None,
                 r_points: list[tuple[float, float]] | None = None,
                 g_points: list[tuple[float, float]] | None = None,
                 b_points: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Apply RGB curve corrections via 1D LUTs.

    Args:
        img: Float32 (H, W, 3), range [0, 1+].
        composite_points: Control points applied to all channels.
        r_points, g_points, b_points: Per-channel control points.

    Returns:
        Corrected image (same shape).
    """
    result = img.copy()
    channels = [r_points, g_points, b_points]

    for c in range(3):
        pts = channels[c]
        if pts and len(pts) >= 2:
            lut = cubic_spline_lut(pts)
            # Map pixel values [0,1] through LUT
            idx = np.clip((result[:, :, c] * 255).astype(np.int32), 0, 255)
            result[:, :, c] = lut[idx]

    # Composite applied last (on top of per-channel)
    if composite_points and len(composite_points) >= 2:
        lut = cubic_spline_lut(composite_points)
        for c in range(3):
            idx = np.clip((result[:, :, c] * 255).astype(np.int32), 0, 255)
            result[:, :, c] = lut[idx]

    return result


# Hue sector boundaries (degrees) for 8-sector HSL
_HUE_SECTORS = [
    ("red", 345, 15),
    ("orange", 15, 45),
    ("yellow", 45, 75),
    ("green", 75, 165),
    ("cyan", 165, 195),
    ("blue", 195, 265),
    ("purple", 265, 285),
    ("magenta", 285, 345),
]


def _rgb_to_hsl(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB to HSL. All inputs/outputs in [0, 1] range, H in [0, 360]."""
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    l = (cmax + cmin) / 2.0

    # Saturation
    s = np.where(delta < 1e-10, 0.0,
                 np.where(l < 0.5, delta / (cmax + cmin + 1e-10),
                          delta / (2.0 - cmax - cmin + 1e-10)))

    # Hue
    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 1e-10)
    mask_g = (cmax == g) & (delta > 1e-10) & ~mask_r
    mask_b = (cmax == b) & (delta > 1e-10) & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    h = h % 360.0
    return h, s, l


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HSL to RGB. H in [0, 360], S and L in [0, 1]."""
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h / 60.0
    x = c * (1.0 - np.abs(hp % 2.0 - 1.0))
    m = l - c / 2.0

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    for lo, hi, rv, gv, bv in [
        (0, 1, 'c', 'x', '0'), (1, 2, 'x', 'c', '0'), (2, 3, '0', 'c', 'x'),
        (3, 4, '0', 'x', 'c'), (4, 5, 'x', '0', 'c'), (5, 6, 'c', '0', 'x'),
    ]:
        mask = (hp >= lo) & (hp < hi)
        vals = {'c': c, 'x': x, '0': np.float32(0.0)}
        r[mask] = vals[rv][mask] if isinstance(vals[rv], np.ndarray) else vals[rv]
        g[mask] = vals[gv][mask] if isinstance(vals[gv], np.ndarray) else vals[gv]
        b[mask] = vals[bv][mask] if isinstance(vals[bv], np.ndarray) else vals[bv]

    # Handle hp >= 6 or hp < 0 (shouldn't happen with %360 but safety)
    mask = (hp >= 6) | (hp < 0)
    r[mask] = c[mask]
    b[mask] = x[mask]

    return r + m, g + m, b + m


def _hue_in_sector(h: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Check which pixels have hue in [lo, hi) with wraparound for red sector."""
    if lo > hi:  # wraps around 360 (e.g. 345→15)
        return (h >= lo) | (h < hi)
    return (h >= lo) & (h < hi)


def apply_hsl(img: np.ndarray,
              hsl_adjustments: dict[str, dict[str, int]] | None = None) -> np.ndarray:
    """Apply per-sector HSL adjustments.

    Args:
        img: Float32 (H, W, 3), range [0, 1+].
        hsl_adjustments: Dict mapping sector name to {"saturation": -100..100, "luminance": -100..100}.
            Sector names: red, orange, yellow, green, cyan, blue, purple, magenta.

    Returns:
        Adjusted image.
    """
    if not hsl_adjustments:
        return img

    # Clamp to [0, 1] for HSL conversion
    clamped = np.clip(img, 0, 1)
    r, g, b = clamped[:, :, 0], clamped[:, :, 1], clamped[:, :, 2]
    h, s, l = _rgb_to_hsl(r, g, b)

    # Build per-pixel adjustment masks
    sat_adj = np.zeros_like(s)
    lum_adj = np.zeros_like(l)

    for sector_name, adj in hsl_adjustments.items():
        # Find matching sector definition
        sector_def = None
        for name, lo, hi in _HUE_SECTORS:
            if name == sector_name:
                sector_def = (lo, hi)
                break
        if sector_def is None:
            continue

        lo, hi = sector_def
        mask = _hue_in_sector(h, lo, hi)

        sat_val = adj.get("saturation", 0) / 100.0
        lum_val = adj.get("luminance", 0) / 100.0

        sat_adj += mask * sat_val
        lum_adj += mask * lum_val

    # Apply adjustments
    s_new = np.clip(s + sat_adj * s, 0, 1)
    l_new = np.clip(l + lum_adj * l, 0, 1)

    # Convert back
    r_new, g_new, b_new = _hsl_to_rgb(h, s_new, l_new)

    result = img.copy()
    result[:, :, 0] = r_new
    result[:, :, 1] = g_new
    result[:, :, 2] = b_new

    # Preserve values > 1.0 from input (highlights)
    over_mask = img > 1.0
    result = np.where(over_mask, img, result)

    return result


def apply_sharpen(img: np.ndarray, amount: float = 0.0,
                  radius: float = 1.0, threshold: float = 0.0) -> np.ndarray:
    """Unsharp mask on luminance channel only.

    Args:
        img: Float32 (H, W, 3), range [0, 1+].
        amount: Sharpening strength 0-200 (0 = off).
        radius: Gaussian-like kernel radius 0.5-5.0.
        threshold: Edge threshold 0-20 (skip flat areas).

    Returns:
        Sharpened image.
    """
    if amount <= 0:
        return img

    # Compute luminance
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # Blur via box filter (approximate Gaussian with multiple passes)
    kr = max(1, int(round(radius)))
    blurred = lum.copy()
    for _ in range(3):  # 3-pass box ≈ Gaussian
        # Horizontal pass
        pad = np.pad(blurred, ((0, 0), (kr, kr)), mode='edge')
        cs = pad.cumsum(axis=1)
        blurred = (cs[:, 2 * kr:] - cs[:, :-2 * kr]) / (2 * kr)
        # Vertical pass
        pad = np.pad(blurred, ((kr, kr), (0, 0)), mode='edge')
        cs = pad.cumsum(axis=0)
        blurred = (cs[2 * kr:, :] - cs[:-2 * kr, :]) / (2 * kr)

    # Edge mask: only sharpen where local contrast exceeds threshold
    diff = np.abs(lum - blurred)
    edge_mask = np.where(diff > threshold / 255.0, 1.0, 0.0)

    # Unsharp mask: sharpened = original + amount * (original - blurred)
    sharpened_lum = lum + (amount / 100.0) * (lum - blurred) * edge_mask

    # Apply luminance change to all channels proportionally
    safe_lum = np.maximum(lum, 1e-10)
    scale = sharpened_lum / safe_lum

    result = img.copy()
    for c in range(3):
        result[:, :, c] = img[:, :, c] * scale

    return result


def apply_post_edits(
    img: np.ndarray,
    tint: float = 0.0,
    curves: dict | None = None,
    hsl: dict | None = None,
    sharpen: dict | None = None,
) -> np.ndarray:
    """Apply full post-edit chain to cached pipeline output.

    Order: tint → curves → HSL → sharpen.
    All operate on Rec.2020 float32.
    """
    result = img

    if abs(tint) > 1e-6:
        result = apply_tint(result, tint)

    if curves and any(curves.values()):
        result = apply_curves(
            result,
            composite_points=curves.get("composite"),
            r_points=curves.get("r"),
            g_points=curves.get("g"),
            b_points=curves.get("b"),
        )

    if hsl and any(v for v in hsl.values() if v):
        result = apply_hsl(result, hsl)

    if sharpen and sharpen.get("amount", 0) > 0:
        result = apply_sharpen(
            result,
            amount=sharpen.get("amount", 0),
            radius=sharpen.get("radius", 1.0),
            threshold=sharpen.get("threshold", 0),
        )

    return result
