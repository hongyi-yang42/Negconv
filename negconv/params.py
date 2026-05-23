from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Default color film Dmin for fallback
_COLOR_DMIN = np.array([1.13, 0.49, 0.27], dtype=np.float32)

# Per-field category tag — single source of truth for carry settings.
# Adding a new param? Add one line here. Carry UI groups by unique values.
PARAM_CATEGORIES = {
    "dmin": "film_base",
    "d_max": "film_base",
    "wb_high": "wb",
    "wb_low": "wb",
    "offset": "tone",
    "exposure": "tone",
    "black": "tone",
    "gamma": "tone",
    "soft_clip": "tone",
    # GuiState fields (not in NegconvParams, but needed for carry)
    "crop_rect": "geometry",
    "orientation": "geometry",
    "flip_h": "geometry",
    "flip_v": "geometry",
}


def carry_fields_for_categories(enabled: dict) -> set[str]:
    """Return field names whose category is enabled. Input: {cat: bool}."""
    return {f for f, cat in PARAM_CATEGORIES.items() if enabled.get(cat, False)}


@dataclass
class NegconvParams:
    """All parameters for the Cineon inversion pipeline."""

    # Film base (linear light values, per-channel RGB)
    # B&W: use [1.0, 1.0, 1.0] (same value broadcast to all channels)
    dmin: np.ndarray = field(
        default_factory=lambda: _COLOR_DMIN.copy()
    )

    # Film dynamic range (scalar, optical density units)
    d_max: float = 1.6

    # White balance corrections (per-channel RGB)
    wb_high: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    wb_low: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )

    # Scanner black point offset (scalar)
    offset: float = -0.05

    # Paper simulation
    exposure: float = 0.9245
    black: float = 0.0755  # raw param, NOT black_fma
    gamma: float = 4.0
    soft_clip: float = 0.75

    @classmethod
    def color_film(cls) -> NegconvParams:
        """Default color negative (C-41) preset."""
        return cls()

    @classmethod
    def bw_film(cls) -> NegconvParams:
        """Default B&W negative preset."""
        return cls(
            dmin=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            d_max=2.2,
            gamma=5.0,
            exposure=1.0,
        )


def detect_dmin(img: np.ndarray, border_frac: float = 0.05) -> np.ndarray | None:
    """Auto-detect Dmin from the brightest border pixels.

    Film base (unexposed border) has maximum transmission = highest pixel
    values. Samples the top 5% brightest pixels from the outermost border
    region and takes the per-channel mean.

    Returns None if the detected Dmin doesn't pass sanity checks (values
    too low, or per-channel ratios don't resemble a film base).
    """
    h, w = img.shape[:2]
    bh = max(int(h * border_frac), 1)
    bw = max(int(w * border_frac), 1)

    # Collect pixels from all 4 border strips
    top = img[:bh, :, :]
    bottom = img[-bh:, :, :]
    left = img[bh:-bh, :bw, :]
    right = img[bh:-bh, -bw:, :]
    border = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3),
                             left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)

    # Camera scans have a film holder (pure black) at the edges, not film base.
    # If even the brightest channel is very dark, it's a film holder — return None
    # and let the caller fall back to preset or manual eyedropper.
    border_max = float(np.max(border))
    if border_max < 0.05:
        print("warning: border too dark (film holder?), cannot auto-detect Dmin; "
              "using preset — use GUI eyedropper for camera scans", file=sys.stderr)
        return None

    # Per-channel: take the top 5% brightest pixels and average them
    per_channel = []
    for c in range(3):
        channel = border[:, c]
        threshold = np.percentile(channel, 95)
        bright = channel[channel >= threshold]
        if len(bright) == 0:
            per_channel.append(channel.max())
        else:
            per_channel.append(np.mean(bright))

    dmin = np.array(per_channel, dtype=np.float32)

    # Sanity: Dmin must be positive and the brightest channel should be R
    # (orange mask: R > G > B for C-41). If not, no clear film border.
    if np.any(dmin < 0.0001):
        return None
    # For color film: red channel should be the brightest (orange mask)
    if dmin[0] <= dmin[1] or dmin[0] <= dmin[2]:
        # Could be B&W (R≈G≈B) — check that values aren't all identical
        # (uniform image = no film border, just cropped content)
        spread = np.max(dmin) - np.min(dmin)
        if spread < 0.001:
            return None
        return dmin

    return dmin


def detect_dmax(img: np.ndarray, dmin: np.ndarray) -> float:
    """Auto-detect Dmax from the darkest exposed area.

    Uses the 5th percentile (not minimum) to ignore sensor noise.
    Real film dynamic range: 1.5-3.0 density units.

    Args:
        img: Linear float32 image, shape (H, W, 3).
        dmin: Per-channel Dmin values.

    Returns:
        Scalar D_max in optical density units, clamped to [1.0, 4.0].
    """
    # Per-channel: use 5th percentile to skip noise floor
    dark_vals = np.array([
        np.percentile(img[:, :, c], 5) for c in range(3)
    ], dtype=np.float32)
    dark_vals = np.maximum(dark_vals, np.float32(1e-6))

    # Density = log10(Dmin / pixel) for each channel; take the max
    densities = np.log10(dmin / dark_vals)
    dmax = float(np.max(densities))

    return max(0.5, min(dmax, 4.0))


def auto_wb(image: np.ndarray, dmin: np.ndarray, d_max: float) -> np.ndarray:
    """Auto white balance via gray-world assumption in log-density space.

    Computes per-channel median log density of exposed areas, then
    equalizes using green-anchor formula (same math as WB picker).
    Returns wb_high array clamped to [0.25, 4.0].
    """
    safe_dmin = np.maximum(dmin, np.float32(1e-6))
    safe_img = np.maximum(image, np.float32(1e-10))
    log_density = -np.log10(safe_img / safe_dmin)  # positive for exposed areas

    # Mask: exclude near-base (< 0.05) and near-saturation (> 0.9 * d_max)
    ld_max = max(d_max * 0.9, 0.1)
    mask = (log_density > 0.05) & (log_density < ld_max)

    median_ld = np.ones(3, dtype=np.float32)
    for c in range(3):
        ch = log_density[:, :, c][mask[:, :, c]]
        if len(ch) < 100 or np.median(ch) < 0.01:
            return np.ones(3, dtype=np.float32)  # underexposed — skip
        median_ld[c] = float(np.median(ch))

    wb_high = np.ones(3, dtype=np.float32)
    wb_high[0] = median_ld[1] / max(median_ld[0], 1e-6)  # R
    wb_high[2] = median_ld[1] / max(median_ld[2], 1e-6)  # B
    return np.clip(wb_high, 0.25, 4.0).astype(np.float32)


def detect_dmin_percentile(image: np.ndarray) -> np.ndarray:
    """Estimate Dmin from image statistics when no border is available.

    Uses 99.5th percentile per channel (brightest pixels ≈ film base in linear space).
    """
    flat = image.reshape(-1, 3)
    return np.percentile(flat, 99.5, axis=0).astype(np.float32)


def auto_detect(img: np.ndarray, fallback_preset: str = "color",
                dmin_mode: str = "auto") -> NegconvParams:
    """Auto-detect all parameters from image content.

    Detects Dmin from border pixels and Dmax from density range.
    Falls back to percentile estimate or preset defaults if detection fails.

    Args:
        img: Linear float32 image, shape (H, W, 3).
        fallback_preset: "color" or "bw" if auto-detect fails.
        dmin_mode: "auto" (border → percentile → preset), "percentile" (skip border),
                   or "manual" (caller sets dmin via CLI overrides).

    Returns:
        NegconvParams with detected or fallback values.
    """
    params = NegconvParams.bw_film() if fallback_preset == "bw" else NegconvParams.color_film()

    try:
        dmin = None
        dmin_source = "preset"

        if dmin_mode == "percentile":
            dmin = detect_dmin_percentile(img)
            dmin_source = "percentile"
        elif dmin_mode == "auto":
            dmin = detect_dmin(img)
            if dmin is not None:
                dmin_source = "border"
            else:
                dmin = detect_dmin_percentile(img)
                dmin_source = "percentile"
                print("info: no film border detected, using percentile estimate", file=sys.stderr)

        if dmin is None or dmin_mode == "manual":
            return params

        # Sanity check: Dmin must be positive and bounded
        if np.all(dmin > 0.0001) and np.all(dmin < 2.0):
            params.dmin = dmin
        else:
            print(f"warning: auto Dmin out of range ({dmin}), using defaults", file=sys.stderr)
            return params

        dmax = detect_dmax(img, dmin)
        if 0.1 < dmax < 6.0:
            params.d_max = dmax
        else:
            print(f"warning: auto Dmax out of range ({dmax:.2f}), using default", file=sys.stderr)

        # Auto WB after Dmin/Dmax are set
        params.wb_high = auto_wb(img, params.dmin, params.d_max)

    except Exception as e:
        print(f"warning: auto-detect failed ({e}), using defaults", file=sys.stderr)

    return params


def save_params(params: NegconvParams, path: str | Path) -> None:
    """Save params to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dmin": params.dmin.tolist(),
        "d_max": params.d_max,
        "wb_high": params.wb_high.tolist(),
        "wb_low": params.wb_low.tolist(),
        "offset": params.offset,
        "exposure": params.exposure,
        "black": params.black,
        "gamma": params.gamma,
        "soft_clip": params.soft_clip,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_params(path: str | Path) -> NegconvParams:
    """Load params from JSON."""
    with open(path) as f:
        data = json.load(f)
    return NegconvParams(
        dmin=np.array(data["dmin"], dtype=np.float32),
        d_max=data["d_max"],
        wb_high=np.array(data["wb_high"], dtype=np.float32),
        wb_low=np.array(data["wb_low"], dtype=np.float32),
        offset=data["offset"],
        exposure=data["exposure"],
        black=data["black"],
        gamma=data["gamma"],
        soft_clip=data["soft_clip"],
    )
