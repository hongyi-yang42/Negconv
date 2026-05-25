from __future__ import annotations

import json
import sys
from collections.abc import Callable
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
    "tint": "tone",
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

    # Post-inversion tint (green-magenta axis, -1 to +1)
    tint: float = 0.0

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


@dataclass
class RollProfile:
    """Aggregated statistics for a roll/directory of film negatives."""

    roll_dmin: np.ndarray          # per-channel median Dmin
    roll_wb_high: np.ndarray       # per-channel median WB
    roll_exposure_offset: float    # suggested exposure adjustment
    num_frames: int                # number of frames analyzed
    outlier_indices: list[int]     # frames where Dmin deviates >2σ
    per_frame_dmin: list[list[float]] = field(default_factory=list)
    per_frame_wb: list[list[float]] = field(default_factory=list)
    per_frame_exposure: list[float] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "roll_dmin": self.roll_dmin.tolist(),
            "roll_wb_high": self.roll_wb_high.tolist(),
            "roll_exposure_offset": self.roll_exposure_offset,
            "num_frames": self.num_frames,
            "outlier_indices": self.outlier_indices,
            "per_frame_dmin": self.per_frame_dmin,
            "per_frame_wb": self.per_frame_wb,
            "per_frame_exposure": self.per_frame_exposure,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> RollProfile:
        with open(path) as f:
            data = json.load(f)
        return cls(
            roll_dmin=np.array(data["roll_dmin"], dtype=np.float32),
            roll_wb_high=np.array(data["roll_wb_high"], dtype=np.float32),
            roll_exposure_offset=data["roll_exposure_offset"],
            num_frames=data["num_frames"],
            outlier_indices=data["outlier_indices"],
            per_frame_dmin=data.get("per_frame_dmin", []),
            per_frame_wb=data.get("per_frame_wb", []),
            per_frame_exposure=data.get("per_frame_exposure", []),
        )


def analyze_roll(
    images: list[np.ndarray],
    progress_callback: Callable[[int, int], None] | None = None,
) -> RollProfile:
    """Analyze a roll of film negatives and compute aggregate statistics.

    Iterates all images, computes per-frame Dmin (border detect → percentile
    fallback), auto_wb, and exposure stats. Aggregates into a RollProfile
    with median-based roll baseline and >2σ outlier detection.

    Args:
        images: List of linear float32 images, shape (H, W, 3).
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        RollProfile with aggregated roll statistics.
    """
    from statistics import median

    per_dmin: list[np.ndarray] = []
    per_wb: list[np.ndarray] = []
    per_exposure: list[float] = []

    for i, img in enumerate(images):
        if progress_callback:
            progress_callback(i, len(images))

        # Dmin: border detect → percentile fallback
        dmin = detect_dmin(img)
        if dmin is None:
            dmin = detect_dmin_percentile(img)

        dmax = detect_dmax(img, dmin)
        wb = auto_wb(img, dmin, dmax)

        per_dmin.append(dmin)
        per_wb.append(wb)

        # Exposure proxy: mean log-density of exposed areas
        safe_dmin = np.maximum(dmin, np.float32(1e-6))
        safe_img = np.maximum(img, np.float32(1e-10))
        ld = -np.log10(safe_img / safe_dmin)
        mask = (ld > 0.05) & (ld < dmax * 0.9)
        if mask.any():
            per_exposure.append(float(np.median(ld[mask])))
        else:
            per_exposure.append(0.0)

    if progress_callback:
        progress_callback(len(images), len(images))

    # Aggregate: per-channel median across frames
    dmin_stack = np.stack(per_dmin)
    wb_stack = np.stack(per_wb)
    roll_dmin = np.median(dmin_stack, axis=0).astype(np.float32)
    roll_wb = np.median(wb_stack, axis=0).astype(np.float32)
    roll_exp_offset = float(median(per_exposure)) - median(per_exposure)  # zero-centered

    # Outlier detection: >2σ from roll median Dmin (using L2 norm)
    dmin_dist = np.linalg.norm(dmin_stack - roll_dmin, axis=1)
    dmin_mean = float(np.mean(dmin_dist))
    dmin_std = float(np.std(dmin_dist))
    outliers = []
    if dmin_std > 1e-6:
        outliers = [i for i, d in enumerate(dmin_dist) if d > 2 * dmin_std]

    return RollProfile(
        roll_dmin=roll_dmin,
        roll_wb_high=roll_wb,
        roll_exposure_offset=roll_exp_offset,
        num_frames=len(images),
        outlier_indices=outliers,
        per_frame_dmin=[d.tolist() for d in per_dmin],
        per_frame_wb=[w.tolist() for w in per_wb],
        per_frame_exposure=per_exposure,
    )


def detect_border_region(img: np.ndarray, border_px: int = 0) -> dict[str, int]:
    """Detect film frame boundary using gradient-based edge detection.

    Finds the content rect (actual film frame) excluding sprocket holes,
    holder edges, and unexposed border. Used for Dmin/WB sampling only —
    does NOT affect export dimensions.

    Args:
        img: Linear float32 image, shape (H, W, 3).
        border_px: Override: use this many pixels as border. 0 = auto-detect.

    Returns:
        dict with x, y, w, h of the content region.
    """
    h, w = img.shape[:2]

    if border_px > 0:
        return {"x": border_px, "y": border_px,
                "w": w - 2 * border_px, "h": h - 2 * border_px}

    # Compute luminance
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # Gradient along each axis (Sobel-like: central difference)
    grad_y = np.zeros_like(lum)
    grad_x = np.zeros_like(lum)
    if h > 2:
        grad_y[1:-1, :] = np.abs(lum[2:, :] - lum[:-2, :])
    if w > 2:
        grad_x[:, 1:-1] = np.abs(lum[:, 2:] - lum[:, :-2])

    # Average gradient per row/col
    row_grad = np.mean(grad_x, axis=1)
    col_grad = np.mean(grad_y, axis=0)

    # Find edges: first significant gradient peak from each side
    def _find_edge(profile: np.ndarray, threshold_factor: float = 0.3) -> int:
        if len(profile) < 4:
            return 0
        threshold = threshold_factor * np.max(profile)
        above = np.where(profile > threshold)[0]
        return int(above[0]) if len(above) > 0 else 0

    top = _find_edge(row_grad)
    bottom = h - _find_edge(row_grad[::-1])
    left = _find_edge(col_grad)
    right = w - _find_edge(col_grad[::-1])

    # Clamp: content rect must be at least 10px
    if right - left < 10:
        left, right = 0, w
    if bottom - top < 10:
        top, bottom = 0, h

    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


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
        "tint": params.tint,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_params(path: str | Path) -> NegconvParams:
    """Load params from JSON. Reads both CLI params.json and GUI sidecar formats."""
    with open(path) as f:
        data = json.load(f)
    p = data.get("params", data)  # GUI sidecar wraps params; CLI doesn't
    return NegconvParams(
        dmin=np.array(p["dmin"], dtype=np.float32),
        d_max=p["d_max"],
        wb_high=np.array(p["wb_high"], dtype=np.float32),
        wb_low=np.array(p["wb_low"], dtype=np.float32),
        offset=p["offset"],
        exposure=p["exposure"],
        black=p["black"],
        gamma=p["gamma"],
        soft_clip=p["soft_clip"],
        tint=p.get("tint", 0.0),
    )
