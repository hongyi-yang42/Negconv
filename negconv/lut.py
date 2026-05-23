"""Parse and apply .cube LUT files (1D and 3D) with trilinear interpolation."""

from __future__ import annotations

import numpy as np
from pathlib import Path


def parse_cube(filepath: str | Path) -> dict:
    """Parse a .cube LUT file.

    Returns dict with keys: title, type ("1D" or "3D"), size,
    domain_min (np.ndarray), domain_max (np.ndarray),
    table (np.ndarray shape (N, 3) for 1D or (N*N*N, 3) for 3D).
    """
    filepath = Path(filepath)
    title = ""
    lut_type = "3D"
    size = None
    domain_min = np.array([0.0, 0.0, 0.0])
    domain_max = np.array([1.0, 1.0, 1.0])
    table_lines = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            keyword = parts[0].upper()

            if keyword == "TITLE":
                title = " ".join(parts[1:]).strip('"')
            elif keyword == "LUT_3D_SIZE":
                lut_type = "3D"
                size = int(parts[1])
            elif keyword == "LUT_1D_SIZE":
                lut_type = "1D"
                size = int(parts[1])
            elif keyword == "DOMAIN_MIN":
                domain_min = np.array([float(v) for v in parts[1:4]])
            elif keyword == "DOMAIN_MAX":
                domain_max = np.array([float(v) for v in parts[1:4]])
            else:
                # Data line: R G B
                try:
                    table_lines.append([float(v) for v in parts[:3]])
                except ValueError:
                    continue

    if size is None:
        raise ValueError("No LUT_1D_SIZE or LUT_3D_SIZE found in .cube file")

    table = np.array(table_lines, dtype=np.float32)
    expected = size if lut_type == "1D" else size ** 3
    if len(table) != expected:
        raise ValueError(f"Expected {expected} entries, got {len(table)}")

    return {
        "title": title,
        "type": lut_type,
        "size": size,
        "domain_min": domain_min.astype(np.float32),
        "domain_max": domain_max.astype(np.float32),
        "table": table,
    }


def apply_lut_3d(
    img: np.ndarray,
    table: np.ndarray,
    size: int,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> np.ndarray:
    """Apply a 3D LUT to an image using trilinear interpolation.

    Input image is clamped to [domain_min, domain_max], then mapped
    into the LUT grid. Output has the same shape as input.
    """
    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float32)

    # Normalize to [0, size-1] grid coordinates
    d_range = domain_max - domain_min
    d_range = np.where(d_range > 0, d_range, 1.0)
    norm = (flat - domain_min) / d_range * (size - 1)
    norm = np.clip(norm, 0, size - 1)

    # Integer and fractional parts
    idx = np.floor(norm).astype(np.int32)
    frac = norm - idx.astype(np.float32)

    # Clamp indices for safe neighbor access
    idx_p = np.minimum(idx + 1, size - 1)

    # Trilinear interpolation
    # 8 corners of the cube
    def _lookup(r, g, b):
        return table[r * size * size + g * size + b]

    c000 = _lookup(idx[:, 0], idx[:, 1], idx[:, 2])
    c001 = _lookup(idx[:, 0], idx[:, 1], idx_p[:, 2])
    c010 = _lookup(idx[:, 0], idx_p[:, 1], idx[:, 2])
    c011 = _lookup(idx[:, 0], idx_p[:, 1], idx_p[:, 2])
    c100 = _lookup(idx_p[:, 0], idx[:, 1], idx[:, 2])
    c101 = _lookup(idx_p[:, 0], idx[:, 1], idx_p[:, 2])
    c110 = _lookup(idx_p[:, 0], idx_p[:, 1], idx[:, 2])
    c111 = _lookup(idx_p[:, 0], idx_p[:, 1], idx_p[:, 2])

    fx, fy, fz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

    result = (
        c000 * (1 - fx) * (1 - fy) * (1 - fz)
        + c001 * (1 - fx) * (1 - fy) * fz
        + c010 * (1 - fx) * fy * (1 - fz)
        + c011 * (1 - fx) * fy * fz
        + c100 * fx * (1 - fy) * (1 - fz)
        + c101 * fx * (1 - fy) * fz
        + c110 * fx * fy * (1 - fz)
        + c111 * fx * fy * fz
    )

    return result.reshape(h, w, 3).astype(np.float32)


def apply_lut_1d(
    img: np.ndarray,
    table: np.ndarray,
    size: int,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> np.ndarray:
    """Apply a 1D LUT per channel using linear interpolation."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float32)

    d_range = domain_max - domain_min
    d_range = np.where(d_range > 0, d_range, 1.0)
    norm = (flat - domain_min) / d_range * (size - 1)
    norm = np.clip(norm, 0, size - 1)

    idx = np.floor(norm).astype(np.int32)
    frac = norm - idx.astype(np.float32)
    idx_p = np.minimum(idx + 1, size - 1)

    # Per-channel lookup
    result = np.empty_like(flat)
    for c in range(3):
        lo = table[idx[:, c], c]
        hi = table[idx_p[:, c], c]
        result[:, c] = lo + (hi - lo) * frac[:, c]

    return result.reshape(h, w, 3).astype(np.float32)


def apply_lut(img: np.ndarray, lut_data: dict) -> np.ndarray:
    """Apply a parsed LUT to an image."""
    if lut_data["type"] == "3D":
        return apply_lut_3d(
            img, lut_data["table"], lut_data["size"],
            lut_data["domain_min"], lut_data["domain_max"],
        )
    return apply_lut_1d(
        img, lut_data["table"], lut_data["size"],
        lut_data["domain_min"], lut_data["domain_max"],
    )
