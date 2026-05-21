from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import tifffile

RAW_EXTENSIONS = {".cr2", ".cr3", ".arw", ".raf", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw", ".sr2", ".k25", ".kc"}


def _read_icc_gamma(icc: bytes) -> float | None:
    """Extract gamma from ICC profile TRC. Returns None if linear or not found."""
    for tag_name in [b"rTRC", b"gTRC", b"bTRC"]:
        idx = icc.find(tag_name)
        if idx < 0:
            continue
        tag_offset = struct.unpack(">I", icc[idx + 4 : idx + 8])[0]
        tag_size = struct.unpack(">I", icc[idx + 8 : idx + 12])[0]
        trc_data = icc[tag_offset : tag_offset + tag_size]
        trc_type = trc_data[:4]
        if trc_type != b"curv":
            continue
        count = struct.unpack(">I", trc_data[8:12])[0]
        if count == 0:
            return None  # linear
        if count == 1:
            val = struct.unpack(">H", trc_data[12:14])[0]
            return val / 256.0
        return None  # LUT-based, skip
    return None


def read_raw(path: str | Path) -> np.ndarray:
    """Read a RAW file and return linear float32 array shaped (H, W, 3).

    Uses rawpy with: output_bps=16, gamma=(1,1), no_auto_bright=True,
    use_camera_wb=True, ColorSpace.sRGB. RAW data is already linear —
    no degamma needed.
    """
    import rawpy
    import sys

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
        )
    img = rgb.astype(np.float32) / 65535.0

    # Heuristic: if the image is nearly black, re-process with auto-brightness.
    # Some camera/scanner combinations produce very dark output without it.
    if img.mean() < 0.08:
        print("warning: low signal detected in RAW, enabling auto-brightness", file=sys.stderr)
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=False,
                use_camera_wb=True,
            )
        img = rgb.astype(np.float32) / 65535.0

    return img


def read_tiff(path: str | Path, linearize: bool = True) -> np.ndarray:
    """Read a TIFF image and return linearized float32 array shaped (H, W, 3).

    Handles uint8, uint16, and float32 input. Integer types are normalized to
    [0, 1]. If linearize=True, applies inverse gamma from the embedded ICC
    profile (defaults to gamma 2.2 for scanner TIFFs with no profile).
    """
    img = tifffile.imread(str(path))

    # Detect gamma from ICC profile before normalizing
    gamma = None
    if linearize:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            icc_tag = page.tags.get("InterColorProfile")
            if icc_tag is not None:
                gamma = _read_icc_gamma(icc_tag.value)
            elif img.dtype in (np.uint8, np.uint16):
                # Scanner TIFF without ICC — assume sRGB gamma ~2.2
                gamma = 2.2

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    # Linearize
    if gamma is not None and gamma != 1.0:
        img = np.power(np.clip(img, 0.0, 1.0), np.float32(gamma))

    # Ensure (H, W, 3) — handle grayscale or alpha channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.broadcast_to(img, (*img.shape[:2], 3)).copy()

    return img


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file (RAW or TIFF) and return linear float32 (H, W, 3).

    Detects file type by extension. RAW files use rawpy (already linear).
    TIFF files are linearized via ICC gamma if present.
    """
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        return read_raw(path)
    return read_tiff(path)


def is_raw(path: str | Path) -> bool:
    """Check if a file is a RAW format based on extension."""
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def write_image(path: str | Path, img: np.ndarray, dtype: str = "float32") -> None:
    """Write a float32 image to TIFF.

    Args:
        path: Output file path.
        img: float32 array shaped (H, W, 3), range ~[0, 1+].
        dtype: 'float32' for 32-bit float output, 'uint16' for 16-bit integer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dtype == "uint16":
        out = np.clip(img, 0.0, 1.0)
        out = (out * 65535.0).astype(np.uint16)
    else:
        out = img.astype(np.float32)

    tifffile.imwrite(str(path), out, photometric="rgb")
