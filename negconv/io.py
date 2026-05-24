from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

RAW_EXTENSIONS = {
    # Canon
    ".cr2", ".cr3", ".crw",
    # Sony
    ".arw", ".sr2", ".srf",
    # Nikon
    ".nef", ".nrw",
    # Fuji
    ".raf",
    # Adobe
    ".dng",
    # Olympus/OM
    ".orf", ".ori",
    # Panasonic/Lumix
    ".raw", ".rw2",
    # Pentax
    ".pef", ".ptx",
    # Samsung
    ".srw",
    # Kodak
    ".dcr", ".k25", ".kc",
    # Hasselblad
    ".3fr", ".fff",
    # Minolta
    ".mrw",
    # Epson
    ".erf",
    # Mamiya
    ".mef",
    # Sigma
    ".x3f",
}


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

    Uses rawpy with unity WB (user_wb=[1,1,1,1]) and Rec.2020 color space.
    rawpy applies the camera-specific color matrix internally, producing
    linear Rec.2020 output. Stage 1 (divide by Dmin) handles channel
    normalization.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1, 1, 1, 1],
            output_color=rawpy.ColorSpace.Rec2020,
        )
    return rgb.astype(np.float32) / 65535.0


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


def read_image(path: str | Path) -> tuple[np.ndarray, bool]:
    """Read an image file (RAW or TIFF) and return (linear float32, is_raw).

    RAW files return linear Rec.2020 (rawpy handles camera matrix).
    TIFF files return linear sRGB (scanner color space).
    The is_raw flag tells the caller whether to apply sRGB→Rec.2020 conversion.
    """
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        return read_raw(path), True
    return read_tiff(path), False


def is_raw(path: str | Path) -> bool:
    """Check if a file is a RAW format based on extension."""
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def extract_exif(path: str | Path) -> bytes | None:
    """Extract EXIF data from an image file.

    For RAW: tries rawpy first. For TIFF: uses Pillow.
    Strips thumbnail and orientation tag to avoid issues.
    Returns raw EXIF bytes or None.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # Try rawpy for RAW files
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                exif = raw.extract_rawpy_exif()
                if exif:
                    return exif
        except Exception:
            pass

    # Try Pillow for TIFF and as fallback for RAW
    try:
        pil = Image.open(str(path))
        exif = pil.getexif()
        if exif:
            # Strip orientation and thumbnail
            exif.pop(0x0112, None)  # Orientation
            exif.pop(0x0201, None)  # Thumbnail offset
            exif.pop(0x0202, None)  # Thumbnail length
            return exif.tobytes()
    except Exception:
        pass

    return None


# np.rot90 rotates CCW by k*90°
# Our orientation: 1=90°CW, 2=180°, 3=270°CW
# So orientation 1 → k=3 (270° CCW = 90° CW)
_ROT90_K = {1: 3, 2: 2, 3: 1}


def apply_orientation(image: np.ndarray, orientation: int,
                      flip_h: bool, flip_v: bool) -> np.ndarray:
    """Apply rotation then flip. Flip is relative to displayed orientation."""
    if orientation in _ROT90_K:
        image = np.rot90(image, k=_ROT90_K[orientation])
    if flip_h:
        image = np.flip(image, axis=1)
    if flip_v:
        image = np.flip(image, axis=0)
    return np.ascontiguousarray(image)


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


def _srgb_lut() -> np.ndarray:
    """65536-entry LUT mapping linear [0,1] to sRGB [0,1]."""
    lut = np.linspace(0, 1, 65536, dtype=np.float32)
    mask = lut <= 0.0031308
    lut[mask] = 12.92 * lut[mask]
    lut[~mask] = 1.055 * np.power(lut[~mask], 1.0 / 2.4) - 0.055
    return np.clip(lut, 0.0, 1.0)


_SRGB_LUT = _srgb_lut()


def _linear_to_srgb_uint8(img: np.ndarray) -> np.ndarray:
    """Convert linear float32 (H,W,3) to sRGB uint8."""
    clamped = np.clip(img, 0.0, 1.0)
    idx = (clamped * 65535).astype(np.int32)
    idx = np.clip(idx, 0, 65535)
    srgb = _SRGB_LUT[idx]
    return (srgb * 255.0 + 0.5).astype(np.uint8)


_SRGB_ICC = None


def _get_srgb_icc() -> bytes:
    """Return sRGB ICC profile bytes (cached)."""
    global _SRGB_ICC
    if _SRGB_ICC is None:
        from PIL import ImageCms
        _SRGB_ICC = ImageCms.ImageCmsProfile(
            ImageCms.createProfile("sRGB")
        ).tobytes()
    return _SRGB_ICC


def write_jpeg(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    """Write a float32 linear image as JPEG with sRGB gamma and ICC profile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _linear_to_srgb_uint8(img)
    pil = Image.fromarray(rgb, "RGB")
    pil.save(str(path), "JPEG", quality=quality, icc_profile=_get_srgb_icc())


def write_heic(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    """Write a float32 linear image as HEIC with sRGB gamma and ICC profile."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        raise RuntimeError(
            "pillow-heif not installed. Install with: pip install pillow-heif"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _linear_to_srgb_uint8(img)
    pil = Image.fromarray(rgb, "RGB")
    pil.save(str(path), "HEIF", quality=quality, icc_profile=_get_srgb_icc())
