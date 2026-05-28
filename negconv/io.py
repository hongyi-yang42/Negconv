from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
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

    Uses daylight WB multipliers to spread weak channels across more of the
    [0,1] range, improving quantization in density space. This gives the
    color matrix correctly balanced input, producing more accurate Rec.2020
    output. Fallback: daylight_whitebalance → camera_whitebalance → unity.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        wb = [1, 1, 1, 1]
        try:
            dwb = list(raw.daylight_whitebalance)
            if dwb and any(v > 0.01 for v in dwb[:3]):
                wb = dwb
        except AttributeError:
            pass
        if wb == [1, 1, 1, 1]:
            try:
                cwb = list(raw.camera_whitebalance)
                if cwb and any(v > 0.01 for v in cwb[:3]):
                    wb = cwb
            except AttributeError:
                pass

        rgb = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=wb,
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


def write_image(path: str | Path, img: np.ndarray, dtype: str = "float32",
                apply_srgb_gamma: bool = False) -> None:
    """Write a float32 image to TIFF.

    Args:
        path: Output file path.
        img: float32 array shaped (H, W, 3), range ~[0, 1+].
        dtype: 'float32' for 32-bit float output, 'uint16' for 16-bit integer.
        apply_srgb_gamma: If True, apply sRGB gamma curve before quantizing
                          to uint16. Use when the input is linear and the output
                          should look correct in sRGB viewers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dtype == "uint16":
        clipped = np.clip(img, 0.0, 1.0)
        if apply_srgb_gamma:
            clipped = _SRGB_LUT[(clipped * 65535).astype(np.int32).clip(0, 65535)]
        out = (clipped * 65535.0 + 0.5).astype(np.uint16)
        tifffile.imwrite(str(path), out, photometric="rgb",
                         compression="deflate", iccprofile=_get_srgb_icc())
    else:
        out = img.astype(np.float32)
        tifffile.imwrite(str(path), out, photometric="rgb",
                         compression="deflate")


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


def write_xmp_sidecar(
    output_path: str | Path,
    params_dict: dict,
    source_exif: bytes | None = None,
    source_filename: str = "",
) -> str:
    """Write an XMP sidecar file alongside *output_path*.

    Returns the path to the written XMP file.
    """
    from . import __version__

    output_path = Path(output_path)
    xmp_path = output_path.with_suffix(output_path.suffix + ".xmp")

    # Root element
    root = ET.Element("x:xmpmeta", {"xmlns:x": "adobe:ns:meta/"})

    # RDF wrapper
    rdf = ET.SubElement(root, "rdf:RDF", {
        "xmlns:rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    })
    desc = ET.SubElement(rdf, "rdf:Description", {
        "rdf:about": "",
        "xmlns:xmp": "http://ns.adobe.com/xap/1.0/",
        "xmlns:dc": "http://purl.org/dc/elements/1.1/",
        "xmlns:negconv": "http://negconv.org/pipeline/1.0/",
    })

    # Creator tool
    desc.set("xmp:CreatorTool", f"Negconv v{__version__}")

    # Source filename
    if source_filename:
        dc_source = ET.SubElement(desc, "dc:source")
        dc_source.text = source_filename

    # EXIF fields from source
    if source_exif:
        try:
            from PIL.ExifTags import Base as ExifBase
            src = Image.Exif()
            src.load(source_exif)
            tag_map = {
                0x0110: "xmp:Model",        # Model
                0x010f: "xmp:Manufacturer",  # Make
                0x829a: "xmp:ExposureTime",  # ExposureTime
                0x8827: "xmp:ISOSpeedRatings",  # ISOSpeedRatings
                0x920a: "xmp:FocalLength",   # FocalLength
            }
            for tag, xmp_name in tag_map.items():
                val = src.get(tag)
                if val is not None:
                    desc.set(xmp_name, str(val))
            # DateTime
            dt = src.get(0x0132)
            if dt:
                desc.set("xmp:CreateDate", str(dt))
        except Exception:
            pass

    # Negconv pipeline params
    for key in ("dmin", "d_max", "wb_high", "wb_low", "offset", "exposure",
                "black", "gamma", "soft_clip", "tint", "tone_profile",
                "angle_deg", "orientation"):
        val = params_dict.get(key)
        if val is not None:
            if isinstance(val, (list, np.ndarray)):
                val = "[" + ",".join(str(v) for v in val) + "]"
            else:
                val = str(val)
            desc.set(f"negconv:{key}", val)

    # Write with XML declaration
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(xmp_path), xml_declaration=True, encoding="UTF-8")
    return str(xmp_path)


def resize_for_export(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so longest edge = *max_dim* (downscale only). Returns float32."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0
