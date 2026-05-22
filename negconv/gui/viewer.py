from __future__ import annotations

import io

import numpy as np
from PIL import Image


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma to a linear float32 image in [0, 1].

    Returns float32 in [0, 1] with sRGB gamma applied.
    """
    out = np.where(
        img > 0.0031308,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055,
        12.92 * img,
    )
    return out.astype(np.float32)


def make_preview(img: np.ndarray, max_width: int = 1200, quality: int = 90) -> bytes:
    """Convert a linear float32 image to a JPEG preview.

    Clips to [0, 1], applies sRGB gamma, resizes to fit max_width,
    and encodes as JPEG.

    Returns JPEG bytes.
    """
    clipped = np.clip(img, 0.0, 1.0)
    srgb = linear_to_srgb(clipped)
    uint8 = (srgb * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(uint8, mode="RGB")

    if pil_img.width > max_width:
        ratio = max_width / pil_img.width
        new_size = (max_width, int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
