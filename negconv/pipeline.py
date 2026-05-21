from __future__ import annotations

import numpy as np

from .params import NegconvParams

THRESHOLD = np.float32(2.3283064365386963e-10)  # -32 EV


def invert(img: np.ndarray, params: NegconvParams) -> np.ndarray:
    """Run the 5-stage Cineon negative-to-positive pipeline.

    Args:
        img: Input image as float32, shape (H, W, 3), range ~[0, Dmin].
        params: Pipeline parameters.

    Returns:
        Positive image as float32, shape (H, W, 3), range ~[0, 1+].
    """
    img = img.astype(np.float32)

    # Pre-compute normalized values
    wb_high_norm = params.wb_high / np.float32(params.d_max)
    black_fma = np.float32(-params.exposure * (1.0 + params.black))
    exposure = np.float32(params.exposure)
    gamma = np.float32(params.gamma)
    soft_clip = np.float32(params.soft_clip)
    dmin = params.dmin.astype(np.float32)

    # STAGE 1: Transmission -> Log density
    clamped = np.maximum(img, THRESHOLD)
    density = dmin / clamped
    log_density = -np.log10(density)

    # STAGE 2: Correct density in log space
    corrected = wb_high_norm * log_density + np.float32(params.offset)

    # STAGE 3: Paper print simulation (FMA form)
    # Readable: print = exposure * (1 + black_param - 10^corrected)
    # FMA:      print = -(exposure * 10^corrected + black_fma)
    ten_x = np.float_power(np.float32(10.0), corrected)
    print_linear = -(exposure * ten_x + black_fma)
    print_linear = np.maximum(print_linear, np.float32(0.0))

    # STAGE 4: Paper grade (gamma / contrast)
    print_gamma = np.power(print_linear, gamma)

    # STAGE 5: Highlights soft-clip (OpenEXR formula)
    if soft_clip < 1.0:
        soft_clip_comp = np.float32(1.0) - soft_clip
        mask = print_gamma > soft_clip
        compressed = soft_clip + (
            np.float32(1.0)
            - np.exp(-(print_gamma - soft_clip) / soft_clip_comp)
        ) * soft_clip_comp
        output = np.where(mask, compressed, print_gamma)
    else:
        output = print_gamma

    return output.astype(np.float32)
