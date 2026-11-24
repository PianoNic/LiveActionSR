"""Colour helpers.

The network sees exactly what the browser plugin computes: non-linear R'G'B' in [0, 1] (as sampled
from the decoded video texture) folded into luma with the BT.709 coefficients. HD and UHD SDR
streams are BT.709, so this is also the Y' the encoder coded.
"""

from __future__ import annotations

import numpy as np

KR, KG, KB = 0.2126, 0.7152, 0.0722
KC = np.array([KR, KG, KB], dtype=np.float32)


def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) float RGB in [0, 1] -> (...) float32 luma in [0, 1]."""
    return (rgb.astype(np.float32) @ KC).astype(np.float32)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Full-range BT.709 Y'CbCr, chroma centred on 0."""
    rgb = rgb.astype(np.float32)
    y = rgb @ KC
    cb = (rgb[..., 2] - y) / (2.0 * (1.0 - KB))
    cr = (rgb[..., 0] - y) / (2.0 * (1.0 - KR))
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    r = y + 2.0 * (1.0 - KR) * cr
    b = y + 2.0 * (1.0 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def to_uint16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 65535.0), 0, 65535).astype(np.uint16)
