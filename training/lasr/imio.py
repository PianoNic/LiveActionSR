"""PNG I/O. HR crops are stored as 16-bit RGB PNG so 10-bit sources keep their precision."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    """Any 8/16-bit PNG/JPEG/TIFF -> (H, W, 3) float32 RGB in [0, 1]."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read image {path}")
    return decode_array(img)


def decode_bytes(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError("cannot decode image bytes")
    return decode_array(img)


def decode_array(img: np.ndarray) -> np.ndarray:
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    return img[..., ::-1].astype(np.float32) / scale


def write_rgb16(path: str | Path, rgb: np.ndarray) -> None:
    arr = np.clip(np.rint(rgb * 65535.0), 0, 65535).astype(np.uint16)[..., ::-1]
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"cannot encode {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def write_rgb8(path: str | Path, rgb: np.ndarray) -> None:
    arr = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)[..., ::-1]
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise OSError(f"cannot encode {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def write_gray8(path: str | Path, y: np.ndarray) -> None:
    arr = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise OSError(f"cannot encode {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
