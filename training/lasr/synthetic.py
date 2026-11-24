"""Procedural HR clips for tests and smoke runs (no downloads). Not training data for real models:
they only exercise the pipeline end to end."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .imio import write_rgb16
from .sources import pan_clip


def synthetic_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-ish noise texture + smooth gradients + hard edges + thin lines, (size, size, 3)."""
    f = np.fft.fftfreq(size)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    r[0, 0] = 1.0
    img = np.zeros((size, size, 3), np.float32)
    for c in range(3):
        spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / r**1.1
        tex = np.real(np.fft.ifft2(spec)).astype(np.float32)
        img[..., c] = (tex - tex.mean()) / (tex.std() + 1e-6)
    img = 0.5 + 0.12 * img
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img += (0.25 * (rng.random(3) - 0.5))[None, None] * (xx[..., None] + yy[..., None])
    for _ in range(int(rng.integers(3, 8))):
        color = tuple(float(v) for v in rng.random(3))
        p1 = tuple(int(v) for v in rng.integers(0, size, 2))
        p2 = tuple(int(v) for v in rng.integers(0, size, 2))
        if rng.random() < 0.5:
            cv2.rectangle(img, p1, p2, color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.line(img, p1, p2, color, int(rng.integers(1, 4)), lineType=cv2.LINE_AA)
    return np.clip(img, 0.0, 1.0)


def make_synthetic_clips(
    out_root: str | Path, split: str, n: int, crop: int = 128, clip_len: int = 4, seed: int = 0
) -> list[Path]:
    rng = np.random.default_rng(seed)
    dirs = []
    for i in range(n):
        still = synthetic_image(crop * 2, rng)
        clip = pan_clip(still, crop, clip_len, rng)
        while clip is None:
            clip = pan_clip(still, crop, clip_len, rng)
        d = Path(out_root) / split / "synthetic" / f"p{i:05d}c0"
        for t, frame in enumerate(clip):
            write_rgb16(d / f"f{t:02d}.png", frame)
        meta = {"source": "synthetic", "split": split, "tier": "permissive", "license": "generated",
                "frames": clip_len, "size": [crop, crop]}  # fmt: skip
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        dirs.append(d)
    return dirs
