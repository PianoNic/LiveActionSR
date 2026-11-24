"""Streaming-style degradation: HR clip -> (LR luma, HR luma).

The recipe follows what a VOD pipeline does to a master before a viewer sees it (and the shape of
the LiveActionV1 recipe described in docs/research.md):

1. optional mild blur (lens / mastering softness),
2. optional extra down-and-up rescale pass (ladder / mastering scaling blur),
3. the 2x downscale with a random kernel (bicubic, lanczos, area, bilinear, aliased bilinear),
4. optional unsharp mask (sharpening halos some encoders' pre-filters and TVs add),
5. optional noise / grain, added before encoding so the codec damages it the way it damages real
   grain,
6. a real codec round-trip through ffmpeg (H.264 / HEVC / AV1, random CRF and preset, 4:2:0,
   BT.709), decoded back to 8-bit RGB like a browser texture.

LR = BT.709 luma of the decoded RGB, HR = luma of the original. Every step operates on whole
clips, so P/B-frame damage (not just intra-frame damage) is in the training data.
"""

from __future__ import annotations

import copy
from typing import Any

import cv2
import numpy as np
from PIL import Image

from . import ffmpeg
from .color import rgb_to_y, to_uint8

DEFAULT_DEGRADATION: dict[str, Any] = {
    "blur": {"p": 0.3, "sigma": [0.3, 1.1]},
    "rescale_pass": {"p": 0.2, "scale": [0.55, 0.9]},
    "downscale_kernels": {
        "bicubic": 0.35,
        "lanczos": 0.25,
        "area": 0.2,
        "bilinear": 0.1,
        "bilinear_aliased": 0.1,
    },
    "sharpen": {"p": 0.15, "amount": [0.3, 1.0], "sigma": [0.6, 1.5]},
    "noise": {"p": 0.25, "sigma": [1.0, 6.0], "luma_only_p": 0.7, "correlated_p": 0.5},
    "codec": {
        "p": 0.95,
        "weights": {"x264": 0.45, "x265": 0.3, "av1": 0.25},
        "crf": {"x264": [18, 32], "x265": [20, 34], "av1": [24, 46]},
        "presets": {
            "x264": ["veryfast", "faster", "fast", "medium", "slow"],
            "x265": ["superfast", "veryfast", "faster", "fast", "medium"],
            "av1": ["6", "7", "8", "9", "10"],
        },
        "keyint": [12, 96],
        "bframes": [0, 3],
    },
}

_PIL_KERNELS = {
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
}


# Weight tables are replaced as a whole (listing only x265 means "only x265"), everything else merges.
REPLACE_KEYS = {"weights", "downscale_kernels"}


def merge_config(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS:
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resize(img: np.ndarray, size: tuple[int, int], kernel: str) -> np.ndarray:
    """Resize an (H, W) or (H, W, C) float image to size=(W, H). All kernels except
    'bilinear_aliased' are antialiased (the kernel widens on downscale, like swscale/zimg)."""
    w, h = size
    if kernel == "area":
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if kernel == "bilinear_aliased":
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if kernel == "lanczos4_cv2":
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    resample = _PIL_KERNELS[kernel]
    if img.ndim == 2:
        return np.asarray(Image.fromarray(img.astype(np.float32), "F").resize((w, h), resample))
    chans = [
        np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c], np.float32), "F").resize((w, h), resample))
        for c in range(img.shape[-1])
    ]
    return np.stack(chans, axis=-1)


def _uniform(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = [k for k, v in weights.items() if v > 0]
    p = np.array([weights[k] for k in keys], dtype=np.float64)
    return str(keys[int(rng.choice(len(keys), p=p / p.sum()))])


def sample_params(cfg: dict, rng: np.random.Generator, codecs: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Draw one degradation. `codecs` limits the codec choice (defaults to what ffmpeg has)."""
    cfg = merge_config(DEFAULT_DEGRADATION, cfg)
    p: dict[str, Any] = {}
    p["blur_sigma"] = _uniform(rng, cfg["blur"]["sigma"]) if rng.random() < cfg["blur"]["p"] else 0.0
    rp = cfg["rescale_pass"]
    p["rescale"] = _uniform(rng, rp["scale"]) if rng.random() < rp["p"] else 0.0
    p["rescale_kernels"] = [
        _choice(rng, cfg["downscale_kernels"]),
        _choice(rng, {k: v for k, v in cfg["downscale_kernels"].items() if k != "area"}),
    ]
    p["kernel"] = _choice(rng, cfg["downscale_kernels"])
    sh = cfg["sharpen"]
    if rng.random() < sh["p"]:
        p["sharpen"] = {"amount": _uniform(rng, sh["amount"]), "sigma": _uniform(rng, sh["sigma"])}
    else:
        p["sharpen"] = None
    nz = cfg["noise"]
    if rng.random() < nz["p"]:
        p["noise"] = {
            "sigma": _uniform(rng, nz["sigma"]) / 255.0,
            "luma_only": bool(rng.random() < nz["luma_only_p"]),
            "correlated": bool(rng.random() < nz["correlated_p"]),
        }
    else:
        p["noise"] = None
    cc = cfg["codec"]
    if codecs is None:
        codecs = ffmpeg.available_codecs()
    weights = {k: v for k, v in cc["weights"].items() if k in codecs}
    if weights and rng.random() < cc["p"]:
        codec = _choice(rng, weights)
        lo, hi = cc["crf"][codec]
        presets = cc["presets"][codec]
        p["codec"] = {
            "name": codec,
            "crf": int(rng.integers(lo, hi + 1)),
            "preset": str(presets[int(rng.integers(len(presets)))]),
            "keyint": int(rng.integers(cc["keyint"][0], cc["keyint"][1] + 1)),
            "bframes": int(rng.integers(cc["bframes"][0], cc["bframes"][1] + 1)),
        }
    else:
        p["codec"] = None
    p["seed"] = int(rng.integers(2**31 - 1))
    return p


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def degrade_rgb(hr: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """hr: (T, H, W, 3) float RGB in [0, 1] with H, W divisible by 4.
    Returns ((T, H/2, W/2, 3) uint8 RGB as decoded, info)."""
    t, h, w, _ = hr.shape
    if h % 4 or w % 4:
        raise ValueError(f"HR size must be divisible by 4, got {w}x{h}")
    lw, lh = w // 2, h // 2
    rng = np.random.default_rng(params["seed"])
    frames = []
    for i in range(t):
        x = hr[i].astype(np.float32)
        if params["blur_sigma"] > 0:
            x = _gauss(x, params["blur_sigma"])
        if params["rescale"] > 0:
            s = params["rescale"]
            k1, k2 = params["rescale_kernels"]
            mid = (max(8, round(w * s)), max(8, round(h * s)))
            x = resize(resize(x, mid, k1), (w, h), k2)
        x = resize(x, (lw, lh), params["kernel"])
        if params["sharpen"]:
            sh = params["sharpen"]
            x = x + sh["amount"] * (x - _gauss(x, sh["sigma"]))
        frames.append(np.clip(x, 0.0, 1.0))
    lr = np.stack(frames)
    if params["noise"]:
        nz = params["noise"]
        shape = (t, lh, lw, 1) if nz["luma_only"] else (t, lh, lw, 3)
        n = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        if nz["correlated"]:
            n = np.stack([_gauss(f, 0.7).reshape(f.shape) for f in n])
            n /= max(float(n.std()), 1e-6)
        lr = np.clip(lr + nz["sigma"] * n, 0.0, 1.0)
    info: dict[str, Any] = {"codec": "none", "bpp": None}
    if params["codec"]:
        c = params["codec"]
        out, nbytes = ffmpeg.codec_roundtrip(
            lr, c["name"], c["crf"], c["preset"], keyint=c["keyint"], bframes=c["bframes"]
        )
        info = {"codec": c["name"], "bpp": 8.0 * nbytes / (t * lh * lw)}
        return out, info
    return to_uint8(lr), info


def degrade_clip(hr: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict]:
    """hr: (T, H, W, 3) float RGB. Returns (lr_y (T, H/2, W/2), hr_y (T, H, W), info), both float32
    luma in [0, 1]."""
    lr_rgb, info = degrade_rgb(hr, params)
    lr_y = rgb_to_y(lr_rgb.astype(np.float32) / 255.0)
    hr_y = rgb_to_y(np.clip(hr, 0.0, 1.0))
    return lr_y, hr_y, info
