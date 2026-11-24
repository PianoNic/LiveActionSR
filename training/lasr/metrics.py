"""PSNR / SSIM on luma and the classical baselines the model has to beat."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SHAVE = 4  # border pixels ignored by every metric (standard SR practice)


def _shave(x: torch.Tensor, s: int = SHAVE) -> torch.Tensor:
    return x[..., s:-s, s:-s] if s > 0 else x


def psnr(pred: torch.Tensor, target: torch.Tensor, shave: int = SHAVE) -> torch.Tensor:
    """Per-image PSNR in dB for (N, 1, H, W) tensors in [0, 1]."""
    p, t = _shave(pred.clamp(0, 1), shave), _shave(target, shave)
    mse = ((p - t) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(pred: torch.Tensor, target: torch.Tensor, shave: int = SHAVE) -> torch.Tensor:
    """Per-image SSIM (Wang et al., 11x11 Gaussian, sigma 1.5) for (N, 1, H, W) in [0, 1]."""
    x, y = _shave(pred.clamp(0, 1), shave).float(), _shave(target, shave).float()
    w = _gauss_window(device=x.device)
    c1, c2 = 0.01**2, 0.03**2
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mx**2
    syy = F.conv2d(y * y, w) - my**2
    sxy = F.conv2d(x * y, w) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


_CV = {"bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


def upscale_baseline(lr: torch.Tensor, kind: str) -> torch.Tensor:
    """(N, 1, h, w) -> (N, 1, 2h, 2w). bilinear = the model's residual base (torch, half-pixel);
    bicubic = Catmull-Rom-like cv2 cubic (a=-0.75); lanczos = cv2 Lanczos4 (8 taps)."""
    if kind == "bilinear":
        return F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)
    flag = _CV[kind]
    arr = lr.detach().cpu().numpy()
    out = np.stack([cv2.resize(a[0], (a.shape[2] * 2, a.shape[1] * 2), interpolation=flag)[None] for a in arr])
    return torch.from_numpy(out).to(lr.device)


BASELINES = ("bilinear", "bicubic", "lanczos")
