"""Losses. L1 (MAE) is the main term: the Amazon ARSR paper found MAE removed the most artifacts,
and fidelity losses keep the network from sharpening codec damage. FFT and edge terms are small
optional extras for crisper detail."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fp = torch.fft.rfft2(pred.float(), norm="ortho")
    ft = torch.fft.rfft2(target.float(), norm="ortho")
    return (fp - ft).abs().mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    k = _SOBEL.to(pred.device, torch.float32)
    kx, ky = k[None, None], k.T[None, None]
    p, t = pred.float(), target.float()
    gp = torch.cat([F.conv2d(p, kx), F.conv2d(p, ky)], 1)
    gt = torch.cat([F.conv2d(t, kx), F.conv2d(t, ky)], 1)
    return F.l1_loss(gp, gt)


class SRLoss(torch.nn.Module):
    def __init__(self, l1: float = 1.0, fft: float = 0.0, edge: float = 0.0):
        super().__init__()
        self.w_l1, self.w_fft, self.w_edge = l1, fft, edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        parts = {"l1": F.l1_loss(pred.float(), target.float())}
        if self.w_fft:
            parts["fft"] = fft_loss(pred, target)
        if self.w_edge:
            parts["edge"] = edge_loss(pred, target)
        total = self.w_l1 * parts["l1"]
        total = total + self.w_fft * parts.get("fft", 0.0) + self.w_edge * parts.get("edge", 0.0)
        return total, {k: float(v.detach()) for k, v in parts.items()}
