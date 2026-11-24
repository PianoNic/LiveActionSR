"""Validation on the held-out compressed set: model vs bilinear / bicubic / Lanczos, overall and
per codec."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .data import FullFrameDataset, PairShards
from .metrics import BASELINES, psnr, ssim, upscale_baseline


class Validator:
    def __init__(self, shards: PairShards, limit: int | None = None, batch: int = 16):
        self.ds = FullFrameDataset(shards, limit)
        self.shards = shards
        self.batch = batch
        self._baselines: dict | None = None

    def _groups(self, idx: list[int]) -> list[str]:
        return [self.shards.meta(i).get("codec", "none") for i in idx]

    def _loader(self):
        return DataLoader(self.ds, batch_size=self.batch, shuffle=False, num_workers=0)

    def baselines(self, device) -> dict:
        """Computed once (they do not change during training)."""
        if self._baselines is None:
            acc = {k: defaultdict(list) for k in BASELINES}
            for lr, hr, idx in self._loader():
                lr, hr = lr.to(device), hr.to(device)
                groups = self._groups(idx.tolist())
                for k in BASELINES:
                    up = upscale_baseline(lr, k)
                    _accumulate(acc[k], up, hr, groups)
            self._baselines = {k: _summarise(v) for k, v in acc.items()}
        return self._baselines

    @torch.no_grad()
    def run(self, model: torch.nn.Module, device) -> dict:
        was_training = model.training
        model.eval()
        acc = defaultdict(list)
        for lr, hr, idx in self._loader():
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr).float().clamp(0, 1)
            _accumulate(acc, pred, hr, self._groups(idx.tolist()))
        model.train(was_training)
        res = {"model": _summarise(acc), **self.baselines(device)}
        res["gain_vs_lanczos_db"] = res["model"]["psnr"] - res["lanczos"]["psnr"]
        res["gain_vs_bicubic_db"] = res["model"]["psnr"] - res["bicubic"]["psnr"]
        return res


def _accumulate(acc, pred, hr, groups) -> None:
    p = psnr(pred, hr).tolist()
    s = ssim(pred, hr).tolist()
    for g, a, b in zip(groups, p, s):
        for key in ("all", g):
            acc[(key, "psnr")].append(a)
            acc[(key, "ssim")].append(b)


def _summarise(acc) -> dict:
    out: dict = {}
    for (group, metric), vals in acc.items():
        v = sum(vals) / len(vals)
        if group == "all":
            out[metric] = v
            out["n"] = len(vals)
        else:
            out.setdefault("by_codec", {}).setdefault(group, {})[metric] = v
    return out


def format_report(res: dict) -> str:
    lines = [f"{'':10s} {'PSNR-Y':>8s} {'SSIM':>7s}"]
    for k in ("model", *BASELINES):
        lines.append(f"{k:10s} {res[k]['psnr']:8.3f} {res[k]['ssim']:7.4f}")
    lines.append(f"gain vs lanczos {res['gain_vs_lanczos_db']:+.3f} dB, vs bicubic {res['gain_vs_bicubic_db']:+.3f} dB")
    for codec, m in sorted(res["model"].get("by_codec", {}).items()):
        base = res["lanczos"]["by_codec"][codec]["psnr"]
        lines.append(f"  {codec:6s} model {m['psnr']:.3f} dB  lanczos {base:.3f} dB  ({m['psnr'] - base:+.3f})")
    return "\n".join(lines)
