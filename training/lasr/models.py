"""ArtCNN-like plain conv nets for 2x luma super-resolution.

The network is a straight chain, because that is what the browser plugin's JSON format can
express (and what is cheapest to run as WGSL):

    y_lr (1ch) -> conv3x3(1->F) relu -> [conv3x3(F->F) relu] x blocks -> conv3x3(F->4)
               -> depth_to_space(2) -> + bilinear_2x(y_lr)

The network predicts a residual over a bilinear upscale, so a weak or wrong prediction degrades
towards plain bilinear instead of towards garbage (the RTX VSR design, see docs/research.md).

Every conv uses replicate padding. A shader that reads the previous layer's texture with clamped
coordinates computes exactly that, so the borders of the browser output match training.

Optionally each conv3x3 can be trained as a RepConv block (3x3 + 1x1 + identity branches, no
batch norm), which `reparameterize()` folds into one plain conv3x3 before export.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

PRESETS: dict[str, dict] = {
    # ~12.3k params, the ArtCNN C4F16 size class ("LA-S" in docs/research.md)
    "balanced": {"channels": 16, "blocks": 5},
    # ~47.7k params, the ArtCNN C4F32 / ARSR-41k size class ("LA-M")
    "quality": {"channels": 32, "blocks": 5},
}


def bilinear_up2(x: torch.Tensor) -> torch.Tensor:
    """2x bilinear with half-pixel centres and clamp-to-edge, i.e. what a GPU sampler with linear
    filtering and clamp addressing returns at the output pixel centres."""
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class RepConv3x3(nn.Module):
    """Training-time 3x3 + 1x1 (+ identity when in == out) block. Folds into a single conv3x3."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.conv3 = nn.Conv2d(cin, cout, 3, padding=0)
        self.conv1 = nn.Conv2d(cin, cout, 1, padding=0)
        self.identity = cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(F.pad(x, (1, 1, 1, 1), mode="replicate")) + self.conv1(x)
        return y + x if self.identity else y

    @torch.no_grad()
    def fused(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.conv3.weight.clone()
        b = self.conv3.bias.clone() + self.conv1.bias
        w[:, :, 1, 1] += self.conv1.weight[:, :, 0, 0]
        if self.identity:
            idx = torch.arange(self.cout)
            w[idx, idx, 1, 1] += 1.0
        return w, b


class Conv3x3(nn.Module):
    """Plain conv3x3 with replicate padding (what gets exported)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.conv = nn.Conv2d(cin, cout, 3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (1, 1, 1, 1), mode="replicate"))

    @torch.no_grad()
    def fused(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conv.weight.clone(), self.conv.bias.clone()


class LumaSR(nn.Module):
    def __init__(self, channels: int = 16, blocks: int = 5, scale: int = 2, rep: bool = False):
        super().__init__()
        if scale != 2:
            raise ValueError("only 2x is supported (the plugin's residual path is bilinear 2x)")
        self.channels, self.blocks, self.scale, self.rep = channels, blocks, scale, rep
        block = RepConv3x3 if rep else Conv3x3
        dims = [1] + [channels] * (blocks + 1) + [scale * scale]
        self.convs = nn.ModuleList(block(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Start as "bilinear + 0": the last layer begins near zero so early training is stable.
        last = self.convs[-1]
        for m in last.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.mul_(0.1)

    def body(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < n - 1:
                x = F.relu(x)
        return F.pixel_shuffle(x, self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, H, W) luma in [0, 1] -> (N, 1, 2H, 2W). Not clamped (the loss sees the raw
        prediction; inference clamps)."""
        return bilinear_up2(x) + self.body(x)

    def layer_specs(self) -> list[dict]:
        """Fused (export-ready) weights per conv layer."""
        n = len(self.convs)
        specs = []
        for i, conv in enumerate(self.convs):
            w, b = conv.fused()
            specs.append({"in": conv.cin, "out": conv.cout, "relu": i < n - 1, "w": w, "b": b})
        return specs

    def reparameterize(self) -> LumaSR:
        """A plain (rep=False) copy with the RepConv branches folded in."""
        plain = LumaSR(self.channels, self.blocks, self.scale, rep=False)
        for dst, spec in zip(plain.convs, self.layer_specs()):
            dst.conv.weight.data.copy_(spec["w"])
            dst.conv.bias.data.copy_(spec["b"])
        return plain.to(next(self.parameters()).device)


def build_model(cfg: dict) -> LumaSR:
    """cfg: {preset: balanced|quality} and/or explicit {channels, blocks, rep}."""
    base = dict(PRESETS.get(cfg.get("preset", "balanced"), PRESETS["balanced"]))
    for k in ("channels", "blocks"):
        if cfg.get(k) is not None:
            base[k] = int(cfg[k])
    return LumaSR(base["channels"], base["blocks"], rep=bool(cfg.get("rep", False)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def plain_param_count(channels: int, blocks: int, scale: int = 2) -> int:
    dims = [1] + [channels] * (blocks + 1) + [scale * scale]
    return sum(a * b * 9 + b for a, b in zip(dims[:-1], dims[1:]))
