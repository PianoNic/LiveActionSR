"""Convert an EfRLFN checkpoint into the plugin's graph weights format, and prove the result.

    python convert.py EfRLFN-2x-model.pt -o ../../models/ultra.json

EfRLFN (Bogatyrev et al., "Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for
Streaming Content", ICLR 2026, https://github.com/EvgeneyBogatyrev/EfRLFN, MIT) is re-implemented
below exactly as upstream's code/model.py and code/blocks.py, so this script needs nothing from that
repository but the checkpoint. The checkpoint is a plain state dict and is loaded with
`torch.load(weights_only=True)`: no pickled code runs.

The graph format (documented in the repository README and in src/graph.ts):

    { "arch": "efrlfn", "scale": 2, "input": "rgb", "output": "rgb", "channels": 52,
      "nodes": [ { "name": ..., "op": ..., "inputs": [...], ... }, ... ] }

* `input` is the tensor called "input": the frame's R'G'B' in 0..1 (3 channels).
* `conv`: cross-correlation, weights [out][in][k][k] (PyTorch OIHW flattened), "same" size,
  `padding` "zero" (PyTorch's default, what EfRLFN uses) or "replicate", then `activation`.
* `add` / `mul`: elementwise; the second operand may be a per-channel vector (broadcast).
* `global_avg_pool`: per-channel mean over the whole frame -> a vector.
* `channel_conv`: 1-D convolution across the channels of a vector (ECA), zero padded, then
  `activation`.
* `depth_to_space`: nn.PixelShuffle; the last node. With `output: "rgb"` its 3 channels are the
  picture (clamped to 0..1 by the plugin); there is no bilinear residual.

Tensors are lists of numbers or `{"dtype": "float16" | "float32", "base64": ...}` (little endian).

Verification (always on): the network is run three ways on a random input and on `--image`:
upstream-equivalent PyTorch with the original weights, PyTorch with the weights exactly as stored in
the JSON, and `numpy_forward` of the JSON. The last two must agree to 1e-4 (they compute the same
function); the first pair shows what storing the weights as float16 costs.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "efrlfn"
SMALL = 64  # tensors up to this many values stay plain JSON numbers


# ---- EfRLFN, as upstream ------------------------------------------------------------------------


class ECABlock(nn.Module):
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1).unsqueeze(-1)
        return x * torch.sigmoid(y)


def conv_layer(cin: int, cout: int, k: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, k, padding=(k - 1) // 2)


class ERLFB(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1_r = conv_layer(c, c, 3)
        self.c2_r = conv_layer(c, c, 3)
        self.c3_r = conv_layer(c, c, 3)
        self.c5 = conv_layer(c, c, 1)
        self.eca = ECABlock()

    def forward(self, x):
        out = torch.tanh(self.c1_r(x))
        out = torch.tanh(self.c2_r(out))
        out = torch.tanh(self.c3_r(out))
        return self.eca(self.c5(out + x))


class EfRLFN(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, feat: int = 52, upscale: int = 2):
        super().__init__()
        self.conv_1 = conv_layer(in_ch, feat, 3)
        for i in range(1, 7):
            setattr(self, f"block_{i}", ERLFB(feat))
        self.conv_2 = conv_layer(feat, feat, 3)
        self.upsampler = nn.Sequential(
            conv_layer(feat, out_ch * upscale**2, 3), nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        feature = self.conv_1(x)
        out = feature
        for i in range(1, 7):
            out = getattr(self, f"block_{i}")(out)
        return self.upsampler(self.conv_2(out + feature))


def load_efrlfn(path: str | Path) -> EfRLFN:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "conv_1.weight" not in state and "state_dict" in state:  # Lightning .ckpt
        state = {k[len("network.") :]: v for k, v in state["state_dict"].items() if k.startswith("network.")}
    feat = state["conv_1.weight"].shape[0]
    out_ch_s2 = state["upsampler.0.weight"].shape[0]
    scale = int(round((out_ch_s2 / 3) ** 0.5))
    model = EfRLFN(feat=feat, upscale=scale)
    model.load_state_dict(state, strict=True)
    return model.eval()


# ---- export -------------------------------------------------------------------------------------


def encode(t: torch.Tensor, dtype: str) -> list[float] | dict:
    a = t.detach().cpu().float().flatten().numpy()
    if a.size <= SMALL:
        return [float(f"{v:.9g}") for v in a.tolist()]
    raw = a.astype("<f2" if dtype == "float16" else "<f4").tobytes()
    return {"dtype": dtype, "base64": base64.b64encode(raw).decode("ascii")}


def decode(value) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    dtype = {"float16": "<f2", "float32": "<f4"}[value["dtype"]]
    return np.frombuffer(base64.b64decode(value["base64"]), dtype=dtype).astype(np.float32)


def model_to_graph(model: EfRLFN, dtype: str) -> dict:
    feat = model.conv_1.out_channels
    nodes: list[dict] = []

    def conv(name: str, src: str, layer: nn.Conv2d, activation: str = "none") -> str:
        nodes.append(
            {
                "name": name,
                "op": "conv",
                "inputs": [src],
                "in": layer.in_channels,
                "out": layer.out_channels,
                "kernel": layer.kernel_size[0],
                "padding": "zero",
                "activation": activation,
                "weights": encode(layer.weight, dtype),
                "bias": encode(layer.bias, dtype),
            }
        )
        return name

    feature = conv("conv_1", "input", model.conv_1)
    x = feature
    for i in range(1, 7):
        block: ERLFB = getattr(model, f"block_{i}")
        p = f"block_{i}"
        h = conv(f"{p}.c1_r", x, block.c1_r, "tanh")
        h = conv(f"{p}.c2_r", h, block.c2_r, "tanh")
        h = conv(f"{p}.c3_r", h, block.c3_r, "tanh")
        nodes.append({"name": f"{p}.skip", "op": "add", "inputs": [h, x]})
        h = conv(f"{p}.c5", f"{p}.skip", block.c5)
        nodes.append({"name": f"{p}.pool", "op": "global_avg_pool", "inputs": [h]})
        eca = block.eca.conv
        nodes.append(
            {
                "name": f"{p}.attention",
                "op": "channel_conv",
                "inputs": [f"{p}.pool"],
                "kernel": eca.kernel_size[0],
                "activation": "sigmoid",
                "weights": encode(eca.weight, dtype),
            }
        )
        nodes.append({"name": p, "op": "mul", "inputs": [h, f"{p}.attention"]})
        x = p
    nodes.append({"name": "global_skip", "op": "add", "inputs": [x, feature]})
    conv("conv_2", "global_skip", model.conv_2)
    up = model.upsampler[0]
    scale = model.upsampler[1].upscale_factor
    conv("upsampler", "conv_2", up)
    nodes.append({"name": "pixel_shuffle", "op": "depth_to_space", "inputs": ["upsampler"], "scale": scale})
    return {
        "arch": ARCH,
        "scale": scale,
        "channels": feat,
        "input": "rgb",
        "output": "rgb",
        "source": "EfRLFN x2, https://github.com/EvgeneyBogatyrev/EfRLFN (MIT)",
        "nodes": nodes,
    }


def graph_to_model(graph: dict) -> EfRLFN:
    """The PyTorch model with exactly the weights the JSON holds (float16-rounded if so stored)."""
    model = EfRLFN(feat=graph["channels"], upscale=graph["scale"])
    by_name = {n["name"]: n for n in graph["nodes"]}
    state = {}
    for key, param in model.state_dict().items():
        module, kind = key.rsplit(".", 1)
        if module.endswith(".eca.conv"):
            node = by_name[module.replace(".eca.conv", ".attention")]
        else:
            node = by_name["upsampler" if module == "upsampler.0" else module]
        state[key] = torch.from_numpy(decode(node[{"weight": "weights", "bias": "bias"}[kind]]).reshape(param.shape).copy())
    model.load_state_dict(state, strict=True)
    return model.eval()


# ---- numpy reference of the JSON ----------------------------------------------------------------

ACTIVATIONS = {
    "none": lambda v, n: v,
    "relu": lambda v, n: np.maximum(v, 0),
    "lrelu": lambda v, n: np.where(v >= 0, v, v * n.get("slope", 0.01)),
    "sigmoid": lambda v, n: 1 / (1 + np.exp(-v)),
    "tanh": lambda v, n: np.tanh(v),
}


def np_conv(x: np.ndarray, node: dict) -> np.ndarray:
    """x: [C, H, W]. Same-size cross-correlation with zero or replicate padding."""
    cin, cout, k = node["in"], node["out"], node["kernel"]
    d = node.get("dilation", 1)
    w = decode(node["weights"]).reshape(cout, cin, k, k).astype(np.float64)
    b = decode(node["bias"]).astype(np.float64) if "bias" in node else np.zeros(cout)
    r = d * (k - 1) // 2
    mode = "edge" if node.get("padding", "zero") == "replicate" else "constant"
    xp = np.pad(x.astype(np.float64), ((0, 0), (r, r), (r, r)), mode=mode)
    _, h, wd = x.shape
    out = np.zeros((cout, h, wd))
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky * d : ky * d + h, kx * d : kx * d + wd]
            out += np.tensordot(w[:, :, ky, kx], patch, axes=(1, 0))
    out += b[:, None, None]
    return ACTIVATIONS[node.get("activation", "none")](out, node)


def numpy_forward(graph: dict, image: np.ndarray) -> np.ndarray:
    """image: [3, H, W] in 0..1 -> [3, H*s, W*s], unclamped, float64."""
    tensors: dict[str, np.ndarray] = {"input": image.astype(np.float64)}
    for node in graph["nodes"]:
        a = [tensors[name] for name in node["inputs"]]
        op = node["op"]
        if op == "conv":
            y = np_conv(a[0], node)
        elif op in ("add", "mul"):
            other = a[1] if a[1].ndim == 3 else a[1][:, None, None]
            y = a[0] + other if op == "add" else a[0] * other
            y = ACTIVATIONS[node.get("activation", "none")](y, node)
        elif op == "activation":
            y = ACTIVATIONS[node["activation"]](a[0], node)
        elif op == "global_avg_pool":
            y = a[0].mean(axis=(1, 2))
        elif op == "channel_conv":
            k = node["kernel"]
            w = decode(node["weights"]).astype(np.float64)
            v = np.pad(a[0], (k // 2, k // 2))
            y = np.array([np.dot(w, v[c : c + k]) for c in range(a[0].shape[0])])
            if "bias" in node:
                y = y + decode(node["bias"])[0]
            y = ACTIVATIONS[node.get("activation", "none")](y, node)
        elif op == "depth_to_space":
            s = node["scale"]
            c, h, w = a[0].shape
            y = a[0].reshape(c // (s * s), s, s, h, w).transpose(0, 3, 1, 4, 2).reshape(c // (s * s), h * s, w * s)
        else:
            raise ValueError(f"unknown op {op}")
        tensors[node["name"]] = y
    return tensors[graph["nodes"][-1]["name"]]


# ---- verification -------------------------------------------------------------------------------


def torch_forward(model: nn.Module, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(image[None].astype(np.float32)))[0].double().numpy()


def load_image(path: str | Path) -> np.ndarray:
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return rgb.transpose(2, 0, 1)


def verify(original: EfRLFN, graph: dict, image: np.ndarray, label: str) -> float:
    stored = graph_to_model(graph)
    t_orig = torch_forward(original, image)
    t_stored = torch_forward(stored, image)
    n = numpy_forward(graph, image)
    err = float(np.abs(n - t_stored).max())
    q = np.abs(np.clip(t_orig, 0, 1) - np.clip(t_stored, 0, 1))
    psnr = 10 * np.log10(1 / max(float((q**2).mean()), 1e-20))
    print(
        f"{label}: numpy vs PyTorch (same weights) max |err| {err:.2e}; "
        f"stored vs original weights max |err| {q.max():.2e}, PSNR {psnr:.1f} dB"
    )
    return err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="EfRLFN state dict (.pt) or Lightning checkpoint (.ckpt)")
    parser.add_argument("-o", "--out", required=True, help="JSON file to write")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--image", action="append", default=[], help="also verify on this image")
    parser.add_argument(
        "--reference",
        nargs=2,
        metavar=("IMAGE", "OUT"),
        help="write numpy_forward of IMAGE as raw little-endian float32 [3, H*s, W*s] to OUT",
    )
    args = parser.parse_args(argv)

    model = load_efrlfn(args.checkpoint)
    params = sum(p.numel() for p in model.parameters())
    graph = model_to_graph(model, args.dtype)
    text = json.dumps(graph, indent=2)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"{args.out}: {params} parameters, {len(graph['nodes'])} nodes, {len(text) / 1e6:.2f} MB")

    graph = json.loads(text)  # verify what was written, not what was meant
    rng = np.random.default_rng(0)
    worst = verify(model, graph, rng.random((3, 40, 56), dtype=np.float32), "random 56x40")
    for path in args.image:
        worst = max(worst, verify(model, graph, load_image(path), Path(path).name))
    if args.reference:
        image = load_image(args.reference[0])
        numpy_forward(graph, image).astype("<f4").tofile(args.reference[1])
        print(f"reference for {args.reference[0]} -> {args.reference[1]}")
    if worst >= 1e-4:
        print(f"FAIL: numpy and PyTorch differ by {worst:.2e} (limit 1e-4)", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
