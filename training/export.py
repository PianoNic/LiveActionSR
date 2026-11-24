"""Export a trained checkpoint to the JSON weight format the browser plugin consumes (and to ONNX
for debugging with ORT-Web / Netron).

    python export.py runs/balanced/best.pt --out ../models/balanced.json --onnx runs/balanced/balanced.onnx

JSON format (consumed by the plugin, do not change without updating it):

    {
      "arch": "artcnn-like", "scale": 2, "channels": F, "residual": "bilinear",
      "layers": [
        {"type": "conv3x3", "in": a, "out": b, "relu": bool,
         "weights": [...],   # layout [out][in][ky][kx], row-major (= PyTorch OIHW, flattened)
         "bias": [...]},     # length out
        ...,
        {"type": "depth_to_space", "scale": 2}
      ]
    }

Semantics the plugin must reproduce (all checked by `numpy_forward` below and the tests):

* input: one channel, Y' = 0.2126 R' + 0.7152 G' + 0.0722 B' of the video's R'G'B' in [0, 1];
* conv3x3: cross-correlation (PyTorch convention, no kernel flip), weight index
  ((o * in + i) * 3 + ky) * 3 + kx, tap (ky, kx) reads input pixel (y + ky - 1, x + kx - 1),
  out-of-range coordinates clamped to the edge (replicate padding) at every layer;
* relu: max(0, x) after the conv when "relu" is true;
* depth_to_space(2): out[2y + i, 2x + j] = in[channel i * 2 + j][y, x] (PyTorch pixel_shuffle);
* residual "bilinear": output = depth_to_space(...) + bilinear_2x(input), where bilinear_2x samples
  the input at ((x + 0.5) / 2 - 0.5, (y + 0.5) / 2 - 0.5) with clamp-to-edge (a GPU linear sampler
  at output pixel centres; weights 0.75 / 0.25);
* the result is clamped to [0, 1] by the consumer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lasr.models import LumaSR

ARCH = "artcnn-like"


def load_checkpoint(path: str | Path, device: str = "cpu") -> LumaSR:
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ck["model_cfg"]
    model = LumaSR(mc["channels"], mc["blocks"], mc.get("scale", 2), rep=mc.get("rep", False))
    model.load_state_dict(ck["model"])
    return model.eval()


def _floats(t: torch.Tensor) -> list[float]:
    # 9 significant digits round-trip float32 exactly; shorter than repr(float64).
    return [float(f"{v:.9g}") for v in t.detach().cpu().float().flatten().tolist()]


def model_to_json(model: LumaSR) -> dict:
    layers = []
    for s in model.layer_specs():
        layers.append(
            {
                "type": "conv3x3",
                "in": s["in"],
                "out": s["out"],
                "relu": s["relu"],
                "weights": _floats(s["w"]),
                "bias": _floats(s["b"]),
            }
        )
    layers.append({"type": "depth_to_space", "scale": model.scale})
    return {
        "arch": ARCH,
        "scale": model.scale,
        "channels": model.channels,
        "residual": "bilinear",
        "layers": layers,
    }


def validate_doc(doc: dict) -> None:
    """Structural checks of the plugin contract. Raises ValueError."""
    for key in ("arch", "scale", "channels", "residual", "layers"):
        if key not in doc:
            raise ValueError(f"missing key {key!r}")
    if doc["arch"] != ARCH or doc["scale"] != 2 or doc["residual"] != "bilinear":
        raise ValueError("unsupported arch/scale/residual")
    layers = doc["layers"]
    if not layers or layers[-1] != {"type": "depth_to_space", "scale": 2}:
        raise ValueError("last layer must be depth_to_space with scale 2")
    cin = 1
    for i, layer in enumerate(layers[:-1]):
        if layer["type"] != "conv3x3":
            raise ValueError(f"layer {i}: unexpected type {layer['type']}")
        if layer["in"] != cin:
            raise ValueError(f"layer {i}: in={layer['in']} but previous out={cin}")
        if len(layer["weights"]) != layer["out"] * layer["in"] * 9 or len(layer["bias"]) != layer["out"]:
            raise ValueError(f"layer {i}: weight/bias length mismatch")
        if not isinstance(layer["relu"], bool):
            raise ValueError(f"layer {i}: relu must be bool")
        cin = layer["out"]
    if cin != 4:
        raise ValueError("last conv must output scale^2 = 4 channels")


# ------------------------------------------------------------------ numpy reference forward


def bilinear_up2_np(y: np.ndarray) -> np.ndarray:
    """Half-pixel 2x bilinear with clamp-to-edge, (H, W) -> (2H, 2W)."""

    def axis(a: np.ndarray, ax: int) -> np.ndarray:
        prev = np.concatenate([np.take(a, [0], ax), np.take(a, range(a.shape[ax] - 1), ax)], ax)
        nxt = np.concatenate([np.take(a, range(1, a.shape[ax]), ax), np.take(a, [-1], ax)], ax)
        even = 0.25 * prev + 0.75 * a
        odd = 0.75 * a + 0.25 * nxt
        out = np.stack([even, odd], ax + 1)
        shape = list(a.shape)
        shape[ax] *= 2
        return out.reshape(shape)

    return axis(axis(y, 0), 1)


def conv3x3_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x (C, H, W), w (O, C, 3, 3), replicate padding -> (O, H, W)."""
    _, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros((w.shape[0], h, wd), dtype=x.dtype) + b[:, None, None].astype(x.dtype)
    for ky in range(3):
        for kx in range(3):
            out += np.einsum("oc,chw->ohw", w[:, :, ky, kx].astype(x.dtype), xp[:, ky : ky + h, kx : kx + wd])
    return out


def depth_to_space_np(x: np.ndarray, r: int) -> np.ndarray:
    c, h, w = x.shape
    oc = c // (r * r)
    return x.reshape(oc, r, r, h, w).transpose(0, 3, 1, 4, 2).reshape(oc, h * r, w * r)


def numpy_forward(doc: dict, y: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Run an exported JSON model on (H, W) luma -> (2H, 2W), unclamped. Pass dtype=np.float16
    to approximate a shader-f16 implementation."""
    x = y.astype(dtype)[None]
    for layer in doc["layers"]:
        if layer["type"] == "conv3x3":
            w = np.asarray(layer["weights"], np.float32).reshape(layer["out"], layer["in"], 3, 3)
            x = conv3x3_np(x, w.astype(dtype), np.asarray(layer["bias"], dtype))
            if layer["relu"]:
                x = np.maximum(x, 0)
        elif layer["type"] == "depth_to_space":
            x = depth_to_space_np(x, layer["scale"])
        else:
            raise ValueError(layer["type"])
    return (x[0] + bilinear_up2_np(y.astype(dtype))).astype(np.float32)


# ---------------------------------------------------------------------------------- ONNX


def export_onnx(model: LumaSR, path: str | Path, size: int = 64) -> None:
    model = model.reparameterize().eval() if model.rep else model.eval()
    dummy = torch.rand(1, 1, size, size)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(
        input_names=["luma"],
        output_names=["luma_2x"],
        dynamic_axes={"luma": {2: "h", 3: "w"}, "luma_2x": {2: "h2", 3: "w2"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kw)
    except Exception:  # legacy exporter unavailable: dynamo exporter (needs onnxscript)
        torch.onnx.export(model, (dummy,), str(path), dynamo=True, **kw)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="JSON output path, e.g. ../models/balanced.json")
    ap.add_argument("--onnx", default=None, help="optional ONNX output path")
    args = ap.parse_args(argv)

    model = load_checkpoint(args.checkpoint)
    doc = model_to_json(model)
    validate_doc(doc)

    # Self-check: JSON + numpy must match PyTorch, and report what f16 would cost.
    rng = np.random.default_rng(0)
    y = rng.random((48, 40), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(y)[None, None])[0, 0].numpy()
    out32 = numpy_forward(doc, y)
    out16 = numpy_forward(doc, y, np.float16)
    err32 = float(np.abs(out32 - ref).max())
    err16 = float(np.abs(out16 - ref).max())
    if err32 > 1e-4:
        raise SystemExit(f"export mismatch: max abs error {err32:.2e} > 1e-4")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    n = sum(len(layer.get("weights", [])) + len(layer.get("bias", [])) for layer in doc["layers"])
    print(f"wrote {args.out}: {n} params, max |json - torch| = {err32:.2e}, f16 max err = {err16:.2e}")
    if args.onnx:
        export_onnx(model, args.onnx)
        print(f"wrote {args.onnx}")


if __name__ == "__main__":
    main()
