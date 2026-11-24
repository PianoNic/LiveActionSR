"""Run a model on a video or a folder of frames and compare it with Lanczos/bicubic (and optionally
another model) side by side, with metrics when a reference exists.

Two modes:

* simulate (`--simulate CODEC:CRF`): the input is a clean HR source. It is downscaled 2x and pushed
  through a real codec round-trip, then upscaled back by every method and scored against the
  source (PSNR-Y, SSIM-Y, optionally VMAF via ffmpeg's libvmaf).
* upscale (default): the input is the real low-resolution stream (e.g. a 720p episode). There is no
  reference, so only side-by-side images are written.

    python evaluate.py --model runs/balanced/best.pt --input clip_1080p.mkv --simulate x264:28 --out eval/b
    python evaluate.py --model ../models/quality.json --input ep_720p.mkv --start 300 --frames 48 --out eval/q
    python evaluate.py --model a.json --compare-model b.json --input frames_dir/ --simulate x265:30 --out eval/ab
    python evaluate.py --model a.json --input lr_frames/ --external-dir artcnn_out/ --external-name ArtCNN --out eval/x
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from export import load_checkpoint
from lasr import ffmpeg
from lasr.color import rgb_to_y, rgb_to_ycbcr, ycbcr_to_rgb
from lasr.degrade import degrade_rgb
from lasr.imio import read_rgb, write_rgb8
from lasr.metrics import psnr, ssim, upscale_baseline
from lasr.models import LumaSR

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def model_from_json(doc: dict) -> LumaSR:
    convs = [layer for layer in doc["layers"] if layer["type"] == "conv3x3"]
    model = LumaSR(doc["channels"], len(convs) - 2, doc["scale"])
    for dst, layer in zip(model.convs, convs):
        if (dst.cin, dst.cout) != (layer["in"], layer["out"]):
            raise ValueError("JSON layer shapes do not match an artcnn-like chain")
        w = torch.tensor(layer["weights"], dtype=torch.float32).reshape(layer["out"], layer["in"], 3, 3)
        dst.conv.weight.data.copy_(w)
        dst.conv.bias.data.copy_(torch.tensor(layer["bias"], dtype=torch.float32))
    return model.eval()


def load_any_model(path: str) -> LumaSR:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return model_from_json(json.load(f))
    m = load_checkpoint(path)
    return m.reparameterize().eval() if m.rep else m


def read_frames(path: str, start: float, count: int) -> np.ndarray:
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)[:count]
        return np.stack([read_rgb(f) for f in files])
    meta = ffmpeg.probe(path)
    return ffmpeg.decode_to_rgb(["-ss", str(start), "-i", path], meta["width"], meta["height"], count)


@torch.no_grad()
def run_luma_model(model: LumaSR, lr_y: np.ndarray, device: torch.device) -> np.ndarray:
    out = []
    for y in lr_y:
        t = torch.from_numpy(y)[None, None].to(device)
        out.append(model(t).clamp(0, 1)[0, 0].cpu().numpy())
    return np.stack(out)


def luma_to_rgb(y2x: np.ndarray, lr_rgb: np.ndarray) -> np.ndarray:
    """Model luma + bilinear-upscaled chroma (what the plugin's balanced/quality tiers do)."""
    out = []
    for y, rgb in zip(y2x, lr_rgb):
        ycc = rgb_to_ycbcr(rgb)
        h, w = y.shape
        cbcr = cv2.resize(ycc[..., 1:], (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(np.clip(ycbcr_to_rgb(np.concatenate([y[..., None], cbcr], -1)), 0, 1))
    return np.stack(out)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))  # putText needs 8-bit
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def center_crop(img: np.ndarray, cw: int, ch: int) -> np.ndarray:
    h, w = img.shape[:2]
    cw, ch = min(cw, w), min(ch, h)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


def vmaf(dist: np.ndarray, ref: np.ndarray, workdir: Path) -> float | None:
    """VMAF (luma) of two gray clips via ffmpeg libvmaf; None if unavailable."""
    t, h, w = ref.shape

    def enc(arr: np.ndarray, path: Path) -> None:
        cmd = [ffmpeg.FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
               "-s", f"{w}x{h}", "-r", "24", "-i", "pipe:0", "-c:v", "ffv1", str(path)]  # fmt: skip
        subprocess.run(cmd, input=(np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes(), check=True)

    try:
        enc(dist, workdir / "dist.mkv")
        enc(ref, workdir / "ref.mkv")
        log = workdir / "vmaf.json"
        log_arg = log.as_posix().replace(":", r"\:")  # filtergraph escaping (Windows drive letters)
        graph = f"[0:v]format=yuv420p[d];[1:v]format=yuv420p[r];[d][r]libvmaf=log_fmt=json:log_path={log_arg}"
        cmd = [ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(workdir / "dist.mkv"),
               "-i", str(workdir / "ref.mkv"), "-lavfi", graph, "-f", "null", "-"]  # fmt: skip
        subprocess.run(cmd, check=True, capture_output=True)
        return float(json.loads(log.read_text())["pooled_metrics"]["vmaf"]["mean"])
    except (subprocess.CalledProcessError, KeyError, OSError, ValueError):
        return None
    finally:
        for f in ("dist.mkv", "ref.mkv"):
            (workdir / f).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint (.pt) or exported JSON")
    ap.add_argument("--name", default="model")
    ap.add_argument("--input", required=True, help="video file or folder of frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--simulate", default=None, metavar="CODEC:CRF", help="e.g. x264:28, x265:30, av1:38, none")
    ap.add_argument("--kernel", default="bicubic", help="downscale kernel in simulate mode")
    ap.add_argument("--compare-model", default=None, help="a second checkpoint/JSON to compare against")
    ap.add_argument("--compare-name", default="compare")
    ap.add_argument("--external-dir", default=None, help="frames upscaled by an external tool (sorted, 2x)")
    ap.add_argument("--external-name", default="external")
    ap.add_argument("--crop", default="640x360", help="side-by-side crop size (WxH, HR pixels)")
    ap.add_argument("--vmaf", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    src = read_frames(args.input, args.start, args.frames)

    ref_y = None
    if args.simulate:
        t, h, w, _ = src.shape
        src = src[:, : h - h % 4, : w - w % 4]
        codec, _, crf = args.simulate.partition(":")
        params = {"blur_sigma": 0.0, "rescale": 0.0, "rescale_kernels": ["bicubic", "bicubic"],
                  "kernel": args.kernel, "sharpen": None, "noise": None, "seed": 0,
                  "codec": None if codec == "none" else
                  {"name": codec, "crf": int(crf), "preset": "8" if codec == "av1" else "medium",
                   "keyint": 48, "bframes": 3}}  # fmt: skip
        lr_u8, info = degrade_rgb(src, params)
        lr_rgb = lr_u8.astype(np.float32) / 255.0
        ref_y = rgb_to_y(src)
        print(f"simulated {args.simulate}: {info}")
    else:
        lr_rgb = src
    lr_y = rgb_to_y(lr_rgb)
    lr_t = torch.from_numpy(lr_y)[:, None]

    results: dict[str, np.ndarray] = {}
    results["lanczos"] = upscale_baseline(lr_t, "lanczos")[:, 0].clamp(0, 1).numpy()
    results["bicubic"] = upscale_baseline(lr_t, "bicubic")[:, 0].clamp(0, 1).numpy()
    results[args.name] = run_luma_model(load_any_model(args.model).to(device), lr_y, device)
    if args.compare_model:
        results[args.compare_name] = run_luma_model(load_any_model(args.compare_model).to(device), lr_y, device)
    ext_rgb = None
    if args.external_dir:
        ext_rgb = read_frames(args.external_dir, 0, len(lr_y))
        results[args.external_name] = rgb_to_y(ext_rgb)

    metrics: dict[str, dict] = {}
    if ref_y is not None:
        ref_t = torch.from_numpy(ref_y)[:, None]
        for k, v in results.items():
            pt = torch.from_numpy(v)[:, None]
            metrics[k] = {"psnr_y": float(psnr(pt, ref_t).mean()), "ssim_y": float(ssim(pt, ref_t).mean())}
            if args.vmaf:
                metrics[k]["vmaf"] = vmaf(v, ref_y, out)
        print(f"{'method':12s} {'PSNR-Y':>8s} {'SSIM-Y':>8s}" + ("   VMAF" if args.vmaf else ""))
        for k, m in metrics.items():
            extra = f" {m['vmaf']:6.2f}" if args.vmaf and m.get("vmaf") is not None else ""
            print(f"{k:12s} {m['psnr_y']:8.3f} {m['ssim_y']:8.4f}{extra}")

    cw, ch = (int(v) for v in args.crop.lower().split("x"))
    lanczos_rgb = np.stack(
        [cv2.resize(f, (f.shape[1] * 2, f.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4) for f in lr_rgb]
    )
    panels_rgb = {"lanczos": np.clip(lanczos_rgb, 0, 1)}
    for k in results:
        if k in ("lanczos", "bicubic"):
            continue
        panels_rgb[k] = (
            ext_rgb if (ext_rgb is not None and k == args.external_name) else luma_to_rgb(results[k], lr_rgb)
        )
    if ref_y is not None:
        panels_rgb["reference"] = src
    for i in range(len(lr_y)):
        tiles = [label(center_crop(v[i], cw, ch), k) for k, v in panels_rgb.items()]
        write_rgb8(out / "side_by_side" / f"frame_{i:05d}.png", np.concatenate(tiles, axis=1))
        if i == 0:
            for k, v in panels_rgb.items():
                write_rgb8(out / "full" / f"{k}.png", v[i])
    summary = {"input": args.input, "simulate": args.simulate, "frames": len(lr_y), "metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'side_by_side'} and {out / 'metrics.json'}")
    return summary


if __name__ == "__main__":
    main()
