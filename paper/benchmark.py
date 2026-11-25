"""Benchmark for the paper: every method on the held-out validation clips, in two tracks.

* Track 1 (clean): bicubic 2x downscale, no codec. The classic SR setting.
* Track 2 (streaming): the same downscale, then one fixed round-trip per codec at a typical
  streaming quality: x264 CRF 26 (medium), x265 CRF 28 (medium), SVT-AV1 CRF 38 (preset 8).

Every method sees the same decoded low-resolution frames. Scores are PSNR-Y and SSIM-Y (BT.709 luma,
4 px border shaved), averaged over all frames of all clips; VMAF (luma) is added per clip when
ffmpeg has libvmaf. Luma models upscale Y and take bilinear chroma, like the plugin; EfRLFN gets RGB.

    python paper/benchmark.py --hr training/work/large/hr/val --out paper/results \
        --model balanced=models/balanced.json --model quality=models/quality.json \
        --model "balanced (3k clips)=training/work/full/runs/balanced/best.pt" \
        --efrlfn models/ultra.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools" / "efrlfn"))

from convert import graph_to_model  # noqa: E402
from evaluate import load_any_model, run_luma_model, vmaf  # noqa: E402
from lasr.color import rgb_to_y  # noqa: E402
from lasr.degrade import degrade_rgb  # noqa: E402
from lasr.imio import read_rgb  # noqa: E402
from lasr.metrics import psnr, ssim, upscale_baseline  # noqa: E402

CONDITIONS = {
    "clean": None,
    "x264": {"name": "x264", "crf": 26, "preset": "medium", "keyint": 48, "bframes": 3},
    "x265": {"name": "x265", "crf": 28, "preset": "medium", "keyint": 48, "bframes": 3},
    "av1": {"name": "av1", "crf": 38, "preset": "8", "keyint": 48, "bframes": 3},
}


def clips(root: Path, limit: int | None) -> list[Path]:
    out = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())
    if limit:  # spread evenly over the sources
        step = max(1, len(out) // limit)
        out = out[::step][:limit]
    return out


def load_clip(path: Path) -> np.ndarray:
    frames = np.stack([read_rgb(f) for f in sorted(path.glob("f*.png"))]).astype(np.float32)
    t, h, w, _ = frames.shape
    return frames[:, : h - h % 4, : w - w % 4]


@torch.no_grad()
def run_rgb_model(model: torch.nn.Module, lr_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(lr_rgb).permute(0, 3, 1, 2).to(device)
    y = torch.cat([model(x[i : i + 1]).clamp(0, 1) for i in range(len(x))])
    return y.permute(0, 2, 3, 1).cpu().numpy()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hr", required=True, help="folder of HR validation clips (<source>/<clip>/f*.png)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH", help="luma model (.json/.pt)")
    ap.add_argument("--efrlfn", default=None, help="EfRLFN graph JSON (models/ultra.json)")
    ap.add_argument("--limit", type=int, default=None, help="number of clips")
    ap.add_argument("--vmaf", action="store_true")
    ap.add_argument("--save-frames", type=int, default=8, help="clips whose middle frame is saved for figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    luma = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        luma[name] = load_any_model(path).to(device)
    efrlfn = None
    if args.efrlfn:
        efrlfn = graph_to_model(json.loads(Path(args.efrlfn).read_text(encoding="utf-8"))).float().eval().to(device)

    items = clips(Path(args.hr), args.limit)
    print(f"{len(items)} clips, methods: bilinear bicubic lanczos {' '.join(luma)}{' EfRLFN' if efrlfn else ''}")
    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    timing: dict[str, list[float]] = defaultdict(list)
    t0 = time.time()

    for n, clip in enumerate(items):
        hr = load_clip(clip)
        hr_y = rgb_to_y(np.clip(hr, 0, 1))
        ref = torch.from_numpy(hr_y)[:, None]
        for cond, codec in CONDITIONS.items():
            params = {"blur_sigma": 0.0, "rescale": 0.0, "rescale_kernels": ["bicubic", "bicubic"],
                      "kernel": "bicubic", "sharpen": None, "noise": None, "seed": 0, "codec": codec}  # fmt: skip
            lr_u8, _ = degrade_rgb(hr, params)
            lr_rgb = lr_u8.astype(np.float32) / 255.0
            lr_y = rgb_to_y(lr_rgb)
            lr_t = torch.from_numpy(lr_y)[:, None]

            outs: dict[str, np.ndarray] = {}
            for k in ("bilinear", "bicubic", "lanczos"):
                outs[k] = upscale_baseline(lr_t, k)[:, 0].clamp(0, 1).numpy()
            for name, model in luma.items():
                s = time.perf_counter()
                outs[name] = run_luma_model(model, lr_y, device)
                timing[name].append(time.perf_counter() - s)
            if efrlfn is not None:
                rgb = run_rgb_model(efrlfn, lr_rgb, device)
                outs["EfRLFN"] = rgb_to_y(rgb)

            for k, v in outs.items():
                pt = torch.from_numpy(np.ascontiguousarray(v))[:, None]
                scores[cond][k]["psnr"] += psnr(pt, ref).tolist()
                scores[cond][k]["ssim"] += ssim(pt, ref).tolist()
                if args.vmaf and cond != "clean":
                    vm = vmaf(v, hr_y, out)
                    if vm is not None:
                        scores[cond][k]["vmaf"].append(vm)

            if n < args.save_frames:
                mid = len(hr_y) // 2
                tag = f"{clip.parent.name}__{clip.name}__{cond}"
                np.savez_compressed(
                    out / "frames" / f"{tag}.npz",
                    hr=hr_y[mid].astype(np.float16),
                    lr=lr_y[mid].astype(np.float16),
                    **{k.replace(" ", "_"): v[mid].astype(np.float16) for k, v in outs.items()},
                )
        if (n + 1) % 10 == 0 or n + 1 == len(items):
            print(f"  {n + 1}/{len(items)} clips, {time.time() - t0:.0f} s", flush=True)

    summary = {
        cond: {k: {m: float(np.mean(v)) for m, v in ms.items()} | {"n": len(ms["psnr"])} for k, ms in per.items()}
        for cond, per in scores.items()
    }
    # Track 2 = the three codecs pooled.
    pooled: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cond in ("x264", "x265", "av1"):
        for k, ms in scores[cond].items():
            for m, v in ms.items():
                pooled[k][m] += v
    summary["streaming"] = {k: {m: float(np.mean(v)) for m, v in ms.items()} | {"n": len(ms["psnr"])}
                            for k, ms in pooled.items()}  # fmt: skip
    meta = {"clips": len(items), "conditions": {k: v for k, v in CONDITIONS.items()}, "device": str(device)}
    (out / "benchmark.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=2), encoding="utf-8")

    for cond in ("clean", "x264", "x265", "av1", "streaming"):
        print(f"\n[{cond}]")
        for k, m in sorted(summary[cond].items(), key=lambda kv: kv[1]["psnr"]):
            vm = f"  VMAF {m['vmaf']:.2f}" if "vmaf" in m else ""
            print(f"  {k:24s} PSNR-Y {m['psnr']:.3f}  SSIM-Y {m['ssim']:.4f}{vm}")


if __name__ == "__main__":
    main()
