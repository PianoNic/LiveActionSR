"""Track 1 per frame: gain of each model over Lanczos, binned by how hard the frame is (Lanczos PSNR).

    python paper/flatness.py --hr training/work/large/hr/val --out paper/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark import clips, load_clip, run_rgb_model  # noqa: E402
from benchmark import graph_to_model, load_any_model, psnr, rgb_to_y, run_luma_model, upscale_baseline, degrade_rgb  # noqa: E402

BINS = [(0, 34), (34, 38), (38, 42), (42, 99)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(HERE.parent))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    models = {n: load_any_model(str(root / f"models/{n}.json")).to(dev) for n in ("balanced", "quality")}
    ef = graph_to_model(json.loads((root / "models/ultra.json").read_text(encoding="utf-8"))).float().eval().to(dev)
    rows = []
    for clip in clips(Path(args.hr), None):
        hr = load_clip(clip)
        ref = torch.from_numpy(rgb_to_y(np.clip(hr, 0, 1)))[:, None]
        params = {"blur_sigma": 0.0, "rescale": 0.0, "rescale_kernels": ["bicubic", "bicubic"],
                  "kernel": "bicubic", "sharpen": None, "noise": None, "seed": 0, "codec": None}  # fmt: skip
        lr_rgb = degrade_rgb(hr, params)[0].astype(np.float32) / 255.0
        lr_y = rgb_to_y(lr_rgb)
        lz = psnr(upscale_baseline(torch.from_numpy(lr_y)[:, None], "lanczos").clamp(0, 1), ref).numpy()
        out = {n: psnr(torch.from_numpy(run_luma_model(m, lr_y, dev))[:, None], ref).numpy() for n, m in models.items()}
        out["EfRLFN"] = psnr(torch.from_numpy(rgb_to_y(run_rgb_model(ef, lr_rgb, dev)))[:, None], ref).numpy()
        for i in range(len(lz)):
            rows.append({"lanczos": float(lz[i]), **{k: float(v[i]) for k, v in out.items()}})
    res = []
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["lanczos"] < hi]
        if not sel:
            continue
        res.append({"bin": [lo, hi], "n": len(sel),
                    **{k: float(np.mean([r[k] - r["lanczos"] for r in sel])) for k in ("balanced", "quality", "EfRLFN")}})  # fmt: skip
    (Path(args.out) / "flatness.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
