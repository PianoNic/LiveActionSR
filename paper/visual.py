"""Visual comparison figure: one textured crop per held-out scene, Track 2 (x264 CRF 26), in colour.

For every validation scene it takes the clips whose Lanczos PSNR is lowest (textured content), runs
every method on the middle frame, and keeps the 96x96 crop with enough detail where `quality` gains
most over Lanczos. Luma models get bilinear chroma, as in the plugin.

    python paper/visual.py --hr training/work/large/hr/val --out paper/figures/visual.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark import CONDITIONS, clips, load_clip, run_rgb_model  # noqa: E402
from benchmark import degrade_rgb, graph_to_model, load_any_model, rgb_to_y, run_luma_model  # noqa: E402
from evaluate import luma_to_rgb  # noqa: E402

CROP = 96
NAMES = {"netflix_chimera_pierseaside": "Chimera", "svt_old_town_cross": "Old Town Cross",
         "tears_of_steel_val": "Tears of Steel", "tum_pedestrian_area": "Pedestrian Area",
         "uvg_campus_view": "UVG Campus View"}  # fmt: skip


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(np.mean((a.astype(np.float64) - b) ** 2), 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="x264")
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()
    root = HERE.parent
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    luma = {n: load_any_model(str(root / f"models/{n}.json")).to(dev) for n in ("balanced", "quality")}
    ef = graph_to_model(json.loads((root / "models/ultra.json").read_text(encoding="utf-8"))).float().eval().to(dev)
    params = {"blur_sigma": 0.0, "rescale": 0.0, "rescale_kernels": ["bicubic", "bicubic"], "kernel": "bicubic",
              "sharpen": None, "noise": None, "seed": 0, "codec": CONDITIONS[args.condition]}  # fmt: skip

    by_scene: dict[str, list[Path]] = {}
    for c in clips(Path(args.hr), None):
        by_scene.setdefault(c.parent.name, []).append(c)

    picks = []
    for scene, items in by_scene.items():
        best = None
        for clip in items[:: max(1, len(items) // 12)]:  # a dozen clips per scene
            hr = load_clip(clip)
            lr = degrade_rgb(hr, params)[0].astype(np.float32) / 255.0
            m = len(hr) // 2
            hr_rgb, lr_rgb = np.clip(hr[m], 0, 1), lr[m]
            hr_y = rgb_to_y(hr_rgb[None])[0]
            lr_y = rgb_to_y(lr_rgb[None])
            h, w = hr_y.shape
            outs = {
                "Bicubic": cv2.resize(lr_rgb, (w, h), interpolation=cv2.INTER_CUBIC),
                "Lanczos": cv2.resize(lr_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4),
                "EfRLFN": run_rgb_model(ef, lr_rgb[None], dev)[0],
            }
            for n, model in luma.items():
                outs[f"Ours: {n}"] = luma_to_rgb(run_luma_model(model, lr_y, dev), lr_rgb[None])[0]
            outs = {k: np.clip(v, 0, 1) for k, v in outs.items()}
            ys = {k: rgb_to_y(v[None])[0] for k, v in outs.items()}
            for y in range(8, h - CROP - 8, 8):
                for x in range(8, w - CROP - 8, 8):
                    ref = hr_y[y : y + CROP, x : x + CROP]
                    if ref.std() < 0.06:  # skip flat crops: nothing to see
                        continue
                    gain = psnr(ys["Ours: quality"][y : y + CROP, x : x + CROP], ref) - psnr(
                        ys["Lanczos"][y : y + CROP, x : x + CROP], ref
                    )
                    if best is None or gain > best[0]:
                        best = (gain, scene, hr_rgb, hr_y, outs, ys, (y, x))
        if best:
            picks.append(best)
    picks.sort(key=lambda p: -p[0])
    picks = picks[: args.rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    order = ["HR", "Bicubic", "Lanczos", "EfRLFN", "Ours: balanced", "Ours: quality"]
    fig, axes = plt.subplots(len(picks), len(order) + 1, figsize=(7.0, 1.2 * len(picks) + 0.15))
    axes = np.atleast_2d(axes)
    for r, (_, scene, hr_rgb, hr_y, outs, ys, (y, x)) in enumerate(picks):
        ax = axes[r, 0]
        ax.imshow(hr_rgb)
        ax.add_patch(patches.Rectangle((x, y), CROP, CROP, fill=False, ec="#e63946", lw=1.2))
        ax.set_ylabel(NAMES.get(scene, scene), fontsize=7)
        ref = hr_y[y : y + CROP, x : x + CROP]
        for c, key in enumerate(order, start=1):
            img = hr_rgb if key == "HR" else outs[key]
            ax = axes[r, c]
            ax.imshow(img[y : y + CROP, x : x + CROP], interpolation="nearest")
            sub = "reference" if key == "HR" else f"{psnr(ys[key][y : y + CROP, x : x + CROP], ref):.2f} dB"
            ax.set_xlabel(sub, fontsize=6.5, labelpad=1)
            if r == 0:
                ax.set_title(key, fontsize=7.5)
        if r == 0:
            axes[r, 0].set_title("Frame", fontsize=7.5)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0.2, w_pad=0.25, h_pad=0.35)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out, [(p[1], round(p[0], 2)) for p in picks])


if __name__ == "__main__":
    main()
