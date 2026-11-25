"""Turn results/benchmark.json into the LaTeX tables and macros the paper inputs, and draw the
figures (trade-off plot, visual crops).

    python paper/make_tables.py --results paper/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# name in benchmark.json -> (label, params, 1080p->4K chain ms in the browser, is ours)
METHODS = {
    "bilinear": ("Bilinear", None, None, False),
    "bicubic": ("Bicubic", None, None, False),
    "lanczos": ("Lanczos (8-tap)", None, None, False),
    "EfRLFN": ("EfRLFN~\\cite{efrlfn}", 487_010, 123.4, False),
    "balanced": ("\\textbf{LiveActionSR} \\emph{balanced}", 12_340, 2.95, True),
    "quality": ("\\textbf{LiveActionSR} \\emph{quality}", 47_716, 9.90, True),
}
CODECS = ("x264", "x265", "av1")
CODEC_LABEL = {"x264": "H.264 (x264)", "x265": "HEVC (x265)", "av1": "AV1 (SVT)"}

# Ablation: both model pairs scored on both validation sets during training (val set A = the 3k-clip
# run's 840 pairs, B = the 6.4k-clip run's 1,024 scored pairs). Gain in PSNR-Y over Lanczos, dB.
ABLATION = {
    ("balanced", "3,054 / 35k"): (0.421, 0.422),
    ("balanced", "6,464 / 75k"): (0.439, 0.443),
    ("quality", "3,054 / 35k"): (0.504, 0.506),
    ("quality", "6,464 / 75k"): (0.513, 0.522),
}


def signed(v: float) -> str:
    """+0.41 / $-$0.19 (a real minus sign in LaTeX)."""
    return f"+{v:.2f}" if v >= 0 else f"$-${-v:.2f}"


def marks(values: list[float | None], higher=True) -> list[str]:
    """\\best / \\second wrappers for the top two of a column."""
    real = sorted({v for v in values if v is not None}, reverse=higher)
    out = []
    for v in values:
        if v is None:
            out.append("--")
            continue
        txt = f"{v:.4f}" if v < 1.5 else f"{v:.2f}"
        out.append(f"\\best{{{txt}}}" if v == real[0] else f"\\second{{{txt}}}" if len(real) > 1 and v == real[1] else txt)
    return out


def main_table(res: dict) -> str:
    names = list(METHODS)
    cols = [
        [res["clean"][n]["psnr"] for n in names],
        [res["clean"][n]["ssim"] for n in names],
        [res["streaming"][n]["psnr"] for n in names],
        [res["streaming"][n]["ssim"] for n in names],
        [res["streaming"][n].get("vmaf") for n in names],
    ]
    marked = [marks(c) for c in cols]
    order = sorted(names, key=lambda n: -res["streaming"][n]["psnr"])
    rank = {n: i + 1 for i, n in enumerate(order)}
    lanczos = res["streaming"]["lanczos"]["psnr"]
    rows = []
    for i, n in enumerate(names):
        label, params, ms, ours = METHODS[n]
        p = f"{params / 1000:.1f}k" if params else "--"
        t = f"{ms:.2f}" if ms else "--"
        gain = res["streaming"][n]["psnr"] - lanczos
        g = signed(gain) if n != "lanczos" else "0.00"
        cells = [label, p, t, marked[0][i], marked[1][i], marked[2][i], marked[3][i], marked[4][i], g, str(rank[n])]
        pre = "\\rowcolor{gray!12} " if ours else ""
        rows.append(pre + " & ".join(cells) + " \\\\")
        if n == "lanczos":
            rows.append("\\midrule")
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{\\textbf{Main results} on held-out live-action clips, 2$\\times$. Track~1: clean bicubic "
        "downscale. Track~2: the same plus one x264 / x265 / SVT-AV1 round-trip at streaming quality, pooled. "
        "PSNR / SSIM on BT.709 luma, VMAF on luma. Runtime: complete in-browser chain at 1080p$\\to$4K (WebGPU, RTX~5080). "
        "\\best{Red}: best, \\second{blue}: second. $\\Delta$ and rank by Track~2 PSNR.}\n"
        "\\label{tab:main}\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrccccccc}\n\\toprule\n"
        "\\multirow{2}{*}{Method} & \\multirow{2}{*}{Params} & \\multirow{2}{*}{ms / frame} & "
        "\\multicolumn{2}{c}{Track 1: clean} & \\multicolumn{3}{c}{Track 2: streaming} & "
        "\\multirow{2}{*}{$\\Delta$ vs Lanczos} & \\multirow{2}{*}{Rank} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\n & & & PSNR$\\uparrow$ & SSIM$\\uparrow$ & PSNR$\\uparrow$ & SSIM$\\uparrow$ & VMAF$\\uparrow$ & & \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def codec_table(res: dict) -> str:
    names = list(METHODS)
    cols = []
    for c in CODECS:
        cols.append([res[c][n]["psnr"] for n in names])
        cols.append([res[c][n]["ssim"] for n in names])
    marked = [marks(c) for c in cols]
    rows = []
    for i, n in enumerate(names):
        label, _, _, ours = METHODS[n]
        pre = "\\rowcolor{gray!12} " if ours else ""
        rows.append(pre + " & ".join([label.replace("\\textbf{LiveActionSR} ", "")] + [m[i] for m in marked]) + " \\\\")
        if n == "lanczos":
            rows.append("\\midrule")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{CODEC_LABEL[c]}}}" for c in CODECS)
    crf = "x264 CRF 26, x265 CRF 28, SVT-AV1 CRF 38 (preset 8)"
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{\\textbf{{Track~2 by codec}} ({crf}). PSNR-Y / SSIM-Y.}}\n"
        "\\label{tab:codecs}\n\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\nMethod & " + head + " \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        " & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def ablation_table() -> str:
    rows = []
    for (model, data), (a, b) in ABLATION.items():
        bold = data.startswith("6")
        f = (lambda v: f"\\textbf{{+{v:.3f}}}") if bold else (lambda v: f"+{v:.3f}")
        rows.append(f"\\emph{{{model}}} & {data} & {f(a)} & {f(b)} \\\\")
        if model == "balanced" and bold:
            rows.append("\\midrule")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Data scale.} PSNR-Y gain over Lanczos (dB) on the two training-time validation "
        "sets (mixed clean and compressed). Set A: 840 pairs; set B: 1{,}024 pairs, adds a Netflix "
        "\\emph{Chimera} and a UVG-VCM scene.}\n"
        "\\label{tab:ablation}\n"
        "\\begin{tabular}{llcc}\n\\toprule\nModel & Clips / pairs & Val A & Val B \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def flat_table(flat: list) -> str:
    rows = []
    for b in flat:
        lo, hi = b["bin"]
        rng = f"$<{hi}$" if lo == 0 else f"$\\geq{lo}$" if hi >= 99 else f"{lo}--{hi}"
        cells = [rng, f"{b['n']:,}".replace(",", "{,}")] + [signed(b[k]) for k in ("balanced", "quality", "EfRLFN")]
        rows.append(" & ".join(cells) + " \\\\")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Track~1 by frame difficulty.} Mean PSNR-Y gain over Lanczos (dB) per frame, "
        "binned by the Lanczos PSNR of the frame. Low PSNR: textured, hard frames; high PSNR: flat, almost "
        "flawless ones.}\n\\label{tab:flat}\n"
        "\\begin{tabular}{lrccc}\n\\toprule\n"
        "Lanczos PSNR & Frames & \\emph{balanced} & \\emph{quality} & EfRLFN \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def macros(res: dict, meta: dict) -> str:
    lz_s = res["streaming"]["lanczos"]["psnr"]
    lz_c = res["clean"]["lanczos"]["psnr"]
    g = lambda track, n, base: signed(res[track][n]["psnr"] - base)  # noqa: E731
    gains = {c: res[c]["quality"]["psnr"] - res[c]["lanczos"]["psnr"] for c in CODECS}
    best = max(gains, key=gains.get)
    m = {
        "gainBalStream": g("streaming", "balanced", lz_s),
        "gainQualStream": g("streaming", "quality", lz_s),
        "gainEfStream": g("streaming", "EfRLFN", lz_s),
        "gainBalClean": g("clean", "balanced", lz_c),
        "gainQualClean": g("clean", "quality", lz_c),
        "gainEfClean": g("clean", "EfRLFN", lz_c),
        "nClips": str(meta["clips"]),
        "nFrames": f"{res['clean']['lanczos']['n']:,}".replace(",", "{,}"),
        "codecBest": CODEC_LABEL[best],
        "vmafQualStream": f"{res['streaming']['quality']['vmaf'] - res['streaming']['lanczos']['vmaf']:+.1f}",
        "vmafBalStream": f"{res['streaming']['balanced']['vmaf'] - res['streaming']['lanczos']['vmaf']:+.1f}",
    }
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def tradeoff_plot(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    lz = res["streaming"]["lanczos"]["psnr"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    pts = [
        ("FSR 1 (EASU+RCAS)", 0.39, None, 0, "#9e9e9e"),
        ("balanced", 2.95, res["streaming"]["balanced"]["psnr"] - lz, 12_340, "#d1495b"),
        ("quality", 9.90, res["streaming"]["quality"]["psnr"] - lz, 47_716, "#c03221"),
        ("EfRLFN", 123.4, res["streaming"]["EfRLFN"]["psnr"] - lz, 487_010, "#30638e"),
    ]
    ax.axhline(0, color="#777", lw=0.7, ls="--")
    ax.text(0.33, 0.015, "Lanczos", color="#555", fontsize=7)
    ax.axvspan(0.2, 12, color="#2a9d8f", alpha=0.08, lw=0)
    ax.text(0.03, 0.04, "12 ms budget", transform=ax.transAxes, color="#2a7d6f", fontsize=7)
    for name, ms, gain, params, color in pts:
        if gain is None:
            continue
        ax.scatter(ms, gain, s=20 + params / 2500, color=color, edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(name, (ms, gain), textcoords="offset points", xytext={"EfRLFN": (6, -12), "balanced": (-10, -13)}.get(name, (6, 6)), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 400)
    gains = [p[2] for p in pts if p[2] is not None]
    ax.set_ylim(min(0.0, min(gains)) - 0.1, max(gains) * 1.3)
    ax.set_xlabel("ms per frame, 1080p$\\to$4K (WebGPU, RTX 5080)")
    ax.set_ylabel("PSNR-Y gain over Lanczos (dB)")
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def visual_figure(frames_dir: Path, path: Path, picks: int = 3, crop: int = 96) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    panels = [("hr", "HR"), ("bicubic", "Bicubic"), ("lanczos", "Lanczos"), ("EfRLFN", "EfRLFN"),
              ("balanced", "Ours: balanced"), ("quality", "Ours: quality")]  # fmt: skip
    cands = []
    for f in sorted(frames_dir.glob("*.npz")):
        if f.stem.endswith("__clean"):
            continue
        d = np.load(f)
        hr = d["hr"].astype(np.float32)
        h, w = hr.shape
        best, pos = -1.0, (0, 0)
        for y in range(8, h - crop - 8, 16):  # crop where quality beats Lanczos the most
            for x in range(8, w - crop - 8, 16):
                sl = (slice(y, y + crop), slice(x, x + crop))
                gain = _psnr(d["quality"][sl], hr[sl]) - _psnr(d["lanczos"][sl], hr[sl])
                if gain > best:
                    best, pos = gain, (y, x)
        cands.append((best, f, pos))
    # one pick per source scene, the strongest crops first
    chosen, seen = [], set()
    for best, f, pos in sorted(cands, key=lambda c: -c[0]):
        scene = f.stem.split("__")[0]
        if scene in seen:
            continue
        seen.add(scene)
        chosen.append((f, pos))
        if len(chosen) == picks:
            break

    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, axes = plt.subplots(len(chosen), len(panels) + 1, figsize=(7.0, 1.25 * len(chosen) + 0.1))
    axes = np.atleast_2d(axes)
    for r, (f, (y, x)) in enumerate(chosen):
        d = np.load(f)
        hr = d["hr"].astype(np.float32)
        ax = axes[r, 0]
        ax.imshow(hr, cmap="gray", vmin=0, vmax=1)
        ax.add_patch(patches.Rectangle((x, y), crop, crop, fill=False, ec="#e63946", lw=1))
        scene, _, codec = f.stem.split("__")
        ax.set_title(f"{scene.replace('_', ' ')} ({codec})", fontsize=6)
        for c, (key, label) in enumerate(panels, start=1):
            img = d[key].astype(np.float32)[y : y + crop, x : x + crop]
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            sub = "reference" if key == "hr" else f"{_psnr(img, hr[y : y + crop, x : x + crop]):.2f} dB"
            ax.set_xlabel(sub, fontsize=6, labelpad=1)
            if r == 0:
                ax.set_title(label, fontsize=7)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    data = json.loads((Path(args.results) / "benchmark.json").read_text(encoding="utf-8"))
    res, meta = data["results"], data["meta"]
    tables = HERE / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "main.tex").write_text(main_table(res), encoding="utf-8")
    (tables / "codecs.tex").write_text(codec_table(res), encoding="utf-8")
    (tables / "ablation.tex").write_text(ablation_table(), encoding="utf-8")
    (tables / "macros.tex").write_text(macros(res, meta), encoding="utf-8")
    flat = Path(args.results) / "flatness.json"
    if flat.exists():
        (tables / "flat.tex").write_text(flat_table(json.loads(flat.read_text(encoding="utf-8"))), encoding="utf-8")
    figs = HERE / "figures"
    tradeoff_plot(res, figs / "tradeoff.pdf")
    print("tables and figures written")


if __name__ == "__main__":
    main()
