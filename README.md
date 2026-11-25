<h1 align="center">LiveActionSR</h1>

<p align="center">
  <strong>Tiny real-time 2x super-resolution for streamed films and TV, trained on real codec damage.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/PianoNic/LiveActionSR-2x"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-LiveActionSR--2x-0d1117" alt="Model" /></a>
  <a href="https://huggingface.co/PianoNic/EfRLFN-2x-WebGPU"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-EfRLFN--2x--WebGPU-0d1117" alt="EfRLFN" /></a>
  <a href="https://huggingface.co/datasets/PianoNic/LiveActionSR-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-LiveActionSR--Data-0d1117" alt="Dataset" /></a>
  <a href="https://github.com/PianoNic/ArtPlayerPluginLiveActionSR"><img src="https://img.shields.io/badge/Plugin-ArtPlayerPluginLiveActionSR-0d1117" alt="Plugin" /></a>
  <img src="https://img.shields.io/github/license/PianoNic/LiveActionSR?color=0d1117" alt="MIT" />
</p>

---

This repository is the **model side** of [ArtPlayerPluginLiveActionSR](https://github.com/PianoNic/ArtPlayerPluginLiveActionSR), which upscales video in the browser with WebGPU. It holds:

- the training pipeline;
- the data sources and their licences;
- the converter for the `ultra` model;
- the research notes;
- the paper.

The plugin repository only holds the runtime: WGSL shaders, the player integration and the shipped weights.

| | Params | Gain over Lanczos (PSNR-Y, compressed val) | 1080p→4K, RTX 5080 |
|---|---|---|---|
| **LiveActionSR-2x `balanced`** | 12,340 | +0.44 dB | 2.95 ms |
| **LiveActionSR-2x `quality`** | 47,716 | +0.52 dB | 9.90 ms |
| EfRLFN 2x (Bogatyrev et al., ICLR 2026), converted | 487,010 | — | 123 ms |

## Why

Real-time shader upscalers exist, but the good CNNs are anime-trained (ArtCNN, CuNNy, Anime4K) and the content-neutral ones (RAVU, nnedi3, FSRCNNX) are LGPL-3.0. Live-action streams need something else:

- **The damage:** the problem is codec damage (banding, smeared texture, blocking), not soft line art.
- **The budget:** a model has to fit a few milliseconds per frame on an integrated GPU.

So these networks are tiny 3x3-conv stacks:

- they upscale luma only;
- they predict a residual over bilinear;
- they are trained on permissively licensed footage pushed through real x264 / x265 / SVT-AV1 round-trips.

[`docs/research.md`](docs/research.md) has the full survey of models, shaders, licences and budgets.

## Layout

| Path | What |
|---|---|
| [`training/`](training/README.md) | Fetch → degrade (bake) → train → export, with configs, tests and a RunPod / local runner |
| [`training/DATASETS.md`](training/DATASETS.md) | Every source with its licence and attribution, plus the rejected ones |
| [`tools/efrlfn/`](tools/efrlfn/README.md) | EfRLFN checkpoint → the plugin's graph weights format, with a numerical check |
| [`models/`](models/NOTICE.md) | The exported weights (mirrored on Hugging Face and in the plugin) |
| [`docs/research.md`](docs/research.md) | The research behind the design |
| [`paper/`](paper/LiveActionSR.pdf) | The paper ([PDF](paper/LiveActionSR.pdf)), its benchmark and figure scripts; CI rebuilds it |

## Quick start

```bash
cd training
python -m venv .venv && . .venv/bin/activate
pip install torch && pip install -r requirements.txt
SKIP_SETUP=1 bash runpod/run.sh   # fetch, bake, train balanced + quality, export to ../models
```

[`training/README.md`](training/README.md) covers the details: smaller trial runs, RunPod, evaluation against Lanczos and other models, and the JSON contract.

## Licence

The code and the trained weights are MIT. The training footage keeps its own licences; see [`models/NOTICE.md`](models/NOTICE.md) and [`training/DATASETS.md`](training/DATASETS.md).
