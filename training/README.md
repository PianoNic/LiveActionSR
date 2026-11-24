# Training: tiny 2x luma SR for live-action streams

This folder trains the networks behind the plugin's **balanced** and **quality** presets. They are
plain 3x3-conv stacks of about 12k and 48k parameters. They upscale **luma only**, 2x, and predict
a **residual over bilinear**. They are trained on **real codec damage** (H.264, HEVC and AV1
round-trips through ffmpeg) of permissively licensed live-action footage. Why it is built this way
is in [`../docs/research.md`](../docs/research.md). In short: every good real-time shader CNN is
anime-trained or LGPL, and at this size the training data matters more than the architecture.

```
data/sources.yaml ─► data/fetch.py ─► HR crops (16-bit PNG clips)
                                         │
     configs/*.yaml (degradation:) ─► data/bake.py ─► (LR, HR) luma shards (.npy, memory-mapped)
                                                            │
                                     train.py ─► runs/<name>/best.pt ─► export.py ─► ../models/<name>.json
                                                                       evaluate.py ─► side-by-sides + PSNR/SSIM/VMAF
```

## Layout

| Path | What |
|---|---|
| `lasr/models.py` | `LumaSR`: conv3x3(1→F) ReLU → [conv3x3(F→F) ReLU] x blocks → conv3x3(F→4) → depth_to_space(2) + bilinear 2x. Presets `balanced` (F=16, 5 blocks, **12,340 params**, ArtCNN C4F16 class) and `quality` (F=32, 5 blocks, **47,716 params**, C4F32 / ARSR-41k class). Optional `rep: true` trains RepConv blocks (3x3 + 1x1 + identity) that fold exactly into plain conv3x3 for export |
| `lasr/degrade.py` | The degradation recipe (below) |
| `lasr/ffmpeg.py` | Pipe-only ffmpeg wrappers: codec round-trip, frame decoding, probing |
| `lasr/sources.py` | Source readers (PNG sequences, ranged y4m over HTTP, remote ffmpeg seeks, still folders with synthetic pans), crop selection and dedup |
| `lasr/bake.py`, `lasr/data.py` | Multiprocess pre-baking into shards; random-patch and full-frame datasets |
| `lasr/validate.py`, `lasr/metrics.py` | PSNR-Y / SSIM-Y vs bilinear, bicubic and Lanczos, overall and per codec |
| `train.py` | L1 (+ optional FFT/edge), AdamW, warmup + cosine, bf16 autocast, CSV + TensorBoard, `last.pt` / `best.pt`, resume |
| `export.py` | Checkpoint → plugin JSON (+ ONNX), numpy reference forward pass, f16 error report |
| `evaluate.py` | Model vs Lanczos (and optionally a second model or an external tool) on a video or frames, side-by-sides + metrics |
| `configs/` | `base.yaml` (shared + degradation), `balanced.yaml`, `quality.yaml`, `smoke.yaml` |
| `runpod/` | `setup.sh`, `run.sh`, `Dockerfile` and a README with GPU choice, time and cost |
| `DATASETS.md` | Every source and its licence |

## The degradation recipe

Each HR clip (6 consecutive frames, 512² crop) is degraded **as a clip**, so P/B-frame damage is
included and not only intra-frame damage. All parameters are random per clip:

1. optional Gaussian blur (p 0.3, σ 0.3–1.1): lens or mastering softness;
2. optional extra down-and-up rescale pass (p 0.2, 0.55–0.9x): ladder or mastering scaling blur;
3. **2x downscale** with bicubic / Lanczos / area / bilinear (all antialiased) or aliased bilinear;
4. optional unsharp mask (p 0.15): sharpening halos;
5. optional noise or grain before encoding (p 0.25, σ 1–6/255, luma or RGB, white or correlated);
6. **codec round-trip** (p 0.95): x264 CRF 18–32 / x265 CRF 20–34 / SVT-AV1 CRF 24–46, random
   preset, GOP 12–96, 0–3 B-frames, **yuv420p BT.709 limited range**, decoded back to **8-bit RGB**
   like a browser video texture.

LR = BT.709 luma of the decoded RGB. HR = luma of the original. Everything is configurable in the
`degradation:` section of `configs/base.yaml`. Grain is kept out of the target on purpose: the
plugin re-adds synthetic grain after upscaling (docs/research.md, "Cleanup decides perceived
quality").

## Run locally

```bash
cd training
uv venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
uv pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest   # ~5 s, CPU only

# smoke run of the whole path (tiny real sample: ~100 MB download)
python data/fetch.py --out work/smoke/hr --only tum_blue_sky tears_of_steel tears_of_steel_val svt_old_town_cross --positions 2 --clip-len 4 --crop 256
python data/bake.py --hr work/smoke/hr --out work/smoke/baked --split train --config configs/base.yaml
python data/bake.py --hr work/smoke/hr --out work/smoke/baked --split val --variants 1 --config configs/base.yaml
python train.py --config configs/smoke.yaml
python export.py work/smoke/run/best.pt --out work/smoke/balanced.json --onnx work/smoke/balanced.onnx
# no network? python data/make_synthetic.py --out work/smoke/hr  (procedural clips)
```

ffmpeg with libx264/libx265/libsvtav1 is needed for codec damage (`ffmpeg -encoders`). Encoders
that are missing are skipped with a warning, and `bake.py --no-codec` runs without ffmpeg at all
(not useful for real models). The codec tests skip themselves when an encoder is missing.

A full run on a local GPU uses the same commands as `runpod/run.sh` (read it, or just run it with
`WORK=./work SKIP_SETUP=1 bash runpod/run.sh`).

## Run on RunPod

See [`runpod/README.md`](runpod/README.md). In short: an RTX 4090 pod with the PyTorch template
and an 80 GB volume, then `bash training/runpod/run.sh`. It takes about 1.5–2 h and costs about
$0.50–1.80 depending on community or secure cloud.

## Export into the plugin

```bash
python export.py runs/balanced/best.pt --out ../models/balanced.json --onnx runs/balanced/balanced.onnx
python export.py runs/quality/best.pt  --out ../models/quality.json  --onnx runs/quality/quality.onnx
```

The plugin loads its weights from **`models/balanced.json`** and **`models/quality.json` at the
repository root** (`runpod/run.sh` writes them there). `export.py` refuses to write a file whose
numpy re-implementation differs from PyTorch by more than 1e-4. It also prints the error of an f16
evaluation: about 1e-3 max abs on random input, i.e. well under one 8-bit code value.

### The JSON contract

```json
{ "arch": "artcnn-like", "scale": 2, "channels": 16, "residual": "bilinear",
  "layers": [
    { "type": "conv3x3", "in": 1,  "out": 16, "relu": true,  "weights": [...], "bias": [...] },
    ...
    { "type": "conv3x3", "in": 16, "out": 4,  "relu": false, "weights": [...], "bias": [...] },
    { "type": "depth_to_space", "scale": 2 } ] }
```

A shader implementation must match these exactly. All of it is verified by the numpy forward pass
in `export.py` and by `tests/test_export.py`:

- **Input**: one channel, `Y' = 0.2126 R' + 0.7152 G' + 0.0722 B'` of the video's non-linear RGB in
  [0, 1]. Not normalised, no offset.
- **conv3x3**: cross-correlation (PyTorch convention, no kernel flip). Weights are `[out][in][ky][kx]`
  row-major: index `((o * in + i) * 3 + ky) * 3 + kx`, and tap `(ky, kx)` reads pixel
  `(y + ky - 1, x + kx - 1)`. **Padding is replicate / clamp-to-edge at every layer.** That is what
  clamped `textureLoad` coordinates give. Zero padding would darken the frame borders.
- **relu**: `max(0, x)` when `"relu": true` (every layer except the last).
- **depth_to_space(2)**: `out[2y + i][2x + j] = in[channel i * 2 + j][y][x]` (PyTorch `pixel_shuffle`).
- **residual "bilinear"**: add a 2x bilinear upscale of the input, sampled at
  `((x + 0.5) / 2 - 0.5, (y + 0.5) / 2 - 0.5)` with clamp-to-edge. That is a linear-filtering
  sampler at output pixel centres, with weights 0.75 / 0.25.
- **Output**: clamp to [0, 1]. Combine it with bilinear-upscaled chroma.

## Results so far

| Run | Data | Iterations | val PSNR-Y gain vs Lanczos |
|---|---|---|---|
| balanced (local RTX 5080, 1.5 min) | 482 real clips (TUM, SVT, ToS), 3 variants | 20k | **+0.39 dB** (33.54 vs 33.14) |
| quality (local RTX 5080, 2 min) | same | 20k | **+0.45 dB** (33.59 vs 33.14); x264 +0.47, x265 +0.35, AV1 +0.51 |
| `evaluate.py`, TUM pedestrian_area (held out), x264 CRF 27 | — | — | quality +0.39 dB PSNR-Y, **+1.5 VMAF** (74.5 vs 73.0) |

These are pipeline checks on about 15 % of the default dataset, not release numbers. The full
RunPod run is the real result.
