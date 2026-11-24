# Training on RunPod

One GPU pod runs the whole pipeline (fetch → bake → train balanced + quality → export) with one
command. Every step resumes, so a restarted pod continues where it stopped.

## Recommended pod

- **GPU: RTX 4090** (24 GB). RunPod list price when this was written (Nov 2026, from the GPU
  catalog): **$0.34/h community, $0.89/h secure**. Any Ampere/Ada/Blackwell card works. These
  models are tiny, so the GPU is not the bottleneck (see below). An **RTX A5000** ($0.16 / $0.27)
  or **A40** ($0.35 / $0.59) roughly doubles the training phase but costs about the same in total.
  An **L4** ($0.44 / $0.69) has low memory bandwidth and is the slowest choice here.
- **vCPUs: as many as the offer has (≥ 16 preferred).** Degradation baking (ffmpeg encodes) and
  the training data loader are CPU work.
- **Disk:** a **≥ 80 GB volume at `/workspace`**. That holds about 20 GB of HR PNG crops, about
  13 GB of baked shards, runs and checkpoints, and leaves headroom.
- **Template:** RunPod's official **PyTorch** template (any image with torch ≥ 2.4 and CUDA 12.x).
  `setup.sh` installs a static ffmpeg build with libx264/libx265/libsvtav1/libvmaf if the image's
  ffmpeg lacks them, then the pip requirements. Or build [`Dockerfile`](Dockerfile), which bakes all
  of that into an image.

## Run

```bash
cd /workspace
git clone <this repo> sr && cd sr
bash training/runpod/run.sh            # ~1.5-2 h on an RTX 4090 pod, see below
# models land in /workspace/sr/models/balanced.json and models/quality.json
```

Knobs (environment variables): `WORK` (default `/workspace/lasr`), `SCALE_POSITIONS` (dataset
size multiplier, e.g. `0.25` for a cheap trial), `VARIANTS` (degradations per clip, default 4),
`BAL_ITERS` / `QUAL_ITERS` (200k / 300k), `PARALLEL=1` (train both models at once), `WORKERS`,
`LOADER_WORKERS`, and `INCLUDE_RESEARCH=1` (adds research-only data, see `../DATASETS.md`).

Watch progress with `tail -f /workspace/lasr/train_*.log`, or with
`tensorboard --logdir /workspace/lasr/runs` (expose port 6006). Copy the two JSON files plus
`runs/*/val_final.json` back with `runpodctl send` or the file browser, then stop the pod.

## Time and cost estimate (RTX 4090 pod, 16 vCPU)

| Phase | Work | Estimate | Basis |
|---|---|---|---|
| setup | static ffmpeg + pip | 2–5 min | |
| fetch | ~3,000 clips x 6 frames; **~32 GB transferred** (ToS PNG 8.8 GB, El Fuente 4K y4m 12.7 GB, Sparks/Nocturne ProRes ~6 GB, TUM/SVT ~4 GB) | **20–40 min** | Measured locally: 514 clips / 3.1 GB in 3m44 on a home link (~14 MB/s, mostly Xiph). A datacenter pod with 16 download threads should reach 30–60 MB/s. Xiph's server is the slowest link |
| bake | ~2,800 train clips x 4 variants = ~11k codec round-trips (+ val) | **~10 min** | Measured: 1,446 round-trips of 512² / 6-frame clips in 32 s on 24 desktop processes ≈ 0.5 CPU-s each. Server vCPUs are about 1.5x slower: 11k x 0.75 s / 16 ≈ 9 min |
| train balanced | 200k it x batch 64 x 64² LR patches (12.3k params) | **~15–20 min** | Measured end to end on an RTX 5080 (similar memory bandwidth to a 4090): **225 it/s** including the data loader (277 it/s pure GPU). 200k / 225 ≈ 15 min, plus about 20 validations of a few seconds each |
| train quality | 300k it x batch 64 x 64² (47.7k params) | **~30–35 min** | Measured: **180 it/s** end to end (214 it/s pure GPU, bf16). 300k / 180 ≈ 28 min |
| export | JSON + ONNX + self-check | < 1 min | |
| **total** | | **~1.5–2 h** | |

**Cost: about $0.50–0.70 on community cloud or $1.40–1.80 on secure cloud**, plus the volume
while it exists. Delete the volume when you are done.

Why it is this fast, and why the iteration counts are what they are: one training step does
64 × 4,096 LR pixels × ~12.3k MAC × 3 (forward + backward) ≈ 10 GMAC for balanced and ≈ 39 GMAC
for quality. Even at a few percent of a 4090's peak (narrow 16/32-channel convolutions are bound by
memory and kernel launches, not FLOPs), that is a few milliseconds per step. Tiny SR nets in this
class (ECBSR, QuickSRNet, ArtCNN) typically converge in 100k–300k iterations at batch 32–64. In
practice, data diversity limits these models more than step count does: a 482-clip local sample
plateaued after ~10k iterations. So if you have budget, scale the dataset (`SCALE_POSITIONS`,
`--include-research` for experiments) before you raise the iteration count.

If the pod has few vCPUs (< 10), the data loader caps training at roughly 120–150 it/s. Expect the
training phases to take about 1.5x as long in that case.

## Checking the result

`runs/<model>/val_final.json` holds PSNR-Y and SSIM-Y on the held-out compressed val set for the
model, bilinear, bicubic and Lanczos, broken down per codec. `gain_vs_lanczos_db` has to be clearly
positive. A local 20k-iteration trial on a 482-clip sample already gave balanced **+0.39 dB** and
quality **+0.45 dB** over Lanczos (all four codec groups positive). On a held-out real clip
(TUM pedestrian_area, x264 CRF 27) it gave **+0.39 dB PSNR-Y and +1.5 VMAF**. The full dataset
should do better. Use `evaluate.py` for side-by-side images on real streams.
