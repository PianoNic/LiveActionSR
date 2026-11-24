# EfRLFN converter

Turns an [EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN) checkpoint into the plugin's graph
weights format (`models/ultra.json`, the `ultra` preset) and proves the conversion.

EfRLFN is the real-time super-resolution network of Bogatyrev et al., _Exploring Real-Time
Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content_ (ICLR 2026,
[arXiv 2602.11339](https://arxiv.org/abs/2602.11339)), MIT licence, Copyright (c) 2026 MSU Graphics
& Media Lab. It was trained and fine-tuned on compressed streaming video (StreamSR).

## The network

RGB in, RGB out at 2x, values 0..1, no input normalisation and no bilinear residual:

```
conv_1      3x3, 3 -> 52                                         (feature)
6 x ERLFB   x -> tanh(c1_r 3x3) -> tanh(c2_r 3x3) -> tanh(c3_r 3x3)
              -> + x -> c5 1x1 -> ECA                             (each 52 -> 52)
            ECA: global average pool -> Conv1d(1, 1, 3, no bias) across channels -> sigmoid
                 -> multiply the 52 channels
conv_2      3x3 on (block_6 + feature)
upsampler   3x3, 52 -> 12, PixelShuffle(2)
```

All convolutions are PyTorch's default zero padding. 487,010 parameters, about 486k
multiply-accumulates per source pixel (10x the `quality` network). The 2x checkpoint has no ESA
branch, so the format needs no strided convolution, pooling window or upsampling.

## Use

```bash
# The 2x weights are linked from the EfRLFN README (Google Drive id 1VeoW94hN1X-8kxGXQSyR53YzRqF1htKQ).
python convert.py EfRLFN-2x-model.pt -o ../../models/ultra.json --image some-frame.png
npx prettier --write ../../models/ultra.json
```

Requirements: `torch`, `numpy`, `pillow` (see `requirements.txt`). The checkpoint is a plain state
dict and is loaded with `torch.load(weights_only=True)`; nothing from the EfRLFN repository is
imported: the model is re-implemented in `convert.py` and was checked to be bit-identical to
upstream's `code/model.py` on the same weights.

`--dtype float16` (default) stores the convolution weights as base64 float16 (1.33 MB);
`--dtype float32` doubles that. Storing as float16 changes the output by at most 2.5e-4 (82 dB
PSNR against the float32 weights): below what the GPU's half-precision path rounds anyway.

## Verification

Every run checks the written file, on a random 56x40 input and on each `--image`:

- `numpy_forward` (a numpy implementation of the graph format, the spec in code) against PyTorch
  with the weights exactly as stored: must agree within 1e-4 (they do to about 1.3e-6);
- PyTorch with the stored weights against the original float32 weights: reported.

`--reference IMAGE OUT` also writes `numpy_forward(IMAGE)` as raw float32 `[3, 2H, 2W]`, which
`demo/model-check.html?frame=...&ref=...` compares with the GPU's output.

`make_fixture.py` writes `test/fixtures/efrlfn-tiny.json`: a narrow (8-channel) EfRLFN with random
weights, an input and its PyTorch/numpy output. The TypeScript tests run `runGraphReference` on it,
which ties the plugin's CPU reference to PyTorch.
