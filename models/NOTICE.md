# Model weights

`balanced.json` (12,340 parameters) and `quality.json` (47,716 parameters) are 2x luma
super-resolution networks trained with the pipeline in [`training/`](../training/README.md) on
compressed (x264 / x265 / SVT-AV1) live-action footage. The weights are released under the MIT
licence of this repository.

**Status: preview.** These weights come from a small run (482 clips, 20k iterations). On held-out
compressed clips they beat Lanczos by about +0.4 dB PSNR-Y (quality: x264 +0.47, x265 +0.35,
AV1 +0.51 dB). A full run on the whole default dataset replaces them.

The training footage and its attribution:

- _Tears of Steel_ — (CC) Blender Foundation | mango.blender.org, CC BY 3.0
- _El Fuente_, _Sparks_, _Nocturne_ — Netflix Open Content, CC BY 4.0
- TU München 1080p test sequences — courtesy TU München, LDV (no restrictions)
- SVT MultiFormat test sequences — courtesy SVT Sveriges Television AB (no restrictions)

See [`training/DATASETS.md`](../training/DATASETS.md) for the sources and their terms.
