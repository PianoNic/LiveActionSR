# Model weights

`balanced.json` (12,340 parameters) and `quality.json` (47,716 parameters) are 2x luma
super-resolution networks trained with the pipeline in [`training/`](../training/README.md) on
compressed (x264 / x265 / SVT-AV1) live-action footage. The weights are released under the MIT
licence of this repository.

On held-out compressed live-action clips they beat Lanczos by +0.44 dB (balanced) and +0.52 dB
(quality) PSNR-Y; see [the training results](../training/README.md#results).

The training footage and its attribution:

- _Tears of Steel_ — (CC) Blender Foundation | mango.blender.org, CC BY 3.0
- _Chimera_, _El Fuente_, _Meridian_, _Nocturne_, _Sparks_ — Netflix Open Content, CC BY 4.0
- TU München 1080p test sequences — courtesy TU München, LDV (no restrictions)
- SVT MultiFormat test sequences — courtesy SVT Sveriges Television AB (no restrictions)
- UVG-VCM — Ultra Video Group, Tampere University, CC BY 4.0 (T. Partanen et al., "UVG-VCM:
  Benchmarking dataset for machine-oriented visual data compression", QoMEX 2026)

See [`training/DATASETS.md`](../training/DATASETS.md) for the sources and their terms.

## ultra.json (EfRLFN)

`ultra.json` (487,010 parameters, float16) is the 2x checkpoint of **EfRLFN**, converted with
[`tools/efrlfn/convert.py`](../tools/efrlfn/README.md) without retraining. It was not trained by
this project: it comes from <https://github.com/EvgeneyBogatyrev/EfRLFN> and is used under its MIT
licence:

```
MIT License

Copyright (c) 2026 MSU Graphics & Media Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

If you use it, please cite the paper:

```bibtex
@article{bogatyrev2026exploring,
  title={Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content},
  author={Bogatyrev, Evgeney and Abud, Khaled and Molodetskikh, Ivan and Alutis, Nikita and Vatolin, Dmitriy},
  journal={arXiv preprint arXiv:2602.11339},
  year={2026}
}
```

EfRLFN was trained and fine-tuned on the StreamSR dataset of its authors; see their repository for
the data and its terms.
