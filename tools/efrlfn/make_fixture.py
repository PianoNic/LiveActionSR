"""Write test/fixtures/efrlfn-tiny.json: a narrow EfRLFN with random weights, exported by
convert.py, with an input frame and the output of PyTorch and of `numpy_forward`. The TypeScript
tests run `runGraphReference` on it, which ties the plugin's CPU reference to PyTorch.

    python make_fixture.py ../../test/fixtures/efrlfn-tiny.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import EfRLFN, model_to_graph, numpy_forward, torch_forward  # noqa: E402


def main(out: str) -> None:
    torch.manual_seed(0)
    model = EfRLFN(feat=8, upscale=2).eval()
    # The default init leaves the attention near 0.5 and the residuals tiny; widen it so every
    # path (tanh saturation, attention, both skips) moves the output.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.5)
    graph = model_to_graph(model, "float32")
    graph = json.loads(json.dumps(graph))
    width, height = 7, 5
    image = np.random.default_rng(1).random((3, height, width), dtype=np.float32)
    expected = numpy_forward(graph, image)
    reference = torch_forward(model, image)
    err = float(np.abs(expected - reference).max())
    assert err < 1e-5, err
    fixture = {
        "about": "tools/efrlfn/make_fixture.py: tiny random EfRLFN, numpy_forward output (PyTorch agrees "
        f"to {err:.1e})",
        "width": width,
        "height": height,
        "input": [float(f"{v:.9g}") for v in image.flatten().tolist()],
        "output": [float(f"{v:.9g}") for v in expected.astype(np.float32).flatten().tolist()],
        "model": graph,
    }
    Path(out).write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    print(f"{out}: max |numpy - torch| {err:.1e}, output range {expected.min():.3f}..{expected.max():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
