import json

import numpy as np
import pytest
import torch

from evaluate import model_from_json
from export import (
    bilinear_up2_np,
    depth_to_space_np,
    export_onnx,
    model_to_json,
    numpy_forward,
    validate_doc,
)
from export import main as export_main
from lasr.models import LumaSR, bilinear_up2, build_model


def _randomised(model):
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.eval()


@pytest.mark.parametrize("preset", ["balanced", "quality"])
def test_json_numpy_forward_matches_torch(preset):
    model = _randomised(build_model({"preset": preset}))
    doc = json.loads(json.dumps(model_to_json(model)))  # through real JSON text
    validate_doc(doc)
    y = np.random.default_rng(0).random((13, 17), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(y)[None, None])[0, 0].numpy()
    assert np.abs(numpy_forward(doc, y) - ref).max() < 1e-4


def test_json_format_exact():
    model = LumaSR(4, 1)
    doc = model_to_json(model)
    assert set(doc) == {"arch", "scale", "channels", "residual", "layers"}
    assert doc["arch"] == "artcnn-like" and doc["scale"] == 2 and doc["channels"] == 4
    assert doc["residual"] == "bilinear"
    assert [(layer["in"], layer["out"], layer["relu"]) for layer in doc["layers"][:-1]] == [
        (1, 4, True), (4, 4, True), (4, 4, False),
    ]  # fmt: skip
    assert doc["layers"][-1] == {"type": "depth_to_space", "scale": 2}
    for layer in doc["layers"][:-1]:
        assert set(layer) == {"type", "in", "out", "relu", "weights", "bias"}


def test_weight_layout_is_out_in_ky_kx():
    model = LumaSR(4, 1)
    w = model.convs[1].conv.weight  # (out=4, in=4, 3, 3)
    with torch.no_grad():
        w.zero_()
        w[2, 1, 0, 2] = 7.0
    flat = model_to_json(model)["layers"][1]["weights"]
    assert flat.index(7.0) == ((2 * 4 + 1) * 3 + 0) * 3 + 2


def test_depth_to_space_matches_pixel_shuffle():
    x = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    ref = torch.nn.functional.pixel_shuffle(torch.from_numpy(x)[None], 2)[0].numpy()
    assert np.array_equal(depth_to_space_np(x, 2), ref)


def test_bilinear_np_matches_torch():
    y = np.random.default_rng(2).random((7, 9), dtype=np.float32)
    ref = bilinear_up2(torch.from_numpy(y)[None, None])[0, 0].numpy()
    assert np.abs(bilinear_up2_np(y) - ref).max() < 1e-6


def test_rep_model_exports_fused():
    model = _randomised(LumaSR(8, 2, rep=True))
    doc = model_to_json(model)
    y = np.random.default_rng(3).random((10, 10), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(y)[None, None])[0, 0].numpy()
    assert np.abs(numpy_forward(doc, y) - ref).max() < 1e-4


def test_validate_doc_rejects_broken_files():
    doc = model_to_json(LumaSR(4, 1))
    bad = json.loads(json.dumps(doc))
    bad["layers"][1]["in"] = 3
    with pytest.raises(ValueError):
        validate_doc(bad)
    bad = json.loads(json.dumps(doc))
    bad["layers"][0]["weights"].pop()
    with pytest.raises(ValueError):
        validate_doc(bad)


def test_json_roundtrip_back_into_torch():
    model = _randomised(build_model({"preset": "balanced"}))
    again = model_from_json(json.loads(json.dumps(model_to_json(model))))
    x = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        assert torch.allclose(model(x), again(x), atol=1e-6)


def test_export_cli_and_onnx(tmp_path):
    model = _randomised(build_model({"preset": "balanced"}))
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "model_cfg": {"channels": 16, "blocks": 5, "scale": 2, "rep": False}},
               ckpt)  # fmt: skip
    out = tmp_path / "models" / "balanced.json"
    export_main([str(ckpt), "--out", str(out)])
    doc = json.loads(out.read_text())
    validate_doc(doc)
    assert out.read_bytes().endswith(b"\n") and b"\r\n" not in out.read_bytes()

    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    onnx_path = tmp_path / "m.onnx"
    export_onnx(model, onnx_path)
    y = np.random.default_rng(4).random((1, 1, 20, 24), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"luma": y})[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(y)).numpy()
    assert got.shape == (1, 1, 40, 48)
    assert np.abs(got - ref).max() < 1e-4
