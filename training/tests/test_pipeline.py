"""End to end on tiny synthetic data: HR clips -> bake -> dataset -> 3 training steps -> export."""

import json

import numpy as np
import torch

from export import main as export_main
from export import validate_doc
from lasr.bake import bake
from lasr.config import load_config
from lasr.data import PairShards, PatchDataset
from lasr.metrics import psnr, ssim
from lasr.sources import dhash, hamming, list_clips, parse_y4m_header
from lasr.synthetic import make_synthetic_clips
from train import main as train_main


def test_metrics_sanity():
    x = torch.rand(2, 1, 32, 32)
    assert torch.all(psnr(x, x) > 90)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-4)
    assert torch.all(psnr(x, (x + 0.1).clamp(0, 1)) < 30)


def test_dhash_dedup():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32)
    assert hamming(dhash(a), dhash(a + 0.001)) <= 2
    assert hamming(dhash(a), dhash(rng.random((64, 64)).astype(np.float32))) > 10


def test_y4m_header_parsing():
    line = b"YUV4MPEG2 W4096 H2160 F60:1 Ip A1:1 C420p10\n"
    frame = 4096 * 2160 * 3 // 2 * 2 + 6
    info = parse_y4m_header(line + b"FRAME\n" + b"\0" * 64, len(line) + 3 * frame)
    assert (info.width, info.height, info.frame_size, info.frames) == (4096, 2160, frame, 3)


def test_configs_load():
    for name in ("balanced", "quality", "smoke"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg["model"]["preset"] in ("balanced", "quality")
        assert cfg["degradation"]["codec"]["weights"]
    cfg = load_config("configs/quality.yaml", ["train.iterations=7", "loss.fft=0.05"])
    assert cfg["train"]["iterations"] == 7 and cfg["loss"]["fft"] == 0.05
    assert cfg["train"]["batch"] == 64  # inherited from base.yaml


def test_bake_train_export(tmp_path):
    hr = tmp_path / "hr"
    make_synthetic_clips(hr, "train", 3, crop=64, clip_len=3, seed=1)
    make_synthetic_clips(hr, "val", 2, crop=64, clip_len=3, seed=2)
    assert len(list_clips(hr, "train")) == 3
    baked = tmp_path / "baked"
    n = bake(hr, baked, "train", variants=2, keep_frames=2, deg_cfg={}, workers=1, shard_size=5, codecs=())
    assert n == 3 * 2 * 2
    bake(hr, baked, "val", variants=1, keep_frames=0, deg_cfg={}, workers=1, codecs=())

    shards = PairShards(baked, "train")
    assert len(shards) == 12 and len(shards.stems) == 3  # 5 + 5 + 2
    lr, hr_arr = shards.pair(11)
    assert lr.shape == (32, 32) and lr.dtype == np.uint16
    assert hr_arr.shape == (64, 64) and hr_arr.dtype == np.uint8
    assert shards.meta(0)["codec"] == "none"
    a, b = PatchDataset(shards, patch=16, length=4)[0]
    assert a.shape == (1, 16, 16) and b.shape == (1, 32, 32)
    assert 0 <= a.min() and a.max() <= 1

    run = tmp_path / "run"
    res = train_main(["--config", "configs/smoke.yaml", "--set", f"data.root={baked.as_posix()}",
                      f"log.out_dir={run.as_posix()}", "train.iterations=3", "train.batch=2",
                      "data.patch=16", "train.device=cpu"])  # fmt: skip
    assert res["iteration"] == 3
    assert (run / "last.pt").exists() and (run / "best.pt").exists()
    assert (run / "train.csv").exists() and (run / "val.csv").exists()
    val = json.loads((run / "val_latest.json").read_text())
    assert {"model", "bilinear", "bicubic", "lanczos", "gain_vs_lanczos_db"} <= set(val)

    out = tmp_path / "balanced.json"
    export_main([str(run / "best.pt"), "--out", str(out)])
    validate_doc(json.loads(out.read_text()))
