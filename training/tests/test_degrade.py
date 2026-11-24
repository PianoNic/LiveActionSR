import numpy as np
import pytest

from lasr import ffmpeg
from lasr.color import rgb_to_y
from lasr.degrade import degrade_clip, degrade_rgb, resize, sample_params

from .conftest import needs_codecs

CLEAN = {"blur_sigma": 0.0, "rescale": 0.0, "rescale_kernels": ["bicubic", "bicubic"], "kernel": "bicubic",
         "sharpen": None, "noise": None, "codec": None, "seed": 0}  # fmt: skip


def test_sampling_is_deterministic_and_respects_codecs():
    a = sample_params({}, np.random.default_rng(3), codecs=("x264",))
    b = sample_params({}, np.random.default_rng(3), codecs=("x264",))
    assert a == b
    for s in range(50):
        p = sample_params({}, np.random.default_rng(s), codecs=("x264",))
        assert p["codec"] is None or p["codec"]["name"] == "x264"
    assert all(sample_params({}, np.random.default_rng(s), codecs=())["codec"] is None for s in range(20))


def test_crf_ranges_from_config():
    cfg = {"codec": {"p": 1.0, "weights": {"x265": 1.0}, "crf": {"x265": [30, 31]}}}
    for s in range(20):
        p = sample_params(cfg, np.random.default_rng(s), codecs=("x264", "x265"))
        assert p["codec"]["name"] == "x265" and 30 <= p["codec"]["crf"] <= 31


@pytest.mark.parametrize("kernel", ["bicubic", "lanczos", "area", "bilinear", "bilinear_aliased"])
def test_resize_kernels_shape_and_range(kernel, hr_clip):
    out = resize(hr_clip[0], (32, 32), kernel)
    assert out.shape == (32, 32, 3)
    assert out.min() > -0.2 and out.max() < 1.2


def test_degrade_without_codec_shapes_and_ranges(hr_clip):
    params = sample_params({"codec": {"p": 0.0}}, np.random.default_rng(1), codecs=())
    lr, hr, info = degrade_clip(hr_clip, params)
    assert lr.shape == (3, 32, 32) and hr.shape == (3, 64, 64)
    assert lr.dtype == np.float32 and hr.dtype == np.float32
    assert 0.0 <= lr.min() and lr.max() <= 1.0 and 0.0 <= hr.min() and hr.max() <= 1.0
    assert info["codec"] == "none"


def test_clean_degradation_is_close_to_downscaled_hr(hr_clip):
    lr, hr, _ = degrade_clip(hr_clip, CLEAN)
    ref = rgb_to_y(resize(hr_clip[0], (32, 32), "bicubic"))
    assert np.abs(lr[0] - ref).max() < 3 / 255  # only 8-bit quantisation


def test_noise_and_sharpen_change_the_output(hr_clip):
    base, _ = degrade_rgb(hr_clip, CLEAN)
    noisy, _ = degrade_rgb(hr_clip, {**CLEAN, "noise": {"sigma": 0.03, "luma_only": False, "correlated": True}})
    sharp, _ = degrade_rgb(hr_clip, {**CLEAN, "sharpen": {"amount": 1.0, "sigma": 1.0}})
    assert np.abs(noisy.astype(int) - base).mean() > 1.0
    assert np.abs(sharp.astype(int) - base).mean() > 0.3


def test_rejects_sizes_not_divisible_by_4(hr_clip):
    with pytest.raises(ValueError):
        degrade_rgb(hr_clip[:, :62], CLEAN)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("codec", ["x264", "x265", "av1"])
def test_codec_roundtrip(codec, hr_clip):
    if codec not in ffmpeg.available_codecs():
        pytest.skip(f"ffmpeg has no {codec} encoder")
    preset = "8" if codec == "av1" else "veryfast"
    params = {**CLEAN, "codec": {"name": codec, "crf": 30 if codec != "av1" else 40, "preset": preset,
                                 "keyint": 12, "bframes": 2}}  # fmt: skip
    lr, hr, info = degrade_clip(hr_clip, params)
    clean, _, _ = degrade_clip(hr_clip, CLEAN)
    assert lr.shape == (3, 32, 32) and info["codec"] == codec and info["bpp"] > 0
    err = np.abs(lr - clean).mean()
    assert 0.0005 < err < 0.08  # damaged, but still the same picture (colour matrix is right)


@needs_codecs("x264")
@pytest.mark.ffmpeg
def test_codec_roundtrip_preserves_luma_of_flat_grey():
    grey = np.full((2, 64, 64, 3), 0.5, np.float32)
    out, _ = ffmpeg.codec_roundtrip(grey, "x264", 18, "veryfast")
    assert np.abs(out.astype(np.float32) / 255 - 0.5).max() < 2 / 255
