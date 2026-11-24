"""Source readers added for the extended dataset: raw planar YUV (with row bands), zipped frame
sequences, HDR tone mapping, tier selection, and a sanity check of data/sources.yaml."""

import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lasr import ffmpeg
from lasr.imio import decode_bytes
from lasr.sources import (
    TIERS,
    RawLayout,
    SourceSpec,
    iter_source_clips,
    j2k_size,
    load_sources,
    read_raw_clip,
    read_zip_member,
    select_sources,
    zip_members,
)

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.ffmpeg_available(), reason="ffmpeg missing")
SOURCES_YAML = Path(__file__).resolve().parents[1] / "data" / "sources.yaml"


def _spec(id, tier="permissive", enabled=True):
    return SourceSpec(id=id, type="y4m", tier=tier, enabled=enabled)


def test_select_sources_tiers():
    specs = [_spec("a"), _spec("b", "sharealike"), _spec("c", "research"), _spec("d", enabled=False)]
    ids = lambda xs: [s.id for s in xs]  # noqa: E731
    assert ids(select_sources(specs)) == ["a"]
    assert ids(select_sources(specs, include_sharealike=True)) == ["a", "b"]
    assert ids(select_sources(specs, include_research=True)) == ["a", "b", "c"]
    assert ids(select_sources(specs, only=["d", "c"])) == ["c", "d"]  # --only ignores enabled/tier
    with pytest.raises(KeyError):
        select_sources(specs, only=["nope"])
    with pytest.raises(ValueError):
        select_sources([_spec("x", "free-ish")])


def _planes(layout: RawLayout, rng, frames: int) -> list[list[np.ndarray]]:
    dt = np.uint8 if layout.planes()[0][0] == layout.width else np.dtype("<u2")
    out = []
    for _ in range(frames):
        ps = []
        for rb, rows, _div in layout.planes():
            cols = rb // np.dtype(dt).itemsize
            ps.append(rng.integers(0, 1023 if dt != np.uint8 else 255, (rows, cols)).astype(dt))
        out.append(ps)
    return out


@pytest.mark.parametrize("pix_fmt", ["yuv420p10le", "yuv422p", "yuv444p16le"])
def test_raw_band_ranges_select_the_right_rows(tmp_path, pix_fmt):
    layout = RawLayout(24, 16, pix_fmt)
    frames = _planes(layout, np.random.default_rng(0), 3)
    blob = b"".join(p.tobytes() for f in frames for p in f)
    assert len(blob) == 3 * layout.frame_size
    row0, rows = 4, 8
    for fi in range(3):
        got = [blob[a : b + 1] for a, b in layout.band_ranges(fi, row0, rows)]
        for plane, (_rb, _n, div), g in zip(frames[fi], layout.planes(), got):
            assert g == plane[row0 // div : (row0 + rows) // div].tobytes()
    if layout.row_align == 2:
        with pytest.raises(ValueError):
            layout.band_ranges(0, 3, 8)  # 4:2:0 bands must start on an even row
    with pytest.raises(ValueError):
        layout.band_ranges(0, 12, 8)  # past the bottom


@needs_ffmpeg
def test_raw_yuv_band_matches_full_frame(tmp_path):
    layout = RawLayout(64, 48, "yuv420p10le")
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:48, 0:64]
    frames = []
    for t in range(3):
        y = (64 + (xx * 9 + yy * 5 + t * 7) % 876).astype("<u2")
        u = (512 + rng.integers(-40, 40, (24, 32))).astype("<u2")
        v = (512 + rng.integers(-40, 40, (24, 32))).astype("<u2")
        frames.append(y.tobytes() + u.tobytes() + v.tobytes())
    path = tmp_path / "clip.yuv"
    path.write_bytes(b"".join(frames))
    full = read_raw_clip(str(path), layout, 1, 2)
    band = read_raw_clip(str(path), layout, 1, 2, band=(16, 24))
    assert full.shape == (2, 48, 64, 3) and band.shape == (2, 24, 64, 3)
    # bicubic chroma upsampling reads neighbouring chroma rows, so compare away from the band edges
    np.testing.assert_allclose(band[:, 6:-6], full[:, 22:34], atol=1e-6)

    spec = SourceSpec(id="raw", type="raw_yuv", positions=2, crops_per_position=1,
                      options={"url": str(path), "width": 64, "height": 48, "pix_fmt": "yuv420p10le",
                               "band_rows": 33})  # fmt: skip
    clips = list(iter_source_clips(spec, 1, np.random.default_rng(0)))
    assert len(clips) == 2
    assert clips[0][1]().shape == (1, 32, 64, 3)  # band rounded down to a multiple of 2


def _png_bytes(rgb8: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", rgb8[..., ::-1])
    assert ok
    return buf.tobytes()


def test_zip_members_and_member_reads(tmp_path):
    rng = np.random.default_rng(0)
    imgs = {f"shot/frame{n:08d}.png": rng.integers(0, 255, (8, 10, 3), dtype=np.uint8) for n in range(5)}
    path = tmp_path / "shot.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("shot/", b"")
        zf.writestr("shot/readme.txt", b"not a frame")
        for i, name in enumerate([*imgs][::-1]):  # stored out of order, mixed compression
            method = zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED
            zf.writestr(zipfile.ZipInfo(name), _png_bytes(imgs[name]), compress_type=method)
    members = zip_members(str(path), ["*.png"])
    assert [m.name for m in members] == sorted(imgs)
    assert {m.method for m in members} == {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
    for m in members:
        np.testing.assert_allclose(decode_bytes(read_zip_member(str(path), m)), imgs[m.name] / 255.0, atol=1e-6)

    # a local extra field longer than the read-ahead slack forces the second request
    big = tmp_path / "bigextra.zip"
    payload = _png_bytes(imgs[sorted(imgs)[0]])
    info = zipfile.ZipInfo("frame.png")
    info.extra = struct.pack("<HH", 0xCAFE, 700) + bytes(700)
    with zipfile.ZipFile(big, "w") as zf:
        zf.writestr(info, payload)
    (m,) = zip_members(str(big), ["*.png"])
    assert read_zip_member(str(big), m) == payload

    spec = SourceSpec(id="z", type="zip_sequence", positions=2, crops_per_position=1,
                      options={"url": str(path), "patterns": ["*.png"]})  # fmt: skip
    clips = list(iter_source_clips(spec, 3, np.random.default_rng(0)))
    frames = clips[0][1]()
    assert len(clips) == 2 and frames.shape == (3, 8, 10, 3)
    first = int(np.argmin([np.abs(frames[0] - v / 255.0).max() for v in imgs.values()]))
    np.testing.assert_allclose(frames[1], list(imgs.values())[first + 1] / 255.0, atol=1e-6)  # consecutive


@needs_ffmpeg
def test_zip_sequence_jpeg2000_ycbcr(tmp_path):
    """The Netflix SDR shots are 10-bit 4:2:0 JPEG 2000 codestreams: decoded through ffmpeg."""
    out = tmp_path / "j2k"
    out.mkdir()
    try:
        ffmpeg._run([ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     "testsrc2=size=96x64:rate=24", "-frames:v", "3", "-pix_fmt", "yuv420p10le",
                     "-c:v", "jpeg2000", "-pred", "dwt53", str(out / "frame%08d.j2k")])  # fmt: skip
    except ffmpeg.FFmpegError as e:
        pytest.skip(f"no jpeg2000 encoder: {e}")
    path = tmp_path / "shot.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for f in sorted(out.glob("*.j2k")):
            zf.write(f, f"SDR/shot/{f.name}")
    first = (out / "frame00000001.j2k").read_bytes()
    assert j2k_size(first) == (96, 64)
    spec = SourceSpec(id="j", type="zip_sequence", positions=1, options={"url": str(path)})
    frames = next(iter_source_clips(spec, 3, np.random.default_rng(0)))[1]()
    assert frames.shape == (3, 64, 96, 3)
    assert 0.0 <= frames.min() and frames.max() <= 1.0 and frames.std() > 0.1


def _pq_encode(nits: np.ndarray) -> np.ndarray:
    m1, m2, c1, c2, c3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    y = (np.asarray(nits, np.float64) / 10000.0) ** m1
    return ((c1 + c2 * y) / (1 + c3 * y)) ** m2


@pytest.mark.skipif(not ffmpeg.has_filter("zscale"), reason="ffmpeg without zscale")
def test_tonemap_pq_p3d65_grey_ramp():
    nits = np.array([0.0, 1.0, 10.0, 50.0, 100.0, 203.0, 1000.0, 4000.0])
    pq = np.repeat(_pq_encode(nits)[:, None], 16, axis=1)  # 8 grey rows of 16 px
    frames = np.repeat(pq[None, :, :, None], 3, axis=-1).astype(np.float32)
    sdr = ffmpeg.tonemap_pq_to_sdr(np.repeat(frames, 2, axis=0))
    assert sdr.shape == (2, 8, 16, 3)
    grey = sdr[0, :, 8]
    np.testing.assert_allclose(grey[:, 0], grey[:, 1], atol=2e-3)  # neutral stays neutral
    np.testing.assert_allclose(grey[:, 0], grey[:, 2], atol=2e-3)
    y = grey[:, 1]
    assert np.all(np.diff(y) > 0)  # monotonic
    assert y[0] < 0.01 and y[-1] > 0.95 and y[-1] <= 1.0
    # pins the curve that was validated against Netflix's Dolby Vision SDR trim of Meridian
    # (34-36 dB PSNR-Y): 100 nits ~ 0.72, 203 nits (reference white) ~ 0.84, highlights roll off
    np.testing.assert_allclose(y[3:6], [0.558, 0.719, 0.843], atol=0.03)
    with pytest.raises(ValueError):
        ffmpeg.tonemap_pq_to_sdr(frames, "hlg_bt2020")


def test_sources_yaml_is_consistent():
    specs = load_sources(SOURCES_YAML)
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids)), "duplicate source ids"
    readers = {"png_sequence", "y4m", "video", "raw_yuv", "zip_sequence", "image_dir"}
    for s in specs:
        assert s.tier in TIERS, s.id
        assert s.split in ("train", "val"), s.id
        assert s.type in readers, s.id
        assert s.license, f"{s.id}: licence missing"
        assert 0 < s.hr_scale[0] <= s.hr_scale[1] <= 1.0, s.id
        if "tonemap" in s.options:
            assert s.options["tonemap"] in ffmpeg.TONEMAPS, s.id
        if s.type == "raw_yuv":
            RawLayout(int(s.options["width"]), int(s.options["height"]), s.options["pix_fmt"])
    # val is held out by sequence: no media file is read by both a train and a val source (Tears of
    # Steel is one file sequence split by disjoint frame ranges, which is checked separately)
    urls = {}
    for s in specs:
        u = s.options.get("url") or s.options.get("path")
        if s.type != "png_sequence":
            urls.setdefault(u, set()).add(s.split)
    leaks = [u for u, splits in urls.items() if len(splits) > 1]
    assert not leaks, f"used by both train and val: {leaks}"
    tos = {s.split: s for s in specs if s.type == "png_sequence" and "tearsofsteel" in s.options["url"]}
    assert tos["train"].options["last"] < tos["val"].options["first"]
    # every default-tier source must be enabled or deliberately disabled, and there must be val data
    assert any(s.split == "val" and s.tier == "permissive" and s.enabled for s in specs)
