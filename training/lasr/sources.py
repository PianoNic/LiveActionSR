"""Reading short HR clips from the dataset sources listed in data/sources.yaml and cutting them into
deduplicated crops.

Source types:

* ``png_sequence``: a numbered frame sequence over HTTP(S) or on disk (Tears of Steel's lossless
  1080p PNG master on media.xiph.org). Only the sampled frames are downloaded.
* ``y4m``: an uncompressed YUV4MPEG2 file over HTTP(S) or on disk (Xiph "derf" collection, Netflix
  El Fuente). Frames are fixed-size, so sampled frames are fetched with HTTP range requests
  instead of downloading multi-gigabyte files.
* ``video``: anything ffmpeg can open, including a remote URL (Netflix Sparks/Nocturne SDR ProRes
  masters). ffmpeg seeks with range requests, so again only the sampled frames are transferred.
* ``raw_yuv``: a headerless planar YUV file (Netflix test-conditions .yuv, UVG-VCM). Like y4m, but
  the geometry comes from sources.yaml. With ``band_rows`` only a horizontal band of each sampled
  frame is fetched (one range request per plane), which roughly halves the transfer for 4K
  masters: the crops come from inside the band anyway.
* ``zip_sequence``: a frame sequence packed in a zip archive over HTTP(S) or on disk (the Netflix
  Open Content SDR JPEG 2000 shots). The central directory is read with a few range requests and
  then only the sampled members are fetched, so a 3 GB zip costs ~45 MB per sampled clip.
* ``image_dir``: a local folder of stills (DIV2K, your own photos). Each still becomes a short clip
  with a synthetic sub-pixel camera pan, so the codec still produces P/B-frame damage.

Any type can set ``tonemap: pq_p3d65`` for HDR10-style masters (PQ, P3-D65 RGB, e.g. the Netflix
Meridian TIFFs). The frames then go through ``ffmpeg.tonemap_pq_to_sdr`` (validated against the
official Dolby Vision SDR trim; see DATASETS.md) before cropping.
"""

from __future__ import annotations

import fnmatch
import io
import json
import struct
import threading
import time
import zipfile
import zlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import ffmpeg
from .color import rgb_to_y
from .degrade import resize
from .imio import decode_bytes, read_rgb, write_rgb16

USER_AGENT = "lasr-dataset-builder/0.1"


# --------------------------------------------------------------------------------------- HTTP


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


_tls = threading.local()


def _session():
    """One keep-alive session per thread. A fresh connection per range request restarts TCP slow
    start every time: measured from Europe against the Netflix bucket, 7 MB ranges ran at ~3.7 MB/s
    on new connections and ~15 MB/s on a reused one."""
    import requests

    s = getattr(_tls, "session", None)
    if s is None:
        s = _tls.session = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
    return s


def http_get(url: str, byte_range: tuple[int, int] | None = None, retries: int = 4) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            r = _session().get(url, headers=headers, timeout=120)
            r.raise_for_status()
            if byte_range is not None and r.status_code != 206:
                raise OSError(f"server ignored the Range header for {url}")
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def remote_size(url: str) -> int:
    r = _session().head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def read_bytes(path_or_url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    if _is_url(path_or_url):
        return http_get(path_or_url, byte_range)
    with open(path_or_url, "rb") as f:
        if byte_range is None:
            return f.read()
        f.seek(byte_range[0])
        return f.read(byte_range[1] - byte_range[0] + 1)


def total_size(path_or_url: str) -> int:
    return remote_size(path_or_url) if _is_url(path_or_url) else Path(path_or_url).stat().st_size


# --------------------------------------------------------------------------------------- Y4M

_Y4M_CHROMA = {
    # name: (chroma samples per luma sample, bytes per sample)
    "420jpeg": (0.5, 1), "420paldv": (0.5, 1), "420mpeg2": (0.5, 1), "420": (0.5, 1),
    "420p10": (0.5, 2), "420p12": (0.5, 2), "422": (1.0, 1), "422p10": (1.0, 2),
    "444": (2.0, 1), "444p10": (2.0, 2), "mono": (0.0, 1),
}  # fmt: skip


@dataclass
class Y4MInfo:
    header: bytes
    width: int
    height: int
    frame_size: int  # including the "FRAME\n" marker
    frames: int


def parse_y4m_header(head: bytes, file_size: int) -> Y4MInfo:
    end = head.index(b"\n")
    header = head[: end + 1]
    tokens = header.decode("ascii").split()
    if tokens[0] != "YUV4MPEG2":
        raise ValueError("not a YUV4MPEG2 stream")
    w = h = 0
    chroma = "420jpeg"
    for tok in tokens[1:]:
        if tok[0] == "W":
            w = int(tok[1:])
        elif tok[0] == "H":
            h = int(tok[1:])
        elif tok[0] == "C":
            chroma = tok[1:]
    if chroma not in _Y4M_CHROMA:
        raise ValueError(f"unsupported y4m chroma format {chroma}")
    ratio, bps = _Y4M_CHROMA[chroma]
    plane = int(round(w * h * (1.0 + ratio))) * bps
    marker = head[end + 1 : end + 7]
    if marker != b"FRAME\n":
        raise ValueError("y4m frames with parameters are not supported")
    fsize = plane + 6
    return Y4MInfo(header, w, h, fsize, (file_size - len(header)) // fsize)


def y4m_info(url: str) -> Y4MInfo:
    return parse_y4m_header(read_bytes(url, (0, 1023)), total_size(url))


def read_y4m_clip(url: str, info: Y4MInfo, start: int, count: int) -> np.ndarray:
    off = len(info.header) + start * info.frame_size
    data = info.header + read_bytes(url, (off, off + count * info.frame_size - 1))
    return ffmpeg.decode_to_rgb(
        ["-f", "yuv4mpegpipe", "-i", "pipe:0"],
        info.width,
        info.height,
        count,
        data=data,
        in_matrix="bt709",
        in_range="tv",
    )


# ---------------------------------------------------------------------------------- raw YUV

# ffmpeg pix_fmt: (bytes per sample, chroma width divisor, chroma height divisor, planes)
RAW_PIX_FMTS = {
    "gray": (1, 1, 1, 1), "gray10le": (2, 1, 1, 1), "gray16le": (2, 1, 1, 1),
    "yuv420p": (1, 2, 2, 3), "yuv420p10le": (2, 2, 2, 3), "yuv420p16le": (2, 2, 2, 3),
    "yuv422p": (1, 2, 1, 3), "yuv422p10le": (2, 2, 1, 3), "yuv422p16le": (2, 2, 1, 3),
    "yuv444p": (1, 1, 1, 3), "yuv444p10le": (2, 1, 1, 3), "yuv444p16le": (2, 1, 1, 3),
}  # fmt: skip


@dataclass
class RawLayout:
    """Geometry of a headerless planar YUV file (frames stored back to back, plane after plane)."""

    width: int
    height: int
    pix_fmt: str

    def __post_init__(self):
        if self.pix_fmt not in RAW_PIX_FMTS:
            raise ValueError(f"unsupported raw pix_fmt {self.pix_fmt!r}")

    @property
    def row_align(self) -> int:
        return RAW_PIX_FMTS[self.pix_fmt][2]

    def planes(self) -> list[tuple[int, int, int]]:
        """(bytes per row, rows, row divisor) of each plane."""
        bps, wdiv, hdiv, n = RAW_PIX_FMTS[self.pix_fmt]
        luma = [(self.width * bps, self.height, 1)]
        chroma = [(-(-self.width // wdiv) * bps, -(-self.height // hdiv), hdiv)] * (n - 1)
        return luma + chroma

    @property
    def frame_size(self) -> int:
        return sum(rb * rows for rb, rows, _ in self.planes())

    def band_ranges(self, frame: int, row0: int, rows: int) -> list[tuple[int, int]]:
        """Inclusive byte ranges that hold rows [row0, row0 + rows) of every plane of one frame.
        row0 and rows must be multiples of row_align (2 for 4:2:0)."""
        a = self.row_align
        if row0 % a or rows % a or row0 < 0 or row0 + rows > self.height:
            raise ValueError(f"bad band {row0}+{rows} for {self.pix_fmt} {self.width}x{self.height}")
        out, off = [], frame * self.frame_size
        for rb, prow, div in self.planes():
            start = off + (row0 // div) * rb
            out.append((start, start + (rows // div) * rb - 1))
            off += rb * prow
        return out


def read_raw_clip(
    url: str,
    layout: RawLayout,
    start: int,
    count: int,
    band: tuple[int, int] | None = None,
    matrix: str = "bt709",
    yuv_range: str = "tv",
) -> np.ndarray:
    """count frames from start as (T, H, W, 3) float RGB; with band=(row0, rows) only those rows."""
    if band is None:
        fs = layout.frame_size
        data = read_bytes(url, (start * fs, (start + count) * fs - 1))
        h = layout.height
    else:
        data = b"".join(read_bytes(url, r) for f in range(start, start + count) for r in layout.band_ranges(f, *band))
        h = band[1]
    return ffmpeg.decode_to_rgb(
        ["-f", "rawvideo", "-pix_fmt", layout.pix_fmt, "-s", f"{layout.width}x{h}", "-i", "pipe:0"],
        layout.width,
        h,
        count,
        data=data,
        in_matrix=matrix,
        in_range=yuv_range,
    )


# --------------------------------------------------------------------------------------- zip


class RangeReader(io.RawIOBase):
    """Seekable read-only file over a URL (HTTP range requests) or a local path. Only used to let
    zipfile parse the central directory; members are then fetched with read_zip_member."""

    def __init__(self, path_or_url: str):
        self.src = path_or_url
        self.size = total_size(path_or_url)
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        data = read_bytes(self.src, (self.pos, end))
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


@dataclass
class ZipMember:
    name: str
    offset: int  # of the local file header
    compressed_size: int
    method: int


def zip_members(path_or_url: str, patterns: list[str]) -> list[ZipMember]:
    """Members whose base name matches one of the patterns, sorted by name (= frame order for
    zero-padded frame numbers)."""
    with zipfile.ZipFile(io.BufferedReader(RangeReader(path_or_url), 1 << 16)) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
    out = [
        ZipMember(i.filename, i.header_offset, i.compress_size, i.compress_type)
        for i in infos
        if any(fnmatch.fnmatch(i.filename.rsplit("/", 1)[-1].lower(), p.lower()) for p in patterns)
    ]
    return sorted(out, key=lambda m: m.name)


_ZIP_EXTRA_SLACK = 512  # bytes fetched past the name for the local extra field (usually 0-32)


def read_zip_member(path_or_url: str, m: ZipMember) -> bytes:
    # local header + name + extra + data in one request; the local extra length is not in the
    # central directory, so guess with some slack and fetch the rest if it was not enough
    name_len = len(m.name.encode("utf-8"))
    want = 30 + name_len + _ZIP_EXTRA_SLACK + m.compressed_size
    blob = read_bytes(path_or_url, (m.offset, m.offset + want - 1))
    sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", blob[:30])
    if sig != 0x04034B50:
        raise ValueError(f"bad local zip header for {m.name}")
    start = 30 + name_len + extra_len
    data = blob[start : start + m.compressed_size]
    if len(data) < m.compressed_size:
        rest = m.offset + start + len(data)
        data += read_bytes(path_or_url, (rest, m.offset + start + m.compressed_size - 1))
    if m.method == zipfile.ZIP_STORED:
        return data
    if m.method == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported zip compression {m.method} for {m.name}")


J2K_EXTS = (".j2k", ".j2c")


def j2k_size(codestream: bytes) -> tuple[int, int]:
    """Image width and height from the SIZ marker of a raw JPEG 2000 codestream."""
    i = codestream.find(b"\xff\x51")
    if i < 0:
        raise ValueError("no SIZ marker: not a JPEG 2000 codestream")
    x, y, xo, yo = struct.unpack(">IIII", codestream[i + 6 : i + 22])
    return x - xo, y - yo


def decode_frames(blobs: list[bytes], names: list[str], matrix: str = "bt709", yuv_range: str = "tv") -> np.ndarray:
    """Encoded frames -> (T, H, W, 3) float RGB. Raw JPEG 2000 codestreams (YCbCr, e.g. the Netflix
    SDR shots) go through ffmpeg with an explicit matrix/range; everything else through OpenCV."""
    if all(n.lower().endswith(J2K_EXTS) for n in names):
        w, h = j2k_size(blobs[0])
        return ffmpeg.decode_to_rgb(
            ["-f", "j2k_pipe", "-i", "pipe:0"], w, h, len(blobs), data=b"".join(blobs),
            in_matrix=matrix, in_range=yuv_range,
        )  # fmt: skip
    return np.stack([decode_bytes(b) for b in blobs])


# ------------------------------------------------------------------------------------ sources


@dataclass
class SourceSpec:
    id: str
    type: str
    split: str = "train"
    tier: str = "permissive"
    license: str = ""
    enabled: bool = True
    positions: int = 50
    crops_per_position: int = 2
    hr_scale: list[float] = field(default_factory=lambda: [0.75, 1.0])
    options: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> SourceSpec:
        known = {f for f in SourceSpec.__dataclass_fields__ if f != "options"}
        return SourceSpec(
            **{k: v for k, v in d.items() if k in known},
            options={k: v for k, v in d.items() if k not in known},
        )


def load_sources(path: str | Path) -> list[SourceSpec]:
    import yaml

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return [SourceSpec.from_dict(d) for d in doc["sources"]]


# permissive: default set (reuse, modification and commercial use allowed, ML training not forbidden)
# sharealike: CC BY-SA. Commercial use and derivatives are allowed, but whether trained weights are
#             "adapted material" that must carry BY-SA is unsettled, so it is opt-in.
# research:   research-only / non-commercial / no-derivatives terms. Never ship weights trained on it.
TIERS = ("permissive", "sharealike", "research")


def select_sources(
    specs: list[SourceSpec],
    only: list[str] | None = None,
    include_sharealike: bool = False,
    include_research: bool = False,
) -> list[SourceSpec]:
    """The sources a fetch run uses. --only names sources explicitly (ignores enabled and tier);
    otherwise enabled sources of the permitted tiers. --include-research also includes sharealike."""
    for s in specs:
        if s.tier not in TIERS:
            raise ValueError(f"source {s.id}: unknown tier {s.tier!r} (expected one of {TIERS})")
    if only:
        unknown = set(only) - {s.id for s in specs}
        if unknown:
            raise KeyError(f"unknown source ids: {sorted(unknown)}")
        return [s for s in specs if s.id in only]
    tiers = {"permissive"}
    if include_sharealike or include_research:
        tiers.add("sharealike")
    if include_research:
        tiers.add("research")
    return [s for s in specs if s.enabled and s.tier in tiers]


def _positions(lo: int, hi: int, n: int, rng: np.random.Generator) -> list[int]:
    """n start indices spread evenly over [lo, hi] with jitter (scene diversity, few repeats)."""
    if hi <= lo:
        return [lo]
    n = max(1, min(n, hi - lo + 1))
    edges = np.linspace(lo, hi + 1, n + 1)
    return sorted({int(rng.uniform(a, max(a + 1, b))) for a, b in zip(edges[:-1], edges[1:])})


def iter_source_clips(spec: SourceSpec, clip_len: int, rng: np.random.Generator) -> Iterator[tuple[int, Any]]:
    """Yields (position index, loader) where loader() returns (T, H, W, 3) float RGB (SDR BT.709;
    HDR sources with a ``tonemap`` option are tone-mapped inside the loader)."""
    tm = spec.options.get("tonemap")
    if tm is not None and tm not in ffmpeg.TONEMAPS:
        raise ValueError(f"source {spec.id}: unknown tonemap {tm!r} (expected one of {sorted(ffmpeg.TONEMAPS)})")
    for i, loader in _iter_clips(spec, clip_len, rng):
        if tm is None:
            yield i, loader
        else:
            yield i, (lambda loader=loader: ffmpeg.tonemap_pq_to_sdr(loader(), tm))


def _iter_clips(spec: SourceSpec, clip_len: int, rng: np.random.Generator) -> Iterator[tuple[int, Any]]:
    o = spec.options
    if spec.type == "png_sequence":
        first, last = int(o["first"]), int(o["last"])
        tmpl = o["url"]

        def load_png(start: int) -> np.ndarray:
            frames = []
            for n in range(start, start + clip_len):
                p = tmpl.format(n=n)
                frames.append(decode_bytes(read_bytes(p)) if _is_url(p) else read_rgb(p))
            return np.stack(frames)

        for i, s in enumerate(_positions(first, last - clip_len + 1, spec.positions, rng)):
            yield i, (lambda s=s: load_png(s))

    elif spec.type == "y4m":
        info = y4m_info(o["url"])
        skip = int(o.get("skip_frames", 0))
        for i, s in enumerate(_positions(skip, info.frames - clip_len - skip, spec.positions, rng)):
            yield i, (lambda s=s: read_y4m_clip(o["url"], info, s, clip_len))

    elif spec.type == "video":
        meta = ffmpeg.probe(o["url"])
        t0 = float(o.get("skip_start", 0.0))
        t1 = meta["duration"] - float(o.get("skip_end", 0.0)) - clip_len / meta["fps"]
        n0, n1 = int(t0 * 1000), int(max(t0, t1) * 1000)

        def load_video(ms: int) -> np.ndarray:
            return ffmpeg.decode_to_rgb(
                ["-ss", f"{ms / 1000:.3f}", "-i", o["url"]],
                meta["width"],
                meta["height"],
                clip_len,
                in_matrix=o.get("matrix", "auto"),
            )

        for i, ms in enumerate(_positions(n0, n1, spec.positions, rng)):
            yield i, (lambda ms=ms: load_video(ms))

    elif spec.type == "raw_yuv":
        layout = RawLayout(int(o["width"]), int(o["height"]), o["pix_fmt"])
        frames = total_size(o["url"]) // layout.frame_size
        skip = int(o.get("skip_frames", 0))
        band_rows = o.get("band_rows")
        if band_rows is not None:
            a = layout.row_align
            band_rows = min(layout.height, int(band_rows)) // a * a

        matrix, yuv_range = o.get("matrix", "bt709"), o.get("range", "tv")

        def load_raw(start: int, band: tuple[int, int] | None) -> np.ndarray:
            return read_raw_clip(o["url"], layout, start, clip_len, band, matrix, yuv_range)

        for i, s in enumerate(_positions(skip, frames - clip_len - skip, spec.positions, rng)):
            band = None
            if band_rows is not None:
                row0 = int(rng.integers(0, layout.height - band_rows + 1)) // layout.row_align * layout.row_align
                band = (row0, band_rows)
            yield i, (lambda s=s, band=band: load_raw(s, band))

    elif spec.type == "zip_sequence":
        members = zip_members(o["url"], o.get("patterns", ["*.j2k", "*.j2c", "*.png", "*.tif", "*.tiff"]))
        skip = int(o.get("skip_frames", 0))

        def load_zip(start: int) -> np.ndarray:
            ms = members[start : start + clip_len]
            blobs = [read_zip_member(o["url"], m) for m in ms]
            return decode_frames(blobs, [m.name for m in ms], o.get("matrix", "bt709"), o.get("range", "tv"))

        for i, s in enumerate(_positions(skip, len(members) - clip_len - skip, spec.positions, rng)):
            yield i, (lambda s=s: load_zip(s))

    elif spec.type == "image_dir":
        root = Path(o["path"])
        pats = o.get("patterns", ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp"])
        files = sorted({p for pat in pats for p in root.rglob(pat)})
        if spec.positions and len(files) > spec.positions:
            files = [files[i] for i in sorted(rng.choice(len(files), spec.positions, replace=False))]
        for i, f in enumerate(files):
            yield i, (lambda f=f: read_rgb(f)[None])  # one still; panned later
    else:
        raise ValueError(f"unknown source type {spec.type!r}")


# -------------------------------------------------------------------------------------- crops


def dhash(y: np.ndarray) -> int:
    small = cv2.resize(y.astype(np.float32), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def pan_clip(still: np.ndarray, crop: int, clip_len: int, rng: np.random.Generator) -> np.ndarray | None:
    """Synthetic camera pan over a still: (clip_len, crop, crop, 3), sub-pixel motion."""
    h, w, _ = still.shape
    v = rng.uniform(-2.5, 2.5, size=2)
    margin = int(np.ceil(np.abs(v).max() * clip_len)) + 4
    if h < crop + 2 * margin or w < crop + 2 * margin:
        return None
    x0 = int(rng.integers(margin, w - crop - margin + 1))
    y0 = int(rng.integers(margin, h - crop - margin + 1))
    frames = []
    for t in range(clip_len):
        m = np.array([[1, 0, -(x0 + v[0] * t)], [0, 1, -(y0 + v[1] * t)]], dtype=np.float32)
        f = cv2.warpAffine(still, m, (crop, crop), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frames.append(np.clip(f, 0.0, 1.0))
    return np.stack(frames)


@dataclass
class ExtractConfig:
    clip_len: int = 6
    crop: int = 512
    min_std: float = 0.035  # luma std below this is a flat crop (sky, black, credits): skipped
    dedup_bits: int = 5  # dHash Hamming distance at or below this counts as a duplicate
    tries: int = 8
    threads: int = 8


class Extractor:
    def __init__(self, out_root: str | Path, cfg: ExtractConfig, seed: int = 0):
        self.root = Path(out_root)
        self.cfg = cfg
        self.seed = seed
        self._lock = threading.Lock()
        self._hashes: dict[str, list[int]] = {}

    def _accept(self, source: str, y0: np.ndarray) -> bool:
        if float(y0.std()) < self.cfg.min_std:
            return False
        h = dhash(y0)
        with self._lock:
            seen = self._hashes.setdefault(source, [])
            if any(hamming(h, s) <= self.cfg.dedup_bits for s in seen):
                return False
            seen.append(h)
        return True

    def _crops(self, spec: SourceSpec, frames: np.ndarray, rng: np.random.Generator):
        c = self.cfg.crop
        lo, hi = spec.hr_scale
        if spec.type == "image_dir":
            still = frames[0]
            s = float(rng.uniform(lo, hi))
            if s < 0.999:
                h, w, _ = still.shape
                still = resize(still, (round(w * s), round(h * s)), "area")
            for _ in range(self.cfg.tries):
                clip = pan_clip(still, c, self.cfg.clip_len, rng)
                if clip is not None:
                    yield clip
            return
        s = float(rng.uniform(lo, hi))
        t, h, w, _ = frames.shape
        if s < 0.999:
            nw, nh = round(w * s), round(h * s)
            frames = np.stack([resize(f, (nw, nh), "area") for f in frames])
            h, w = nh, nw
        if h < c or w < c:
            return
        for _ in range(self.cfg.tries):
            y0 = int(rng.integers(0, h - c + 1)) & ~3
            x0 = int(rng.integers(0, w - c + 1)) & ~3
            yield frames[:, y0 : y0 + c, x0 : x0 + c]

    def run_source(self, spec: SourceSpec, log=print) -> int:
        rng = np.random.default_rng([self.seed, zlib.crc32(spec.id.encode())])
        out = self.root / spec.split / spec.id
        jobs = []
        for pos, loader in iter_source_clips(spec, self.cfg.clip_len, rng):
            jobs.append((pos, loader, int(rng.integers(2**31 - 1))))

        def work(job) -> int:
            pos, loader, s = job
            done = [p for p in out.glob(f"p{pos:05d}c*/meta.json")]
            if done:
                return len(done)
            try:
                frames = loader()
            except Exception as e:  # network hiccup, truncated file: skip this position
                log(f"  [{spec.id}] position {pos}: {type(e).__name__}: {e}")
                return 0
            if frames.ndim != 4 or (spec.type != "image_dir" and len(frames) < self.cfg.clip_len):
                return 0
            r = np.random.default_rng(s)
            kept = 0
            for clip in self._crops(spec, frames, r):
                if kept >= spec.crops_per_position:
                    break
                if not self._accept(spec.id, rgb_to_y(clip[0])):
                    continue
                d = out / f"p{pos:05d}c{kept}"
                for i, f in enumerate(clip):
                    write_rgb16(d / f"f{i:02d}.png", f)
                meta = {
                    "source": spec.id,
                    "split": spec.split,
                    "tier": spec.tier,
                    "license": spec.license,
                    "frames": len(clip),
                    "size": [int(clip.shape[2]), int(clip.shape[1])],
                }
                (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                kept += 1
            return kept

        with ThreadPoolExecutor(self.cfg.threads) as ex:
            total = sum(ex.map(work, jobs))
        log(f"  [{spec.id}] {total} clips from {len(jobs)} positions")
        return total


def list_clips(hr_root: str | Path, split: str) -> list[Path]:
    return sorted(p.parent for p in (Path(hr_root) / split).glob("*/*/meta.json"))


def load_clip(clip_dir: str | Path) -> np.ndarray:
    files = sorted(Path(clip_dir).glob("f*.png"))
    return np.stack([read_rgb(f) for f in files])
