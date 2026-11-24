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
* ``image_dir``: a local folder of stills (DIV2K, your own photos). Each still becomes a short clip
  with a synthetic sub-pixel camera pan, so the codec still produces P/B-frame damage.
"""

from __future__ import annotations

import json
import threading
import time
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


def http_get(url: str, byte_range: tuple[int, int] | None = None, retries: int = 4) -> bytes:
    import requests

    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
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
    import requests

    r = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=60, allow_redirects=True)
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


def _positions(lo: int, hi: int, n: int, rng: np.random.Generator) -> list[int]:
    """n start indices spread evenly over [lo, hi] with jitter (scene diversity, few repeats)."""
    if hi <= lo:
        return [lo]
    n = max(1, min(n, hi - lo + 1))
    edges = np.linspace(lo, hi + 1, n + 1)
    return sorted({int(rng.uniform(a, max(a + 1, b))) for a, b in zip(edges[:-1], edges[1:])})


def iter_source_clips(spec: SourceSpec, clip_len: int, rng: np.random.Generator) -> Iterator[tuple[int, Any]]:
    """Yields (position index, loader) where loader() returns (T, H, W, 3) float RGB."""
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
