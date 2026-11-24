"""Thin ffmpeg wrappers: codec round-trips for the degradation pipeline and frame reading for data
preparation. Everything goes through pipes (raw video in, Matroska in memory, raw video out), so no
temporary files are written.

Colour handling is explicit in both directions (BT.709 matrix, limited range in the bitstream).
ffmpeg's default for untagged material is BT.601, which would shift every luma value.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

ENCODERS = {"x264": "libx264", "x265": "libx265", "av1": "libsvtav1"}

_SWS = "flags=bicubic+accurate_rnd+full_chroma_int"


class FFmpegError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    try:
        subprocess.run([FFMPEG, "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@functools.lru_cache(maxsize=1)
def available_codecs() -> tuple[str, ...]:
    """Short names (x264/x265/av1) whose encoder this ffmpeg build has."""
    if not ffmpeg_available():
        return ()
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return tuple(name for name, enc in ENCODERS.items() if f" {enc} " in out)


def _run(cmd: list[str], data: bytes | None = None) -> bytes:
    proc = subprocess.run(cmd, input=data, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-2000:]
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd[:12])} ...\n{err}")
    return proc.stdout


def encoder_args(codec: str, crf: int, preset: str, keyint: int, bframes: int) -> list[str]:
    """Encoder settings modelled on VOD streaming ladders (CRF stands in for capped VBR). Each
    encoder is pinned to one thread: parallelism comes from running many clips at once."""
    if codec == "x264":
        return [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-g", str(keyint),
            "-bf", str(bframes), "-threads", "1",
            "-x264-params", "log-level=error",
        ]  # fmt: skip
    if codec == "x265":
        return [
            "-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-g", str(keyint),
            "-bf", str(bframes),
            "-x265-params", "log-level=error:pools=1:frame-threads=1",
        ]  # fmt: skip
    if codec == "av1":
        return [
            "-c:v", "libsvtav1", "-preset", str(preset), "-crf", str(crf), "-g", str(keyint),
            "-svtav1-params", "lp=1",
        ]  # fmt: skip
    raise ValueError(f"unknown codec {codec!r}")


# The smallest frame side an encoder accepts.
MIN_FRAME_SIDE = {"av1": 64}


def codec_roundtrip(
    frames: np.ndarray,
    codec: str,
    crf: int,
    preset: str,
    keyint: int = 48,
    bframes: int = 3,
    fps: int = 24,
) -> tuple[np.ndarray, int]:
    """Encode a clip and decode it again, as a viewer's browser would receive it.

    frames: (T, H, W, 3) float RGB in [0, 1], H and W even.
    Returns ((T, H, W, 3) uint8 RGB, bitstream size in bytes). The bitstream is 4:2:0 BT.709
    limited range; it is decoded back to 8-bit RGB like a browser's video texture.
    """
    t, h, w, _ = frames.shape
    if h % 2 or w % 2:
        raise ValueError("frame size must be even for 4:2:0")
    # Some encoders refuse tiny frames (SVT-AV1 wants at least 64x64): pad by repeating the edge,
    # encode, and crop the decoded frames back.
    min_side = MIN_FRAME_SIDE.get(codec, 0)
    if h < min_side or w < min_side:
        ph, pw = max(0, min_side - h), max(0, min_side - w)
        padded = np.pad(frames, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="edge")
        out, size = codec_roundtrip(padded, codec, crf, preset, keyint, bframes, fps)
        return np.ascontiguousarray(out[:, :h, :w]), size
    raw = (np.clip(frames, 0.0, 1.0) * 65535.0 + 0.5).astype("<u2").tobytes()
    enc = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-vf", f"scale=out_color_matrix=bt709:out_range=tv:{_SWS},format=yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        *encoder_args(codec, crf, preset, keyint, bframes),
        "-f", "matroska", "pipe:1",
    ]  # fmt: skip
    bitstream = _run(enc, raw)
    dec = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
        "-f", "matroska", "-i", "pipe:0",
        "-vf", f"scale=in_color_matrix=bt709:in_range=tv:{_SWS},format=rgb24",
        "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip
    out = _run(dec, bitstream)
    n = len(out) // (h * w * 3)
    if n < t:
        raise FFmpegError(f"decoder returned {n} frames, expected {t}")
    return np.frombuffer(out, np.uint8)[: t * h * w * 3].reshape(t, h, w, 3), len(bitstream)


def decode_to_rgb(
    args_in: list[str],
    width: int,
    height: int,
    frames: int,
    data: bytes | None = None,
    in_matrix: str = "auto",
    in_range: str = "auto",
) -> np.ndarray:
    """Run ffmpeg with the given input arguments and return (N, H, W, 3) float32 RGB in [0, 1]
    (16-bit intermediate, so 10-bit sources keep their precision)."""
    vf = f"scale=in_color_matrix={in_matrix}:in_range={in_range}:{_SWS},format=rgb48le"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args_in,
        "-frames:v", str(frames), "-vf", vf, "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip
    out = _run(cmd, data)
    px = width * height * 3
    arr = np.frombuffer(out, "<u2")
    n = arr.size // px
    return arr[: n * px].reshape(n, height, width, 3).astype(np.float32) / 65535.0


def probe(url: str) -> dict:
    """Width, height, fps and duration of the first video stream."""
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json", url,
    ]  # fmt: skip
    info = json.loads(_run(cmd))
    s = info["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": num / den if den else 24.0,
        "duration": float(info.get("format", {}).get("duration", 0.0) or 0.0),
    }
