"""Pre-bake degraded (LR, HR) luma pairs to disk.

Codec round-trips cost far more CPU than a forward/backward pass of a 12k-parameter net costs GPU,
so degradations are baked once (several random variants per HR clip) and training streams random
patches out of memory-mapped shards.

Shard layout (per split):
    shard_0000_lr.npy   (N, h, w)   uint16  LR luma * 65535 (decoded 8-bit RGB -> BT.709 Y)
    shard_0000_hr.npy   (N, 2h, 2w) uint8   HR luma * 255
    shard_0000_meta.jsonl            one line per pair: clip, variant, frame, codec, crf, ...
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import traceback
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .color import to_uint8, to_uint16
from .degrade import degrade_clip, sample_params
from .sources import list_clips, load_clip


def _task(args: tuple) -> tuple[np.ndarray, np.ndarray, list[dict]] | str:
    clip_dir, rel, variant, seed, deg_cfg, keep, codecs = args
    try:
        hr = load_clip(clip_dir)
        rng = np.random.default_rng(seed)
        params = sample_params(deg_cfg, rng, codecs)
        lr_y, hr_y, info = degrade_clip(hr, params)
        t = len(hr)
        idx = np.arange(t) if not keep or keep >= t else np.sort(rng.choice(t, keep, replace=False))
        c = params["codec"] or {}
        metas = [
            {
                "clip": rel,
                "variant": variant,
                "frame": int(i),
                "codec": info["codec"],
                "crf": c.get("crf"),
                "preset": c.get("preset"),
                "bpp": info["bpp"],
                "kernel": params["kernel"],
                "blur": round(params["blur_sigma"], 3),
                "rescale": round(params["rescale"], 3),
                "sharpen": bool(params["sharpen"]),
                "noise": bool(params["noise"]),
            }
            for i in idx
        ]
        return to_uint16(lr_y[idx]), to_uint8(hr_y[idx]), metas
    except Exception:
        return f"{rel} v{variant}: {traceback.format_exc(limit=3)}"


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out = out_dir
        self.size = shard_size
        self.lr: list[np.ndarray] = []
        self.hr: list[np.ndarray] = []
        self.meta: list[dict] = []
        self.index = 0
        self.total = 0
        self.out.mkdir(parents=True, exist_ok=True)

    def add(self, lr: np.ndarray, hr: np.ndarray, metas: list[dict]) -> None:
        self.lr.extend(lr)
        self.hr.extend(hr)
        self.meta.extend(metas)
        while len(self.lr) >= self.size:
            self._flush(self.size)

    def _flush(self, n: int) -> None:
        if n == 0:
            return
        stem = self.out / f"shard_{self.index:04d}"
        np.save(f"{stem}_lr.npy", np.stack(self.lr[:n]))
        np.save(f"{stem}_hr.npy", np.stack(self.hr[:n]))
        with open(f"{stem}_meta.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for m in self.meta[:n]:
                f.write(json.dumps(m) + "\n")
        del self.lr[:n], self.hr[:n], self.meta[:n]
        self.index += 1
        self.total += n

    def close(self) -> int:
        self._flush(len(self.lr))
        return self.total


def bake(
    hr_root: str | Path,
    out_root: str | Path,
    split: str,
    variants: int,
    keep_frames: int,
    deg_cfg: dict[str, Any] | None,
    workers: int,
    shard_size: int = 2048,
    seed: int = 0,
    codecs: tuple[str, ...] | None = None,
    max_clips: int | None = None,
    log=print,
) -> int:
    clips = list_clips(hr_root, split)
    if max_clips:
        clips = clips[:max_clips]
    out_dir = Path(out_root) / split
    if any(out_dir.glob("shard_*_lr.npy")):
        raise FileExistsError(f"{out_dir} already has shards; delete it to re-bake")
    root = Path(hr_root)
    tasks = []
    for c in clips:
        rel = c.relative_to(root).as_posix()
        for v in range(variants):
            s = zlib.crc32(f"{seed}/{rel}/{v}".encode())
            tasks.append((str(c), rel, v, s, deg_cfg, keep_frames, codecs))
    log(f"baking {split}: {len(clips)} clips x {variants} variants = {len(tasks)} tasks, {workers} workers")
    writer = ShardWriter(out_dir, shard_size)
    errors = 0
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover

        def tqdm(x, **_):
            return x

    if workers <= 1:
        results = map(_task, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn" if sys.platform == "win32" else "fork").Pool(workers)
        results = pool.imap_unordered(_task, tasks, chunksize=1)
    try:
        for r in tqdm(results, total=len(tasks), desc=f"bake {split}", mininterval=5):
            if isinstance(r, str):
                errors += 1
                if errors <= 10:
                    log("  error: " + r)
                continue
            writer.add(*r)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    n = writer.close()
    log(f"baked {n} pairs into {out_dir} ({errors} failed tasks)")
    return n
