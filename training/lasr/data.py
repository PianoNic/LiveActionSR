"""Training / validation datasets over baked shards (see lasr/bake.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PairShards:
    """Memory-mapped view over all shards of one split. Opened lazily so it can be pickled into
    DataLoader workers."""

    def __init__(self, root: str | Path, split: str):
        self.dir = Path(root) / split
        self.stems = sorted(str(p)[: -len("_lr.npy")] for p in self.dir.glob("shard_*_lr.npy"))
        if not self.stems:
            raise FileNotFoundError(f"no baked shards in {self.dir}")
        self.counts = []
        for s in self.stems:
            arr = np.load(f"{s}_lr.npy", mmap_mode="r")
            self.counts.append(arr.shape[0])
            self.lr_shape = arr.shape[1:]
        self.offsets = np.cumsum([0] + self.counts)
        self._lr: list[np.ndarray] | None = None
        self._hr: list[np.ndarray] | None = None
        self._meta: list[dict] | None = None

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _open(self) -> None:
        if self._lr is None:
            self._lr = [np.load(f"{s}_lr.npy", mmap_mode="r") for s in self.stems]
            self._hr = [np.load(f"{s}_hr.npy", mmap_mode="r") for s in self.stems]

    def locate(self, i: int) -> tuple[int, int]:
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return k, i - int(self.offsets[k])

    def pair(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        self._open()
        k, j = self.locate(i)
        return self._lr[k][j], self._hr[k][j]

    def meta(self, i: int) -> dict:
        if self._meta is None:
            self._meta = []
            for s in self.stems:
                with open(f"{s}_meta.jsonl", encoding="utf-8") as f:
                    self._meta.extend(json.loads(line) for line in f if line.strip())
        return self._meta[i]


def lr_to_float(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 65535.0


def hr_to_float(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 255.0


class PatchDataset(Dataset):
    """Random LR patches (patch x patch) with the aligned 2x HR patch, plus flips/transposes.
    `length` is the virtual epoch length; every item draws a random pair and position."""

    def __init__(self, shards: PairShards, patch: int, length: int, augment: bool = True):
        self.shards = shards
        self.patch = patch
        self.length = length
        self.augment = augment
        h, w = shards.lr_shape
        if h < patch or w < patch:
            raise ValueError(f"patch {patch} larger than baked LR frames {w}x{h}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(torch.randint(len(self.shards), ()).item())
        lr, hr = self.shards.pair(i)
        h, w = lr.shape
        p = self.patch
        y = int(torch.randint(h - p + 1, ()).item())
        x = int(torch.randint(w - p + 1, ()).item())
        a = lr_to_float(lr[y : y + p, x : x + p])
        b = hr_to_float(hr[2 * y : 2 * (y + p), 2 * x : 2 * (x + p)])
        if self.augment:
            r = int(torch.randint(8, ()).item())
            if r & 1:
                a, b = a[:, ::-1], b[:, ::-1]
            if r & 2:
                a, b = a[::-1], b[::-1]
            if r & 4:
                a, b = a.T, b.T
        return (
            torch.from_numpy(np.ascontiguousarray(a))[None],
            torch.from_numpy(np.ascontiguousarray(b))[None],
        )


class FullFrameDataset(Dataset):
    """Validation: every baked pair at full size (deterministic)."""

    def __init__(self, shards: PairShards, limit: int | None = None):
        self.shards = shards
        self.n = min(len(shards), limit) if limit else len(shards)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        lr, hr = self.shards.pair(i)
        return (
            torch.from_numpy(lr_to_float(np.asarray(lr)))[None],
            torch.from_numpy(hr_to_float(np.asarray(hr)))[None],
            i,
        )
