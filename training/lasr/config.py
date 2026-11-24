"""YAML config loading with `--set a.b=c` overrides and an `extends:` key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .degrade import merge_config

DEFAULTS: dict[str, Any] = {
    "name": "balanced",
    "seed": 0,
    "data": {"root": "work/baked", "patch": 64, "val_limit": 512},
    "model": {"preset": "balanced", "rep": False},
    "train": {
        "iterations": 200_000,
        "batch": 64,
        "lr": 5.0e-4,
        "min_lr": 1.0e-6,
        "warmup": 1000,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.99],
        "amp": "bf16",
        "grad_clip": 1.0,
        "workers": 8,
        "compile": False,
        "device": "auto",
    },
    "loss": {"l1": 1.0, "fft": 0.0, "edge": 0.0},
    "log": {"out_dir": "runs/balanced", "every": 200, "val_every": 5000, "tensorboard": True},
}


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg = merge_config(cfg, _load_with_extends(Path(path)))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def _load_with_extends(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    parent = doc.pop("extends", None)
    if parent:
        return merge_config(_load_with_extends(path.parent / parent), doc)
    return doc
