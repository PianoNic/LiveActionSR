"""Train a 2x luma SR model on baked shards.

python train.py --config configs/balanced.yaml
python train.py --config configs/quality.yaml --set train.iterations=300000
python train.py --config configs/balanced.yaml --resume runs/balanced/last.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lasr.config import load_config
from lasr.data import PairShards, PatchDataset
from lasr.losses import SRLoss
from lasr.models import build_model, count_params
from lasr.validate import Validator, format_report


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_lambda(cfg: dict):
    total, warm = cfg["iterations"], cfg["warmup"]
    floor = cfg["min_lr"] / cfg["lr"]

    def f(it: int) -> float:
        if it < warm:
            return (it + 1) / warm
        t = (it - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return f


def model_cfg_of(model) -> dict:
    return {"channels": model.channels, "blocks": model.blocks, "scale": model.scale, "rep": model.rep}


def save_ckpt(path: Path, model, opt, sched, it: int, best: float, cfg: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": model_cfg_of(model),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "iteration": it,
            "best_psnr": best,
            "config": cfg,
        },
        tmp,
    )
    tmp.replace(path)


class CSVLog:
    def __init__(self, path: Path, fields: list[str]):
        new = not path.exists()
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def write(self, row: dict) -> None:
        self.w.writerow(row)
        self.f.flush()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    tc, lc = cfg["train"], cfg["log"]
    torch.manual_seed(cfg["seed"])
    device = pick_device(tc["device"])
    out = Path(lc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    model = build_model(cfg["model"]).to(device)
    print(f"model: {model_cfg_of(model)} params={count_params(model)} device={device}")

    train_shards = PairShards(cfg["data"]["root"], "train")
    val_shards = PairShards(cfg["data"]["root"], "val")
    print(f"data: {len(train_shards)} train pairs, {len(val_shards)} val pairs")
    ds = PatchDataset(train_shards, cfg["data"]["patch"], length=tc["iterations"] * tc["batch"])
    loader = DataLoader(
        ds,
        batch_size=tc["batch"],
        num_workers=tc["workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=tc["workers"] > 0,
        prefetch_factor=4 if tc["workers"] > 0 else None,
    )
    validator = Validator(val_shards, cfg["data"].get("val_limit"))

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]), weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc))
    loss_fn = SRLoss(**cfg["loss"])

    start, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best = ck["iteration"], ck["best_psnr"]
        print(f"resumed from {args.resume} at iteration {start}")

    amp = tc["amp"] if device.type == "cuda" else "off"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    step_model = torch.compile(model) if tc["compile"] else model

    tb = None
    if lc.get("tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb = SummaryWriter(str(out / "tb"))
        except ImportError:
            print("tensorboard not installed, logging to CSV only")
    train_log = CSVLog(out / "train.csv", ["iteration", "loss", "l1", "fft", "edge", "lr", "it_per_s"])
    val_log = CSVLog(
        out / "val.csv",
        ["iteration", "psnr", "ssim", "psnr_bilinear", "psnr_bicubic", "psnr_lanczos", "gain_vs_lanczos_db"],
    )

    def validate(it: int) -> float:
        nonlocal best
        res = validator.run(model, device)
        print(f"[val @ {it}]\n{format_report(res)}")
        row = {
            "iteration": it,
            "psnr": res["model"]["psnr"],
            "ssim": res["model"]["ssim"],
            "psnr_bilinear": res["bilinear"]["psnr"],
            "psnr_bicubic": res["bicubic"]["psnr"],
            "psnr_lanczos": res["lanczos"]["psnr"],
            "gain_vs_lanczos_db": res["gain_vs_lanczos_db"],
        }
        val_log.write(row)
        if tb:
            for k, v in row.items():
                if k != "iteration":
                    tb.add_scalar(f"val/{k}", v, it)
        (out / "val_latest.json").write_text(json.dumps({"iteration": it, **res}, indent=2), encoding="utf-8")
        if res["model"]["psnr"] > best:
            best = res["model"]["psnr"]
            save_ckpt(out / "best.pt", model, opt, sched, it, best, cfg)
        save_ckpt(out / "last.pt", model, opt, sched, it, best, cfg)
        return res["model"]["psnr"]

    model.train()
    it = start
    t0, seen = time.time(), 0
    running: dict[str, float] = {}
    last_val: tuple[int, float] = (-1, 0.0)
    for lr_b, hr_b in loader:
        if it >= tc["iterations"]:
            break
        lr_b = lr_b.to(device, non_blocking=True)
        hr_b = hr_b.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            pred = step_model(lr_b)
        loss, parts = loss_fn(pred.float(), hr_b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if tc["grad_clip"]:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        it += 1
        seen += 1
        for k, v in {"loss": float(loss.detach()), **parts}.items():
            running[k] = running.get(k, 0.0) + v

        if it % lc["every"] == 0 or it == tc["iterations"]:
            dt = time.time() - t0
            row = {k: v / seen for k, v in running.items()}
            row.update(iteration=it, lr=sched.get_last_lr()[0], it_per_s=seen / max(dt, 1e-9))
            train_log.write(row)
            if tb:
                for k, v in row.items():
                    if k != "iteration":
                        tb.add_scalar(f"train/{k}", v, it)
            print(f"it {it:>7d}  loss {row['loss']:.5f}  lr {row['lr']:.2e}  {row['it_per_s']:.1f} it/s")
            t0, seen, running = time.time(), 0, {}
        if it % lc["val_every"] == 0:
            last_val = (it, validate(it))

    final = last_val[1] if last_val[0] == it else validate(it)
    if tb:
        tb.close()
    print(f"done: {it} iterations, final val PSNR {final:.3f} dB, best {best:.3f} dB -> {out}")
    return {"iteration": it, "final_psnr": final, "best_psnr": best, "out_dir": str(out)}


if __name__ == "__main__":
    main()
