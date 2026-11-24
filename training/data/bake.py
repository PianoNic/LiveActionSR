"""Step 2: degrade the HR clips (codec round-trips etc.) into memory-mappable training shards.

    python data/bake.py --hr work/hr --out work/baked --split train --variants 3 --keep 3
    python data/bake.py --hr work/hr --out work/baked --split val --variants 1 --keep 3

Degradation settings come from the `degradation:` section of a training config (defaults in
lasr/degrade.py). The val split uses a fixed seed, so every model is scored on the same pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lasr import ffmpeg  # noqa: E402
from lasr.bake import bake  # noqa: E402
from lasr.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hr", required=True, help="output of data/fetch.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--variants", type=int, default=3, help="random degradations per HR clip")
    ap.add_argument("--keep", type=int, default=3, help="frames kept per encoded clip (0 = all)")
    ap.add_argument("--config", default=None, help="training config whose `degradation:` section to use")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=None, help="default: 0 for train, 12345 for val")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--no-codec", action="store_true", help="skip codec round-trips (no ffmpeg)")
    args = ap.parse_args(argv)

    deg = load_config(args.config).get("degradation", {}) if args.config else {}
    codecs = () if args.no_codec else ffmpeg.available_codecs()
    if not args.no_codec:
        if not ffmpeg.ffmpeg_available():
            ap.error("ffmpeg not found (use --no-codec to bake without codec damage)")
        missing = {"x264", "x265", "av1"} - set(codecs)
        if missing:
            print(f"warning: ffmpeg lacks encoders for {sorted(missing)}; they will not be used")
    seed = args.seed if args.seed is not None else (0 if args.split == "train" else 12345)
    bake(
        args.hr,
        args.out,
        args.split,
        variants=args.variants,
        keep_frames=args.keep,
        deg_cfg=deg,
        workers=args.workers,
        shard_size=args.shard_size,
        seed=seed,
        codecs=codecs,
        max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
