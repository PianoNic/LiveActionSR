"""Write procedural HR clips (no downloads) for smoke tests of the whole pipeline.

python data/make_synthetic.py --out work/hr --train 24 --val 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lasr.synthetic import make_synthetic_clips  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=24)
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--clip-len", type=int, default=4)
    args = ap.parse_args(argv)
    make_synthetic_clips(args.out, "train", args.train, args.crop, args.clip_len, seed=1)
    make_synthetic_clips(args.out, "val", args.val, args.crop, args.clip_len, seed=2)
    print(f"wrote {args.train} train + {args.val} val synthetic clips to {args.out}")


if __name__ == "__main__":
    main()
