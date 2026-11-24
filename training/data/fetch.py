"""Step 1: sample short HR clips from the sources in sources.yaml and store deduplicated crops as
16-bit PNG.

    python data/fetch.py --out work/hr                       # permissive set
    python data/fetch.py --out work/hr --only tum_blue_sky --positions 2   # tiny sample
    python data/fetch.py --out work/hr --include-sharealike  # + CC BY-SA sources
    python data/fetch.py --out work/hr --include-research    # + research-only (never ship those weights)

Output: <out>/<split>/<source>/p<position>c<crop>/f00.png ... + meta.json. Re-running resumes
(positions that already have crops are skipped).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lasr.sources import ExtractConfig, Extractor, load_sources, select_sources  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(Path(__file__).with_name("sources.yaml")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="source ids to run (ignores enabled/tier)")
    ap.add_argument("--include-sharealike", action="store_true", help="also run CC BY-SA sources (tier sharealike)")
    ap.add_argument("--include-research", action="store_true", help="also run research-only and sharealike sources")
    ap.add_argument("--positions", type=int, default=None, help="override positions per source")
    ap.add_argument("--scale-positions", type=float, default=1.0, help="multiply every source's positions")
    ap.add_argument("--clip-len", type=int, default=6)
    ap.add_argument("--crop", type=int, default=512, help="HR crop size (multiple of 4)")
    ap.add_argument("--threads", type=int, default=8, help="parallel downloads per source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.crop % 4:
        ap.error("--crop must be a multiple of 4")

    try:
        specs = select_sources(load_sources(args.sources), args.only, args.include_sharealike, args.include_research)
    except (KeyError, ValueError) as e:
        ap.error(str(e))

    ex = Extractor(args.out, ExtractConfig(clip_len=args.clip_len, crop=args.crop, threads=args.threads), args.seed)
    total = 0
    for spec in specs:
        if args.positions is not None:
            spec.positions = args.positions
        spec.positions = max(1, round(spec.positions * args.scale_positions))
        t0 = time.time()
        print(f"[{spec.id}] {spec.type} split={spec.split} tier={spec.tier} positions={spec.positions}")
        try:
            total += ex.run_source(spec)
        except Exception as e:  # one dead mirror must not kill the whole run
            print(f"  [{spec.id}] FAILED: {type(e).__name__}: {e}")
        print(f"  [{spec.id}] {time.time() - t0:.0f}s")
    print(f"done: {total} clips in {args.out}")


if __name__ == "__main__":
    main()
