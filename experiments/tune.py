#!/usr/bin/env python
"""Pick the bandwidth constant c_h on tuning seeds that the reported runs never use.

    python experiments/tune.py --config B --T 8000

The schedule itself is fixed by the corollary, h_t propto M_t^{-1/(2 beta + d)}; only
the constant is free.  It is selected on a published grid by mean regret at the
tuning horizon, using seeds offset well past the reported ones, and the chosen
value is then held fixed for every reported run.  An optimum at a grid edge is
reported as such.
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bjko.configs import MAIN, Config  # noqa: E402
from bjko.run import run_one  # noqa: E402

# The schedule's constant only.  Values much above 1.5 are clipped by the
# h <= (hi-lo)/2 cap for most of a run, so they would not be the schedule they claim.
GRID = (0.4, 0.6, 0.8, 1.0, 1.25, 1.5)
TUNING_SEED_OFFSET = 10_000  # reported runs use seeds 0..n-1


def _work(args):
    cfg, seed = args
    r = run_one(cfg, seed)
    return float(r["cum_regret"][-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="B")
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    base = MAIN[args.config]
    jobs, labels = [], []
    for c_h in GRID:
        # share streams across the grid so the comparison is paired
        cfg = Config(**{**base.as_dict(), "c_h": c_h, "T": args.T,
                        "name": f"tune-{args.config}-ch{c_h}",
                        "stream_name": base.name, "track_explore_only": False})
        for s in range(args.seeds):
            jobs.append((cfg, TUNING_SEED_OFFSET + s))
        labels.append(c_h)

    with Pool(processes=args.procs) as pool:
        vals = pool.map(_work, jobs, chunksize=1)
    vals = np.asarray(vals).reshape(len(GRID), args.seeds)

    print(f"config {args.config}: regret at T={args.T}, {args.seeds} tuning seeds "
          f"(offset {TUNING_SEED_OFFSET})")
    print(f"{'c_h':>6} {'mean':>10} {'sd':>9}")
    for c_h, row in zip(GRID, vals):
        print(f"{c_h:>6.2f} {row.mean():>10.4f} {row.std(ddof=1):>9.4f}")
    expo = 1.0 / (2.0 * base.beta + base.d)
    for c_h in GRID:
        h = c_h * (base.alpha * (np.arange(args.T) + 1.0)) ** -expo
        frac = float(np.mean(h > 1.0))
        if frac > 0.05:
            print(f"  warning: c_h={c_h} is clipped by the bandwidth cap in "
                  f"{frac:.0%} of rounds")
    best = int(np.argmin(vals.mean(axis=1)))
    edge = " (GRID EDGE)" if best in (0, len(GRID) - 1) else ""
    print(f"\nchosen c_h = {GRID[best]}{edge}")


if __name__ == "__main__":
    main()
