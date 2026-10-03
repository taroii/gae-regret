#!/usr/bin/env python
"""Run the Bandit--JKO experiments and store the logs.

    python experiments/run_experiments.py smoke
    python experiments/run_experiments.py gate  --seeds 50
    python experiments/run_experiments.py main  --configs B C --seeds 200

Results go to ``results/<config>.npz``: every logged quantity is stacked with one
row per seed, alongside the configuration and the log times.  Seeds are derived
from a fixed project entropy and the configuration name, so a rerun reproduces
the same numbers, and each worker is pinned to one BLAS thread with parallelism
across seeds instead.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bjko.configs import MAIN, SMOKE, Config, gate_configs  # noqa: E402
from bjko.run import run_one  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _work(args: tuple[Config, int]) -> dict[str, np.ndarray]:
    cfg, seed = args
    return run_one(cfg, seed)


def run_config(cfg: Config, seed_range: range, procs: int, tag: str = "",
               skip_existing: bool = False) -> Path | None:
    """Run ``cfg`` over ``seed_range`` and store the stacked logs.

    Seeds are independent, so a long run can be split across machines with
    ``--seed-start/--seed-end`` and a distinct ``--tag`` per shard; ``analyze.py``
    and ``make_figures.py`` read every shard of a configuration together.
    """
    path = RESULTS / f"{cfg.name}{tag}.npz"
    if skip_existing and path.exists():
        print(f"  {cfg.name}{tag}: exists, skipped")
        return path
    t0 = time.time()
    with Pool(processes=procs) as pool:
        runs = pool.map(_work, [(cfg, s) for s in seed_range], chunksize=1)
    seeds = len(seed_range)
    keys = [k for k in runs[0] if k not in ("delta0", "phi0")]
    data = {k: np.stack([r[k] for r in runs]) for k in keys}
    data["delta0"] = np.concatenate([r["delta0"] for r in runs])
    data["phi0"] = np.concatenate([r["phi0"] for r in runs])
    data["config_json"] = np.array(json.dumps(cfg.as_dict()))
    data["predicted_regret_exponent"] = np.array([cfg.regret_exponent])
    data["predicted_eta_exponent"] = np.array([cfg.eta_exponent])
    data["seeds"] = np.asarray(list(seed_range))
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(path, **data)
    el = time.time() - t0
    print(f"  {cfg.name}: {seeds} seeds, T={cfg.T}, {el:.1f}s "
          f"({el / seeds:.2f}s/seed) -> {path.relative_to(RESULTS.parent)}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["smoke", "gate", "main"])
    ap.add_argument("--seeds", type=int, default=None, help="run seeds 0..seeds-1")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="first seed (use with --seed-end to shard a long run)")
    ap.add_argument("--seed-end", type=int, default=None, help="last seed, exclusive")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--T", type=int, default=None, help="override the horizon")
    ap.add_argument("--tag", default="", help="suffix for the output file, for shards")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave finished outputs alone, so a run can be resumed")
    args = ap.parse_args()

    if args.suite == "smoke":
        cfgs, n_default = SMOKE, 2
    elif args.suite == "gate":
        cfgs, n_default = gate_configs(), 50
    else:
        names = args.configs or ["B", "C"]
        cfgs, n_default = {n: MAIN[n] for n in names}, 200
    if args.seed_start is not None or args.seed_end is not None:
        seed_range = range(args.seed_start or 0, args.seed_end
                           if args.seed_end is not None else (args.seeds or n_default))
    else:
        seed_range = range(args.seeds or n_default)

    if args.T is not None:
        cfgs = {k: Config(**{**v.as_dict(), "T": args.T}) for k, v in cfgs.items()}

    print(f"suite={args.suite} seeds={seed_range.start}..{seed_range.stop - 1} "
          f"procs={args.procs}")
    for cfg in cfgs.values():
        run_config(cfg, seed_range, args.procs, args.tag, args.skip_existing)


if __name__ == "__main__":
    main()
