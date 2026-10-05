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


def _work(args: tuple[Config, int]) -> tuple[int, dict[str, np.ndarray]]:
    cfg, seed = args
    return seed, run_one(cfg, seed)


def _save(path: Path, cfg: Config, done: dict[int, dict[str, np.ndarray]]) -> None:
    """Write every seed finished so far, keyed by seed so a run can resume."""
    seeds = sorted(done)
    runs = [done[s] for s in seeds]
    keys = [k for k in runs[0] if k not in ("delta0", "phi0")]
    data = {k: np.stack([r[k] for r in runs]) for k in keys}
    data["delta0"] = np.concatenate([r["delta0"] for r in runs])
    data["phi0"] = np.concatenate([r["phi0"] for r in runs])
    data["config_json"] = np.array(json.dumps(cfg.as_dict()))
    data["predicted_regret_exponent"] = np.array([cfg.regret_exponent])
    data["predicted_eta_exponent"] = np.array([cfg.eta_exponent])
    data["seeds"] = np.asarray(seeds)
    RESULTS.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def _load_done(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Recover finished seeds from an earlier, possibly interrupted, run."""
    if not path.exists():
        return {}
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return {}
    if "seeds" not in z:
        return {}
    skip = {"config_json", "predicted_regret_exponent", "predicted_eta_exponent", "seeds"}
    keys = [k for k in z.files if k not in skip]
    out: dict[int, dict[str, np.ndarray]] = {}
    for i, seed in enumerate(z["seeds"]):
        out[int(seed)] = {k: (z[k][i:i + 1] if k in ("delta0", "phi0") else z[k][i])
                          for k in keys}
    return out


def run_config(cfg: Config, seed_range: range, procs: int, tag: str = "",
               skip_existing: bool = False) -> Path | None:
    """Run ``cfg`` over ``seed_range`` and store the stacked logs.

    Seeds are independent, so a long run can be split across machines with
    ``--seed-start/--seed-end`` and a distinct ``--tag`` per shard; ``analyze.py``
    and ``make_figures.py`` read every shard of a configuration together.
    """
    path = RESULTS / f"{cfg.name}{tag}.npz"
    wanted = list(seed_range)
    done = _load_done(path) if skip_existing else {}
    todo = [s for s in wanted if s not in done]
    if not todo:
        print(f"  {cfg.name}{tag}: {len(wanted)} seeds already present, skipped")
        return path
    if done:
        print(f"  {cfg.name}{tag}: resuming, {len(done)} seeds already present")
    t0 = time.time()
    with Pool(processes=procs) as pool:
        for n, (seed, run) in enumerate(
                pool.imap_unordered(_work, [(cfg, s) for s in todo], chunksize=1), 1):
            done[seed] = run
            _save(path, cfg, done)
            el = time.time() - t0
            rate = el / n
            print(f"  {cfg.name}{tag}: {n}/{len(todo)} seeds, {el:.0f}s elapsed, "
                  f"{rate * (len(todo) - n):.0f}s remaining", flush=True)
    el = time.time() - t0
    print(f"  {cfg.name}: {len(done)} seeds, T={cfg.T}, {el:.1f}s "
          f"({el / len(todo):.2f}s/seed) -> {path.relative_to(RESULTS.parent)}")
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
