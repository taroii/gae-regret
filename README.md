# GAE Regret

Experiments for *Regret Bounds for Wasserstein–JKO Policy Optimization under
Bandit Feedback*.

## Install

```
conda env create -f environment.yml
conda activate gaeexp
```

`dm_control` and `mujoco` are only needed for the cartpole experiment; the
synthetic experiments need numpy and matplotlib alone.

## Run

```
make smoke      # end-to-end check, a few seeds        (~1 min)
make gate       # Experiment 3 + correctness gate      (~15 min)
make main       # configs B, C (d=1)
make main-d2    # config D (d=2)
make dmc        # cartpole: Algorithm 1 vs one-point
make figures    # analysis table + all figures
make overnight  # everything above, in order
```

Variables: `PY` (interpreter), `SEEDS` (200), `SEEDS_D2` (100), `T` (horizon
override). Longer horizons are recommended for the rate figures:

```
make main SEEDS=200 T=300000
make main-d2 SEEDS_D2=100 T=100000
```

Seeds are independent and reproducible from the seed index, so a run can be split
across machines with a distinct tag per shard; `analyze.py` and `make_figures.py`
merge the shards automatically.

```
python experiments/run_experiments.py main --configs B --seed-start 0 --seed-end 100 --tag -sh0
```

`--skip-existing` leaves finished outputs alone, so an interrupted run resumes by
re-issuing the same command.

## Output

* `results/<config>.npz` — one row per seed for every logged quantity
* `results/summary.json` — fitted exponents and diagnostics
* `results/figures/*.pdf` — one figure per file

Figures regenerate from `results/` without re-running anything (`make figures`).

## Layout

```
experiments/bjko/           reward, estimator, JKO loop, configs
experiments/run_experiments.py   run a suite over seeds
experiments/analyze.py           exponents, bootstrap CIs, bound checks
experiments/make_figures.py      figures
experiments/tune.py              bandwidth-constant sweep
experiments/dmc_cartpole.py      cartpole swingup demo
```
