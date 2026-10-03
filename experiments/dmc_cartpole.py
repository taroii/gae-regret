#!/usr/bin/env python
"""Algorithm 1 on a real simulator: a parameter-space bandit on cartpole swing-up.

    # landscape scan for the figure (about 2 minutes)
    python experiments/dmc_cartpole.py landscape --grid 21 --episodes 3
    # the algorithm, and the one-point baseline at the same budget
    python experiments/dmc_cartpole.py run --T 1200 --seeds 8

The action set is a two-parameter controller family for dm_control's
``cartpole swingup``,

    a(z) = tanh( w_1(z) * sin(theta) + w_2(z) * theta_dot ),     z in [-1,1]^2,

with the gains obtained from the decision variable by the affine map
``w_1 = 10 + 10 z_1``, ``w_2 = 2.5 + 7.5 z_2``.  A coarse scan of the gain plane
puts the best return near ``w = (17.5, 2.5)``, so this map keeps the action set the
square ``[-1,1]^2`` used elsewhere while placing the optimum in its interior.  The
bandit reward of one pull is the (normalised) return of one 1000-step episode,
which is noisy through the random initial state.  Algorithm 1 then runs
unchanged: a measure over ``w``, an exploration mixture, a local-polynomial fit of
the return gradient, and the implicit JKO step.

This is a demonstration, not a verification.  The return landscape is not concave,
its smoothness exponent is unknown, and the optimum is unknown, so none of the
rate statements apply; nor can the field error be measured, since the true
gradient is unavailable.  What the run shows is the committed policy's return over
rounds, against a one-point baseline given the same number of episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bjko.estimator import LocalPolyGrid  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
BOX = (-1.0, 1.0)
# decision variable z in [-1,1]^2 -> controller gains w (see the module docstring)
GAIN_CENTRE = np.array([10.0, 2.5])
GAIN_SCALE = np.array([10.0, 7.5])
MAX_RETURN = 1000.0  # one unit of reward per step at most, 1000 steps


def gains(z: np.ndarray) -> np.ndarray:
    return GAIN_CENTRE + GAIN_SCALE * np.asarray(z)


def episode_return(z: np.ndarray, seed: int) -> float:
    """Normalised return of one episode under the decision variable ``z``."""
    from dm_control import suite

    env = suite.load("cartpole", "swingup", task_kwargs={"random": int(seed)})
    ts = env.reset()
    w = gains(z)
    total = 0.0
    while not ts.last():
        pos, vel = ts.observation["position"], ts.observation["velocity"]
        sin_th, th_dot = pos[2], vel[1]
        a = np.tanh(w[0] * sin_th + w[1] * th_dot)
        ts = env.step(np.array([a]))
        total += ts.reward or 0.0
    return total / MAX_RETURN


def landscape(grid: int, episodes: int, out: Path) -> None:
    axis = np.linspace(BOX[0], BOX[1], grid)
    vals = np.zeros((grid, grid))
    for i, w1 in enumerate(axis):
        for j, w2 in enumerate(axis):
            vals[i, j] = np.mean([episode_return(np.array([w1, w2]), 1000 + k)
                                  for k in range(episodes)])
        print(f"  row {i + 1}/{grid} done", flush=True)
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(out, axis=axis, value=vals, episodes=np.array([episodes]))
    k = np.unravel_index(np.argmax(vals), vals.shape)
    print(f"landscape -> {out}; best w=({axis[k[0]]:.2f}, {axis[k[1]]:.2f}) "
          f"return={vals[k]:.3f}, worst={vals.min():.3f}")


def _log_times(T: int) -> list[int]:
    ts = [t for t in (2**k - 1 for k in range(64)) if t < T]
    if ts[-1] != T - 1:
        ts.append(T - 1)
    return ts


def _policy_return(x: np.ndarray, rng: np.random.Generator, n_eval: int) -> float:
    """Return of the committed policy, estimated by rolling out sampled particles."""
    idx = rng.integers(0, x.shape[0], n_eval)
    return float(np.mean([episode_return(x[i], int(rng.integers(0, 2**31 - 1)))
                          for i in idx]))


def run_jko(T: int, seed: int, n_particles: int = 256, n_grid: int = 81,
            alpha: float = 0.2, gamma: float = 0.5, c_h: float = 0.5,
            order: int = 2, n_eval: int = 5) -> dict:
    rng = np.random.default_rng(90_000 + seed)
    lo, hi = BOX
    k = int(round(np.sqrt(n_particles)))
    q = (np.arange(k) + 0.5) / k
    a, b = np.meshgrid(lo + (hi - lo) * q, lo + (hi - lo) * q, indexing="ij")
    x = np.stack([a.ravel(), b.ravel()], axis=1)

    est = LocalPolyGrid(2, order, lo, hi, n_grid, ridge=1e-8)
    X = np.zeros((T, 2)); R = np.zeros(T)
    h_min = 3.0 * (hi - lo) / (n_grid - 1)
    logs = {"t": [], "policy_return": [], "pull_return": [], "spread": [], "h": [],
            "ratio_max": [], "mean_w1": [], "mean_w2": []}
    log_t = set(_log_times(T))
    cap = 4.0  # the fitted slope is capped; see the deviations note in the paper

    for t in range(T):
        if rng.random() < alpha:
            w = rng.uniform(lo, hi, size=2)
        else:
            w = x[rng.integers(0, x.shape[0])]
        X[t] = w
        R[t] = episode_return(w, int(rng.integers(0, 2**31 - 1)))

        h_t = float(np.clip(c_h * (alpha * (t + 1)) ** (-1.0 / (2 * 2.5 + 2)), h_min, 0.5 * (hi - lo)))
        if not est.maybe_refit(X[: t + 1], R[: t + 1], h_t):
            est.add(X[t : t + 1], R[t : t + 1])
        est.solve()

        def field(z):
            f = est.interp(z)
            nrm = np.linalg.norm(f, axis=1, keepdims=True)
            return f * np.minimum(1.0, cap / np.maximum(nrm, 1e-300))

        y = x.copy()
        for _ in range(30):
            y = y + 0.6 * (np.clip(x + gamma * field(y), lo, hi) - y)
        x = y

        if t in log_t:
            mask = est.active_mask(x)
            w_tot, lam = est.design_diagnostics(mask)
            good = lam > 0
            logs["t"].append(t + 1)
            logs["policy_return"].append(_policy_return(x, rng, n_eval))
            logs["pull_return"].append(float(R[: t + 1].mean()))
            logs["spread"].append(float(np.mean(np.var(x, axis=0))))
            logs["h"].append(h_t)
            logs["ratio_max"].append(float(np.max(w_tot[good] / lam[good])) if good.any() else np.nan)
            logs["mean_w1"].append(float(x[:, 0].mean()))
            logs["mean_w2"].append(float(x[:, 1].mean()))
    return {k: np.asarray(v, dtype=float) for k, v in logs.items()}


def run_onepoint(T: int, seed: int, alpha: float = 0.2, gamma: float = 0.5,
                 delta: float = 0.25, n_eval: int = 5) -> dict:
    """Route (a): a single-pull one-point estimate translating a point policy."""
    rng = np.random.default_rng(91_000 + seed)
    lo, hi = BOX
    w = np.array([0.0, 0.0])
    logs = {"t": [], "policy_return": [], "pull_return": []}
    log_t = set(_log_times(T))
    run_sum = 0.0
    for t in range(T):
        u = rng.normal(size=2); u /= np.linalg.norm(u)
        probe = np.clip(w + delta * u, lo, hi)
        r = episode_return(probe, int(rng.integers(0, 2**31 - 1)))
        run_sum += r
        g = (2.0 / delta) * r * u          # one-point gradient estimate
        w = np.clip(w + gamma * g / (t + 1) ** 0.5, lo, hi)
        if t in log_t:
            logs["t"].append(t + 1)
            logs["policy_return"].append(float(np.mean(
                [episode_return(w, int(rng.integers(0, 2**31 - 1))) for _ in range(n_eval)])))
            logs["pull_return"].append(run_sum / (t + 1))
    return {k: np.asarray(v, dtype=float) for k, v in logs.items()}


def _work(args):
    kind, T, seed = args
    return kind, seed, (run_jko(T, seed) if kind == "jko" else run_onepoint(T, seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["landscape", "run"])
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.mode == "landscape":
        landscape(args.grid, args.episodes, RESULTS / "dmc_landscape.npz")
        return

    from multiprocessing import Pool

    jobs = [("jko", args.T, s) for s in range(args.seeds)]
    jobs += [("onepoint", args.T, s) for s in range(args.seeds)]
    with Pool(processes=args.procs) as pool:
        got = pool.map(_work, jobs, chunksize=1)
    out = {}
    for kind in ("jko", "onepoint"):
        runs = [r for k, _, r in got if k == kind]
        for key in runs[0]:
            out[f"{kind}_{key}"] = np.stack([r[key] for r in runs])
    out["meta"] = np.array(json.dumps({"T": args.T, "seeds": args.seeds,
                                   "gain_centre": GAIN_CENTRE.tolist(),
                                   "gain_scale": GAIN_SCALE.tolist()}))
    np.savez_compressed(RESULTS / "dmc_cartpole.npz", **out)
    print(f"-> results/dmc_cartpole.npz")
    for kind in ("jko", "onepoint"):
        r = out[f"{kind}_policy_return"]
        print(f"  {kind:9s} committed-policy return: start {r[:,0].mean():.3f} "
              f"-> end {r[:,-1].mean():.3f} (mean over {r.shape[0]} seeds)")


if __name__ == "__main__":
    main()
