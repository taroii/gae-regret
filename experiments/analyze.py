#!/usr/bin/env python
"""Fit rate exponents and check the regret bound.

    python experiments/analyze.py                 # all results/*.npz
    python experiments/analyze.py B C

Estimand.  The theorems bound *expected* regret, so exponents are fitted to the
mean curve over seeds, not to per-seed slopes (whose average is a different
quantity by Jensen).

Exponent.  From doubling-window increments,

    e(t) = log2 [ (Reg_4t - Reg_2t) / (Reg_2t - Reg_t) ],

which cancels the O(1) start-up offset that biases a log-log fit of the
cumulative curve downwards.  We report the last available window and a 95%
bootstrap interval over seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_BOOT = 2000
RNG = np.random.default_rng(12345)


def _window_exponents(t: np.ndarray, reg: np.ndarray) -> list[tuple[int, float]]:
    """Local exponent on every doubling triple (n, 2n, 4n).

    A single triple is not a rate: adjacent windows can differ by several tenths
    while the curve is still leaving its transient, so the whole sequence is
    reported and the figures plot it.
    """
    idx = {int(v): i for i, v in enumerate(t)}
    out = []
    for v in t:
        v = int(v)
        if 2 * v in idx and 4 * v in idx:
            d1 = reg[idx[2 * v]] - reg[idx[v]]
            d2 = reg[idx[4 * v]] - reg[idx[2 * v]]
            if d1 > 0 and d2 > 0:
                out.append((v, float(np.log2(d2 / d1))))
    return out


def _increment_slope(t: np.ndarray, reg: np.ndarray) -> float:
    """Exponent from a fit to all doubling increments: log(Reg_2n - Reg_n) ~ e log n.

    This uses every window rather than the last triple, and the O(1) start-up
    offset drops out of the increments.
    """
    idx = {int(v): i for i, v in enumerate(t)}
    xs, ys = [], []
    for v in t:
        v = int(v)
        if 2 * v in idx and v >= 32:
            d = reg[idx[2 * v]] - reg[idx[v]]
            if d > 0:
                xs.append(np.log(v)); ys.append(np.log(d))
    if len(xs) < 4:
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def _dyadic_exponent(t: np.ndarray, reg: np.ndarray) -> float:
    w = _window_exponents(t, reg)
    return w[-1][1] if w else np.nan


def _loglog_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Slope over the last decade and a half, skipping the start-up rounds."""
    keep = (t >= max(64.0, t.max() / 30.0)) & (y > 0)
    if keep.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log(t[keep]), np.log(y[keep]), 1)[0])


def _boot(stat, mat: np.ndarray, t: np.ndarray, reduce=np.mean,
          seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap over seeds, with its own generator.

    A shared module-level generator would make an interval depend on how many
    configurations were analysed before it.
    """
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        v = stat(t, reduce(mat[pick], axis=0))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def _group_shards(paths: list[Path]) -> dict[str, list[Path]]:
    """Group ``<config><tag>.npz`` files by the configuration they belong to."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        cfg = json.loads(str(np.load(p, allow_pickle=False)["config_json"]))
        groups.setdefault(cfg["name"], []).append(p)
    return groups


def _stack_shards(paths: list[Path]) -> tuple[dict, dict[str, np.ndarray]]:
    """Concatenate the per-seed rows of several shards of one configuration."""
    zs = [np.load(p, allow_pickle=False) for p in sorted(paths)]
    cfg = json.loads(str(zs[0]["config_json"]))
    lens = {z["t"].shape[1] for z in zs}
    if len(lens) != 1:
        raise ValueError(f"{cfg['name']}: shards logged different horizons {lens}")
    data = {}
    for k in zs[0].files:
        if k in ("config_json",):
            continue
        arrs = [z[k] for z in zs]
        data[k] = np.concatenate(arrs, axis=0) if arrs[0].ndim >= 1 else arrs[0]
    return cfg, data


def analyze(paths: list[Path]) -> dict:
    cfg, z = _stack_shards(paths)
    t = z["t"][0]
    reg, eta = z["cum_regret"], z["eta"]
    e_pred = float(z["predicted_regret_exponent"][0])
    eta_pred = float(z["predicted_eta_exponent"][0])

    e_hat = _increment_slope(t, reg.mean(axis=0))
    e_lo, e_hi = _boot(_increment_slope, reg, t, seed=11)
    e_last = _dyadic_exponent(t, reg.mean(axis=0))
    windows = _window_exponents(t, reg.mean(axis=0))
    # the assumption is stated on E||.||^2, so fit the root-mean-square curve
    rms = lambda m, axis: np.sqrt(np.mean(m**2, axis=axis))
    eta_hat = _loglog_slope(t, rms(eta, 0))
    eta_lo, eta_hi = _boot(_loglog_slope, eta, t, reduce=rms, seed=22)

    # bound validity: realised mean regret against the theorem's right-hand side
    rhs = z["rhs_strong"] if cfg["lam"] > 0 else z["rhs_convex"]
    rhs_mean, reg_mean = rhs.mean(axis=0), reg.mean(axis=0)
    ok = bool(np.all(reg_mean <= rhs_mean + 1e-9))
    tightness = float(np.nanmax(reg_mean / np.where(rhs_mean > 0, rhs_mean, np.nan)))

    out = dict(
        name=cfg["name"], d=cfg["d"], beta=cfg["beta"], lam=cfg["lam"],
        T=cfg["T"], seeds=int(reg.shape[0]), shards=len(paths),
        e_pred=e_pred, e_hat=e_hat, e_ci=(e_lo, e_hi), e_last_window=e_last,
        window_exponents=windows,
        eta_pred=eta_pred, eta_hat=eta_hat, eta_ci=(eta_lo, eta_hi),
        bound_holds=ok, bound_tightness=tightness,
        eps_map_max=float(np.nanmax(z["eps_map"])),
        solver_max=float(np.nanmax(z["solver_resid"])),
        ratio_early=float(np.nanmean(z["ratio_max"][:, int(np.argmax(t >= 64))])),
        ratio_last=float(np.nanmean(z["ratio_max"][:, -1])),
        eta_last=float(np.sqrt(np.mean(eta[:, -1] ** 2))),
        eta_uniform_last=float(np.sqrt(np.mean(z["eta_uniform"][:, -1] ** 2)))
        if "eta_uniform" in z else float("nan"),
        spread_last=float(np.mean(z["spread"][:, -1])),
        h_last=float(np.mean(z["h"][:, -1])),
        eta_explore_only_last=float(np.nanmean(z["eta_explore_only"][:, -1])),
        monotone_frac=float(np.nanmin(z["order_ok"][:, -1])),
        frac_boundary_max=float(np.nanmax(z["frac_boundary"])),
    )
    return out


def main() -> None:
    names = sys.argv[1:]
    paths = [p for p in sorted(RESULTS.glob("*.npz")) if not p.stem.startswith("dmc")]
    groups = _group_shards(paths)
    if names:
        groups = {k: v for k, v in groups.items() if k in names}
    rows = [analyze(v) for v in groups.values()]
    if not rows:
        print("no results found"); return

    print("\n=== rate fits (exponents are upper bounds; below prediction is consistent) ===")
    hdr = f"{'cfg':>12} {'d':>2} {'beta':>5} {'lam':>4} {'seeds':>6} " \
          f"{'reg pred':>9} {'reg fit':>9} {'95% CI':>18} {'eta pred':>9} {'eta fit':>9} {'95% CI':>18}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        e_ci = "[{:.3f}, {:.3f}]".format(*r["e_ci"])
        eta_ci = "[{:.3f}, {:.3f}]".format(*r["eta_ci"])
        print(f"{r['name']:>12} {r['d']:>2} {r['beta']:>5.1f} {r['lam']:>4.1f} {r['seeds']:>6} "
              f"{r['e_pred']:>9.3f} {r['e_hat']:>9.3f} {e_ci:>18} "
              f"{r['eta_pred']:>9.3f} {r['eta_hat']:>9.3f} {eta_ci:>18}")

    print("\n=== local regret exponent per doubling window (n: e) ===")
    for r in rows:
        ws = "  ".join(f"{n}:{e:+.2f}" for n, e in r["window_exponents"][-6:])
        print(f"{r['name']:>12}  pred {r['e_pred']:.3f}   {ws}")

    print("\n=== bound validity and diagnostics ===")
    for r in rows:
        print(f"{r['name']:>12}  bound holds: {str(r['bound_holds']):>5}   "
              f"max regret/RHS: {r['bound_tightness']:.3g}   "
              f"eps_map<= {r['eps_map_max']:.1e}   solver<= {r['solver_max']:.1e}")
        print(f"{'':>12}  design ratio W/lam_min: {r['ratio_early']:.3g} -> {r['ratio_last']:.3g}   "
              f"eta(all)={r['eta_last']:.4g} vs eta(explore-only)={r['eta_explore_only_last']:.4g}"
              f"  eta(uniform ref)={r['eta_uniform_last']:.4g}")
        print(f"{'':>12}  policy spread {r['spread_last']:.2e} (a point mass once this is ~0)   "
              f"h in use {r['h_last']:.3f}")
        print(f"{'':>12}  monotone-map rounds: {r['monotone_frac']:.4f}   "
              f"max boundary mass: {r['frac_boundary_max']:.3g}")

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
