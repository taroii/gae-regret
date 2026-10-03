"""One run of Algorithm 1, instrumented.

Per round the learner draws one action from the exploration mixture
``q_t = (1-alpha) pi_t + alpha Unif(A)``, observes one noisy reward, refits the
local-polynomial slope field on the accumulated data, and advances the policy by
the implicit JKO step

    T(a) = Pi_A( a + gamma * s_hat_t(T(a)) ) ,      pi_{t+1} = T_# pi_t ,

solved per particle by a damped fixed-point iteration.  The policy is carried as
``n_particles`` particles, started on a deterministic lattice so that averages are
quadrature rather than Monte Carlo.

Everything the regret bound needs is logged: the realised field error
``eta_t = ||s_hat_t - grad rbar||_{L2(pi_{t+1})}``, the map residual
``eps_map_t = ||T - Id - gamma s_hat o T||_{L2(pi_t)} / gamma``, and the running
sums that enter the theorems.  Policy regret is computed exactly, since rbar and
``pi* = delta_{a*}`` are known.
"""

from __future__ import annotations

import numpy as np

from .configs import Config
from .estimator import LocalPolyGrid
from .reward import Reward

ENTROPY = 20260206_1724  # fixed project entropy; seeds are spawn keys off this


def _streams(cfg: Config, seed: int):
    import zlib

    cfg_key = zlib.crc32((cfg.stream_name or cfg.name).encode()) % (2**31)
    ss = np.random.SeedSequence(entropy=ENTROPY, spawn_key=(cfg_key, seed))
    init, coin, action, noise = ss.spawn(4)
    return tuple(np.random.default_rng(s) for s in (init, coin, action, noise))


def _init_particles(cfg: Config) -> np.ndarray:
    lo, hi = cfg.pi0
    if cfg.d == 1:
        q = (np.arange(cfg.n_particles) + 0.5) / cfg.n_particles
        return (lo + (hi - lo) * q).reshape(-1, 1)
    k = int(round(np.sqrt(cfg.n_particles)))
    if k * k != cfg.n_particles:
        raise ValueError(f"d=2 needs a square particle count; got {cfg.n_particles}, "
                         f"nearest square is {k * k}")
    q = (np.arange(k) + 0.5) / k
    a, b = np.meshgrid(lo + (hi - lo) * q, lo + (hi - lo) * q, indexing="ij")
    return np.stack([a.ravel(), b.ravel()], axis=1)


def _log_times(T: int) -> np.ndarray:
    """Round indices t such that the elapsed count t+1 is a power of two.

    Logging at elapsed counts 1, 2, 4, ... lets the exponent be read off doubling
    triples (n, 2n, 4n) without interpolating.
    """
    ts = [t for t in (2**k - 1 for k in range(0, 64)) if t < T]
    if ts[-1] != T - 1:
        ts.append(T - 1)
    return np.asarray(ts, dtype=np.int64)


def _field_bound(rew: Reward, lo: float, hi: float, n: int = 2001) -> float:
    """A bound on |grad rbar| over the box, used to cap the fitted field."""
    g = np.linspace(lo, hi, n)
    if rew.d == 1:
        pts = g.reshape(-1, 1)
    else:
        a, b = np.meshgrid(g[:: max(1, n // 200)], g[:: max(1, n // 200)], indexing="ij")
        pts = np.stack([a.ravel(), b.ravel()], axis=1)
    return float(np.max(np.linalg.norm(rew.grad(pts), axis=1)))


def _cap(field_fn, cap: float):
    def wrapped(z):
        f = field_fn(z)
        nrm = np.linalg.norm(f, axis=1, keepdims=True)
        return f * np.minimum(1.0, cap / np.maximum(nrm, 1e-300))

    return wrapped


def _implicit_step(a, field_fn, gamma, lo, hi, n_iter=200, tol=1e-13):
    """Solve ``y = Pi(a + gamma f(y))`` per particle.

    A damped Picard iteration converges only for damping below
    ``2 / (1 + gamma * Lip(f))``, and ``Lip(f)`` blows up at the optimum when
    ``beta < 2``, where a fixed damping settles into a limit cycle instead.  The
    damping is therefore adapted from a finite-difference estimate of the local
    Lipschitz constant, and the iteration runs to a residual tolerance.

    Identical particles are solved once: after a few rounds the policy is a point
    mass, and the field evaluations dominate the runtime.
    """
    uniq, inv = np.unique(a, axis=0, return_inverse=True)
    y = uniq.copy()
    step = max(1e-6, 1e-3 * (hi - lo))
    f = field_fn(y)
    probe = field_fn(np.clip(y + step, lo, hi))
    lip = float(np.max(np.abs(probe - f))) / step
    w = min(0.9, 1.8 / (1.0 + gamma * lip))
    resid = np.max(np.abs(y - np.clip(uniq + gamma * f, lo, hi)))
    for _ in range(n_iter):
        target = np.clip(uniq + gamma * f, lo, hi)
        y_new = y + w * (target - y)
        f_new = field_fn(y_new)
        resid_new = np.max(np.abs(y_new - np.clip(uniq + gamma * f_new, lo, hi)))
        if resid_new > resid and w > 1e-4:
            # the step made things worse: the interpolated field is stiffer here
            # than the probe suggested, so back off instead of cycling
            w *= 0.5
            continue
        y, f, resid = y_new, f_new, resid_new
        if resid < tol:
            break
    f = field_fn(y)
    solver = np.sqrt(np.mean(np.sum((y - np.clip(uniq + gamma * f, lo, hi)) ** 2, axis=1)))
    # the paper's computable residual, which also carries any projection term
    eps_map = np.sqrt(np.mean(np.sum((y - uniq - gamma * f) ** 2, axis=1))) / gamma
    return y[inv], eps_map, solver


def run_one(cfg: Config, seed: int) -> dict[str, np.ndarray]:
    _, rng_coin, rng_act, rng_noise = _streams(cfg, seed)  # particle init is deterministic
    lo, hi = cfg.box
    rew = Reward(d=cfg.d, beta=cfg.beta, lam=cfg.lam,
                 astar=np.full(cfg.d, cfg.astar), cp=cfg.cp, cm=cfg.cm)

    x = _init_particles(cfg)
    delta0 = rew.regret_of_particles(x)
    phi0 = float(np.mean(np.sum((x - rew.astar) ** 2, axis=1)))

    X = np.zeros((cfg.T, cfg.d))
    R = np.zeros(cfg.T)
    is_exp = np.zeros(cfg.T, dtype=bool)

    est = est_exp = None
    if not cfg.oracle:
        est = LocalPolyGrid(cfg.d, cfg.order, lo, hi, cfg.n_grid, cfg.ridge)
        if cfg.track_explore_only:
            est_exp = LocalPolyGrid(cfg.d, cfg.order, lo, hi, cfg.n_grid, cfg.ridge)

    cap = _field_bound(rew, lo, hi) if cfg.cap_field else np.inf
    ref_axis = np.linspace(lo, hi, 65 if cfg.d == 1 else 17)
    ref_pts = (ref_axis.reshape(-1, 1) if cfg.d == 1 else
               np.stack([g.ravel() for g in np.meshgrid(ref_axis, ref_axis, indexing="ij")], 1))
    h_min = 3.0 * (hi - lo) / (cfg.n_grid - 1)
    expo = 1.0 / (2.0 * cfg.beta + cfg.d)
    log_t = set(int(v) for v in _log_times(cfg.T))
    out: dict[str, list] = {k: [] for k in (
        "t", "cum_regret", "eta", "eta_explore_only", "eta_uniform", "eps_map",
        "solver_resid", "phi", "rhs_strong", "rhs_convex", "ratio_max",
        "ratio_at_astar", "spread", "h", "n_explore", "frac_boundary", "order_ok")}

    cum_regret = 0.0
    order_viol = 0  # rounds where the 1-D map failed to preserve particle order
    sum_sq = 0.0   # sum (eps^2 + eta^2)
    sum_lin = 0.0  # sum (eps + eta)
    d_w = (hi - lo) * np.sqrt(cfg.d)

    for t in range(cfg.T):
        # --- one pull from the exploration mixture
        explore = rng_coin.random() < cfg.alpha
        if explore:
            a_t = rng_act.uniform(lo, hi, size=(1, cfg.d))
        else:
            a_t = x[rng_act.integers(0, x.shape[0]), :][None, :]
        X[t] = a_t
        R[t] = rew.value(a_t)[0] + cfg.sigma * rng_noise.normal()
        is_exp[t] = explore

        # --- regret of the committed policy pi_t (scored before the update)
        cum_regret += rew.regret_of_particles(x)

        # --- the field driving the step
        m_t = cfg.alpha * (t + 1)
        h_t = float(np.clip(cfg.c_h * m_t**-expo, h_min, 0.5 * (hi - lo)))
        if cfg.oracle:
            if np.isfinite(cfg.inject_s):
                bias = np.zeros(cfg.d)
                bias[0] = cfg.inject_c * (t + 1) ** (-cfg.inject_s)
            else:
                bias = np.zeros(cfg.d)
            field_fn = lambda z, b=bias: rew.grad(z) + b
        else:
            if not est.maybe_refit(X[: t + 1], R[: t + 1], h_t, cfg.refit_tol):
                est.add(X[t : t + 1], R[t : t + 1])
            # Solve on the whole grid: the fixed-point iteration below may step
            # outside any neighbourhood of the current particles, and reading
            # unsolved coefficients would silently corrupt the field.
            est.solve()
            mask = est.active_mask(x)
            field_fn = _cap(est.interp, cap)

        # --- implicit JKO step
        x_new, eps_map, solver = _implicit_step(x, field_fn, cfg.gamma, lo, hi)
        eta = float(np.sqrt(np.mean(np.sum((field_fn(x_new) - rew.grad(x_new)) ** 2, axis=1))))
        # A genuine failure of monotonicity breaks the non-degeneracy that
        # Assumption 4 asks of the map.  Judge it against the solver residual:
        # particles closer together than that residual scramble for numerical
        # reasons alone, which is not a property of the map.
        if cfg.d == 1 and not bool(np.all(np.diff(x_new[:, 0]) >= -max(1e-12, 10.0 * solver))):
            order_viol += 1
        sum_sq += eps_map**2 + eta**2
        sum_lin += eps_map + eta

        if t in log_t:
            # The committed policy becomes a point mass, so eta_t is a one-point
            # error.  Evaluating the same field error on a fixed uniform reference
            # measure gives a companion number that does not degenerate.
            eta_unif = float(np.sqrt(np.mean(np.sum(
                (field_fn(ref_pts) - rew.grad(ref_pts)) ** 2, axis=1))))
            eta_exp = np.nan
            ratio_max = ratio_astar = np.nan
            if est is not None:
                if est_exp is not None:
                    # Compare at the bandwidth actually in use, not the scheduled
                    # one: a mismatched h alone produces a spurious gap.
                    est_exp.rebuild(X[: t + 1][is_exp[: t + 1]],
                                    R[: t + 1][is_exp[: t + 1]], est.h)
                    est_exp.solve()
                    eta_exp = float(np.sqrt(np.mean(np.sum(
                        (_cap(est_exp.interp, cap)(x_new) - rew.grad(x_new)) ** 2, axis=1))))
                w_tot, lam_min = est.design_diagnostics(mask)
                good = lam_min > 0
                if good.any():
                    ratio_max = float(np.max(w_tot[good] / lam_min[good]))
                astar_mask = est.active_mask(rew.astar[None, :], pad_cells=0)
                w_a, l_a = est.design_diagnostics(astar_mask)
                if (l_a > 0).any():
                    ratio_astar = float(np.max(w_a[l_a > 0] / l_a[l_a > 0]))
            rhs_strong = (delta0 + phi0 / (2 * cfg.gamma) + 2.0 / cfg.lam * sum_sq
                          if cfg.lam > 0 else np.nan)
            rhs_convex = delta0 + phi0 / (2 * cfg.gamma) + d_w * sum_lin
            order_ok = 1.0 - order_viol / (t + 1)  # fraction of rounds with a monotone map
            for k, v in (
                ("t", t + 1), ("cum_regret", cum_regret), ("eta", eta),
                ("eta_explore_only", eta_exp), ("eps_map", eps_map),
                ("solver_resid", solver), ("phi", float(np.mean(np.sum((x_new - rew.astar) ** 2, axis=1)))),
                ("rhs_strong", rhs_strong), ("rhs_convex", rhs_convex),
                ("ratio_max", ratio_max), ("ratio_at_astar", ratio_astar),
                ("spread", float(np.mean(np.var(x_new, axis=0)))),
                ("h", est.h if est is not None else h_t),
                ("eta_uniform", eta_unif),
                ("n_explore", int(is_exp[: t + 1].sum())),
                ("frac_boundary", float(np.mean(np.any(
                    (x_new <= lo + 1e-9) | (x_new >= hi - 1e-9), axis=1)))),
                ("order_ok", float(order_ok)),
            ):
                out[k].append(v)
        x = x_new

    res = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    res["delta0"] = np.asarray([delta0])
    res["phi0"] = np.asarray([phi0])
    return res
