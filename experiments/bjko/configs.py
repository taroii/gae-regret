"""Experiment configurations.

The predicted exponents are those of the paper's Corollary (rate under constant
forced exploration):

    strongly concave (lam > 0):  (d + 2) / (2 beta + d)
    concave          (lam = 0):  (beta + d + 1) / (2 beta + d)

``eta`` decays at ``-(beta - 1) / (2 beta + d)``.  Both are upper bounds, so an
observed exponent below the prediction is consistent with the theory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Config:
    name: str
    d: int
    beta: float
    lam: float
    T: int = 30_000
    alpha: float = 0.1
    sigma: float = 0.1
    gamma: float = 0.5
    c_h: float = 1.0
    ridge: float = 1e-8
    n_particles: int = 2048
    n_grid: int = 401
    p: int | None = None
    cp: float = 1.0
    cm: float = 0.5
    astar: float = 0.3
    pi0: tuple[float, float] = (-0.8, 0.0)
    box: tuple[float, float] = (-1.0, 1.0)
    # Experiment 3: replace the fitted field by the true one plus a known error
    oracle: bool = False
    inject_c: float = 0.0
    inject_s: float = float("inf")
    # Experiment 4: also fit on the exploration rounds alone, for comparison
    track_explore_only: bool = True
    # Safeguard: cap the fitted field at a bound on |grad rbar| over the box.  At
    # the first few rounds the local fit is unconstrained and can return an
    # arbitrarily large slope; capping it cannot increase the field error, since
    # the true gradient lies inside the cap.  This is the projection Pi_L of the
    # exploration-only variant of the draft, applied here as a numerical
    # safeguard and reported as a deviation.
    cap_field: bool = True
    # Rebuild the statistics once the scheduled bandwidth has drifted this far from
    # the one in use.  Close to 1 keeps the bandwidth on schedule, at the cost of
    # more rebuilds; a loose tolerance leaves a sawtooth in h that shows up in eta.
    refit_tol: float = 0.97
    # Streams are keyed on this name when set, so variants of one configuration
    # (a bandwidth sweep, say) can share common random numbers.
    stream_name: str | None = None

    @property
    def order(self) -> int:
        return int(np.floor(self.beta)) if self.p is None else self.p

    @property
    def regret_exponent(self) -> float:
        if self.lam > 0:
            return (self.d + 2.0) / (2.0 * self.beta + self.d)
        return (self.beta + self.d + 1.0) / (2.0 * self.beta + self.d)

    @property
    def eta_exponent(self) -> float:
        return -(self.beta - 1.0) / (2.0 * self.beta + self.d)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- main configs
MAIN = {
    "B": Config(name="B", d=1, beta=2.5, lam=1.0, T=30_000),
    "C": Config(name="C", d=1, beta=2.5, lam=0.0, T=30_000),
    "A": Config(name="A", d=1, beta=1.5, lam=1.0, T=30_000),
    "D": Config(name="D", d=2, beta=2.5, lam=1.0, T=20_000, n_grid=121, n_particles=2025),
}

# ------------------------------------------- Experiment 3: injected residuals
def gate_configs(base: str = "B", ss: tuple[float, ...] = (0.1, 0.25, 0.4, 0.75)) -> dict[str, Config]:
    b = MAIN[base]
    out = {
        f"gate-{base}-oracle": Config(
            **{**b.as_dict(), "name": f"gate-{base}-oracle", "stream_name": b.name,
               "oracle": True,
               "inject_c": 0.0, "inject_s": float("inf"), "T": 20_000,
               "track_explore_only": False}
        )
    }
    for s in ss:
        out[f"gate-{base}-s{s}"] = Config(
            **{**b.as_dict(), "name": f"gate-{base}-s{s}", "stream_name": b.name,
               "oracle": True,
               "inject_c": 0.3, "inject_s": s, "T": 20_000,
               "track_explore_only": False}
        )
    return out


SMOKE = {
    "smoke-B": Config(name="smoke-B", d=1, beta=2.5, lam=1.0, T=2_000, n_particles=512, n_grid=201),
    "smoke-D": Config(name="smoke-D", d=2, beta=2.5, lam=1.0, T=1_000, n_particles=529, n_grid=81),
}
