"""Reward families with a known smoothness exponent and known optimum.

The family used throughout is

    rbar(a) = -(lam/2) |a - astar|^2
              - sum_j [ cp * ((a_j - astar_j)_+)^beta + cm * ((a_j - astar_j)_-)^beta ],

which is concave on R^d, lam-strongly concave when lam > 0, and exactly C^beta at
astar and no smoother.  Two properties of this choice matter for the experiments:

* beta is non-integer.  A local polynomial of order p = floor(beta) reproduces
  polynomials exactly, so an integer beta (a pure quadratic, say) would make the
  Taylor bias vanish identically and the observed rate would be pure variance.
* cp != cm.  A symmetric kink has an odd-symmetric remainder, whose contribution
  to the fitted *slope* cancels at astar under a symmetric design -- precisely
  where the policy concentrates.  The asymmetry keeps the bias term alive.

The maximum is attained at astar with value 0, so the instantaneous policy regret
of a policy pi is simply -E_pi[rbar].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Reward:
    d: int
    beta: float
    lam: float
    astar: np.ndarray
    cp: float = 1.0
    cm: float = 0.5

    def __post_init__(self) -> None:
        if self.beta <= 1.0:
            raise ValueError("beta must exceed 1")
        if float(self.beta).is_integer():
            raise ValueError("beta must be non-integer (see module docstring)")
        if self.astar.shape != (self.d,):
            raise ValueError("astar has the wrong shape")

    def value(self, x: np.ndarray) -> np.ndarray:
        """rbar(x) for x of shape (..., d)."""
        u = x - self.astar
        quad = -0.5 * self.lam * np.sum(u * u, axis=-1)
        up = np.clip(u, 0.0, None)
        um = np.clip(-u, 0.0, None)
        kink = -np.sum(self.cp * up**self.beta + self.cm * um**self.beta, axis=-1)
        return quad + kink

    def grad(self, x: np.ndarray) -> np.ndarray:
        """grad rbar(x) for x of shape (..., d); returns shape (..., d)."""
        u = x - self.astar
        up = np.clip(u, 0.0, None)
        um = np.clip(-u, 0.0, None)
        b = self.beta
        return -self.lam * u - b * (self.cp * up ** (b - 1.0) - self.cm * um ** (b - 1.0))

    @property
    def max_value(self) -> float:
        return 0.0

    def regret_of_particles(self, x: np.ndarray) -> float:
        """Instantaneous policy regret of the empirical measure on x: F(pi*) - F(pi)."""
        return float(self.max_value - np.mean(self.value(x)))
