"""Ridge-regularised local-polynomial fit of a reward field on a fixed grid.

The estimator of Algorithm 1 is defined pointwise: at a query point ``a`` it
solves a weighted least-squares problem in the monomials of ``(A_s - a)/h``.  We
evaluate it on a uniform grid and interpolate to the particle locations, which is
what makes a long horizon affordable: the per-grid-point sufficient statistics

    Sigma(x) = sum_s K(u_s) psi(u_s) psi(u_s)^T ,   b(x) = sum_s K(u_s) psi(u_s) R_s,
    u_s = (A_s - x) / h,

are additive in the samples, so a new observation is folded in with a rank-one
update touching only the grid points within ``h`` of it.  The bandwidth is held
fixed between refits, and the statistics are rebuilt from the stored history when
the schedule has moved it far enough (see ``Estimator.maybe_refit``).

Readout.  ``theta[0]`` estimates rbar(x) and ``theta[1:1+d]/h`` estimates
grad rbar(x); the latter is the slope field ``s_hat`` of the paper, which is the
only object the policy update uses.
"""

from __future__ import annotations

import itertools

import numpy as np


def monomial_exponents(d: int, p: int) -> list[tuple[int, ...]]:
    """Exponent tuples up to total degree ``p``, constant first, then e_1..e_d."""
    first = [tuple(1 if j == i else 0 for j in range(d)) for i in range(d)]
    rest = []
    for deg in range(2, p + 1):
        for e in itertools.product(range(deg + 1), repeat=d):
            if sum(e) == deg:
                rest.append(e)
    return [tuple([0] * d)] + first + sorted(rest)


def epanechnikov(u2: np.ndarray) -> np.ndarray:
    """K(u) = (1 - |u|^2)_+ , nonnegative, Lipschitz, supported in the unit ball."""
    return np.clip(1.0 - u2, 0.0, None)


class LocalPolyGrid:
    def __init__(
        self,
        d: int,
        p: int,
        lo: float,
        hi: float,
        n_grid: int,
        ridge: float = 1e-8,
        chunk: int = 512,
    ) -> None:
        if d not in (1, 2):
            raise NotImplementedError("d must be 1 or 2")
        self.d, self.p, self.lo, self.hi = d, p, lo, hi
        self.ridge, self.chunk = ridge, chunk
        self.exps = np.asarray(monomial_exponents(d, p), dtype=np.int64)
        self.m = len(self.exps)
        self.axis = np.linspace(lo, hi, n_grid)
        self.dx = float(self.axis[1] - self.axis[0])
        self.n_axis = n_grid
        self.n_pts = n_grid**d
        if d == 1:
            self.points = self.axis.reshape(-1, 1)
        else:
            gx, gy = np.meshgrid(self.axis, self.axis, indexing="ij")
            self.points = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.h: float | None = None
        self.Sig = np.zeros((self.n_pts, self.m, self.m))
        self.bvec = np.zeros((self.n_pts, self.m))
        self.theta = np.zeros((self.n_pts, self.m))
        self._slope: np.ndarray | None = None

    # ---------------------------------------------------------------- internals
    def _psi(self, u: np.ndarray) -> np.ndarray:
        """Monomial matrix for rescaled offsets ``u`` of shape (n, d)."""
        out = np.ones((u.shape[0], self.m))
        for k, e in enumerate(self.exps):
            if k == 0:
                continue
            col = np.ones(u.shape[0])
            for j in range(self.d):
                if e[j]:
                    col = col * u[:, j] ** e[j]
            out[:, k] = col
        return out

    def _neighbourhood(self, X: np.ndarray):
        """Grid indices within the current bandwidth, and the rescaled offsets."""
        h, W = self.h, int(np.ceil(self.h / self.dx))
        off = np.arange(-W, W + 1)
        if self.d == 1:
            i0 = np.rint((X[:, 0] - self.lo) / self.dx).astype(np.int64)
            idx = i0[:, None] + off[None, :]
            inside = (idx >= 0) & (idx < self.n_axis)
            idx = np.clip(idx, 0, self.n_axis - 1)
            u = (X[:, 0, None] - self.axis[idx]) / h
            u = u[..., None]
            pos = idx
        else:
            i0 = np.rint((X[:, 0] - self.lo) / self.dx).astype(np.int64)
            j0 = np.rint((X[:, 1] - self.lo) / self.dx).astype(np.int64)
            oi, oj = np.meshgrid(off, off, indexing="ij")
            oi, oj = oi.ravel()[None, :], oj.ravel()[None, :]
            ii, jj = i0[:, None] + oi, j0[:, None] + oj
            inside = (ii >= 0) & (ii < self.n_axis) & (jj >= 0) & (jj < self.n_axis)
            ii, jj = np.clip(ii, 0, self.n_axis - 1), np.clip(jj, 0, self.n_axis - 1)
            u = np.stack(
                [
                    (X[:, 0, None] - self.axis[ii]) / h,
                    (X[:, 1, None] - self.axis[jj]) / h,
                ],
                axis=-1,
            )
            pos = ii * self.n_axis + jj
        w = epanechnikov(np.sum(u * u, axis=-1)) * inside
        return pos, u, w

    def _accumulate(self, X: np.ndarray, R: np.ndarray) -> None:
        pos, u, w = self._neighbourhood(X)
        keep = w > 0.0
        if not keep.any():
            return
        pos = pos[keep]
        wts = w[keep]
        P = self._psi(u[keep])
        Rrep = np.repeat(R[:, None], u.shape[1], axis=1)[keep]
        n = self.n_pts
        for i in range(self.m):
            wi = wts * P[:, i]
            self.bvec[:, i] += np.bincount(pos, weights=wi * Rrep, minlength=n)
            for j in range(i, self.m):
                v = np.bincount(pos, weights=wi * P[:, j], minlength=n)
                self.Sig[:, i, j] += v
                if j != i:
                    self.Sig[:, j, i] += v

    # ------------------------------------------------------------------- public
    def add(self, X: np.ndarray, R: np.ndarray) -> None:
        """Fold samples into the statistics at the current bandwidth."""
        for a in range(0, X.shape[0], self.chunk):
            self._accumulate(X[a : a + self.chunk], R[a : a + self.chunk])

    def rebuild(self, X: np.ndarray, R: np.ndarray, h: float) -> None:
        """Reset to bandwidth ``h`` and re-accumulate the whole history."""
        self.h = float(h)
        self.Sig[:] = 0.0
        self.bvec[:] = 0.0
        if X.shape[0]:
            self.add(X, R)

    def maybe_refit(self, X: np.ndarray, R: np.ndarray, h_target: float, tol: float = 0.85) -> bool:
        """Rebuild if the scheduled bandwidth has drifted from the one in use.

        Holding ``h`` fixed between rebuilds keeps the run linear in the horizon and
        still satisfies the schedule up to a constant, which is all the corollary's
        ``h_t asymp M_t^{-1/(2beta+d)}`` requires.
        """
        if self.h is None or h_target < tol * self.h:
            self.rebuild(X, R, h_target)
            return True
        return False

    def solve(self, active: np.ndarray | None = None, min_weight: float | None = None) -> None:
        """Fit every grid point and cache the slope field.

        ``min_weight`` gates the readout: where the total local kernel weight does
        not support the fit (fewer than a couple of effective samples per
        coefficient) the slope is set to zero rather than left to the ridge, which
        would otherwise return an arbitrary direction of enormous magnitude in the
        first rounds and throw the policy across the action set.
        """
        sl = slice(None) if active is None else active
        A = self.Sig[sl] + self.ridge * np.eye(self.m)
        # numpy 2 treats a 2-D right-hand side as a matrix, so solve a stack of
        # column vectors explicitly.
        self.theta[sl] = np.linalg.solve(A, self.bvec[sl][..., None])[..., 0]
        thr = 2.0 * self.m if min_weight is None else min_weight
        self._slope = self.theta[:, 1 : 1 + self.d] / self.h
        self._slope[self.Sig[:, 0, 0] < thr] = 0.0

    def slope_grid(self) -> np.ndarray:
        if self._slope is None:
            raise RuntimeError("call solve() before reading the slope field")
        return self._slope

    def interp(self, x: np.ndarray) -> np.ndarray:
        """(Bi)linear interpolation of the slope field at points ``x``."""
        s = self.slope_grid()
        xc = np.clip(x, self.lo, self.hi - 1e-12)
        g = (xc - self.lo) / self.dx
        i0 = np.floor(g).astype(np.int64)
        i0 = np.clip(i0, 0, self.n_axis - 2)
        f = g - i0
        if self.d == 1:
            a, b = s[i0[:, 0]], s[i0[:, 0] + 1]
            return (1 - f[:, :1]) * a + f[:, :1] * b
        ii, jj = i0[:, 0], i0[:, 1]
        fx, fy = f[:, 0:1], f[:, 1:2]
        n = self.n_axis
        s00, s10 = s[ii * n + jj], s[(ii + 1) * n + jj]
        s01, s11 = s[ii * n + jj + 1], s[(ii + 1) * n + jj + 1]
        return (
            (1 - fx) * (1 - fy) * s00
            + fx * (1 - fy) * s10
            + (1 - fx) * fy * s01
            + fx * fy * s11
        )

    def active_mask(self, x: np.ndarray, pad_cells: int = 3) -> np.ndarray:
        """Grid points needed to interpolate at ``x``, as a boolean mask."""
        mask = np.zeros(self.n_pts, dtype=bool)
        lo_i = np.clip(
            np.floor((x.min(axis=0) - self.lo) / self.dx).astype(int) - pad_cells, 0, self.n_axis - 1
        )
        hi_i = np.clip(
            np.ceil((x.max(axis=0) - self.lo) / self.dx).astype(int) + pad_cells, 0, self.n_axis - 1
        )
        if self.d == 1:
            mask[lo_i[0] : hi_i[0] + 1] = True
        else:
            m2 = mask.reshape(self.n_axis, self.n_axis)
            m2[lo_i[0] : hi_i[0] + 1, lo_i[1] : hi_i[1] + 1] = True
        return mask

    def design_diagnostics(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Total local kernel weight and smallest eigenvalue of Sigma on ``mask``.

        Their ratio is the quantity that controls the bias of the fit: it is bounded
        under exploration alone, and grows if exploitation samples cluster.
        """
        S = self.Sig[mask]
        w_tot = S[:, 0, 0]
        lam_min = np.linalg.eigvalsh(S)[:, 0]
        return w_tot, lam_min
