#!/usr/bin/env python
"""Build the paper figures from ``results/*.npz``.

    python experiments/make_figures.py

The numbers live in the tables (see ``make_tables.py``); these two figures carry
only what a number cannot show.

Figure 1, main text: realised regret against the theorem's own right-hand side,
evaluated with the field error and map residual measured in the same run, with one
panel per setting.  Faceting rather than overlaying keeps each panel at two lines,
so no legend box is needed and nothing occludes the curves.

Panels are titled by the parameters that define them, never by an internal letter.
Colours are categorical slots of the validated reference palette, assigned in
fixed order and never cycled, with direct labels so identity never rests on colour.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = RESULTS / "figures"
# Categorical slots, assigned in fixed order and never cycled.  Validated
# all-pairs in light mode: worst CVD dE 9.2, worst normal-vision dE 16.3.  Aqua
# sits below 3:1 against the surface, so every series also carries a direct label.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # blue, orange, aqua, violet
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


def _band(mat: np.ndarray, n_boot: int = 1000, seed: int = 7
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean curve with a 95% bootstrap interval over seeds.

    The generator is created per call, so a band does not depend on how many
    panels were drawn before it.
    """
    rng = np.random.default_rng(seed)
    mean = np.nanmean(mat, axis=0)
    n = mat.shape[0]
    draws = np.stack([np.nanmean(mat[rng.integers(0, n, n)], axis=0) for _ in range(n_boot)])
    return mean, np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def _load(names: list[str]) -> list[tuple[dict, dict]]:
    """Load configurations by name, concatenating the seed rows of every shard."""
    by_name: dict[str, list[Path]] = {}
    for p in sorted(RESULTS.glob("*.npz")):
        if p.stem.startswith("dmc"):
            continue
        cfg = json.loads(str(np.load(p, allow_pickle=False)["config_json"]))
        by_name.setdefault(cfg["name"], []).append(p)
    out = []
    for nm in names:
        if nm not in by_name:
            continue
        zs = [np.load(p, allow_pickle=False) for p in by_name[nm]]
        cfg = json.loads(str(zs[0]["config_json"]))
        data = {k: (np.concatenate([z[k] for z in zs], axis=0) if zs[0][k].ndim >= 1
                    else zs[0][k]) for k in zs[0].files if k != "config_json"}
        out.append((cfg, data))
    return out


def _endlabel(ax, x, y, text, color) -> None:
    ax.annotate(text, xy=(x, y), xytext=(3, 0), textcoords="offset points",
                color=color, fontsize=7.5, va="center", ha="left", clip_on=False)


def _title(cfg: dict) -> str:
    """Panels are identified by their parameters; everything else is caption."""
    return (f"$d={cfg['d']}$,  $\\beta={cfg['beta']}$,  "
            f"$\\lambda={cfg['lam']:g}$")


def figure1(names: list[str]) -> None:
    """Realised regret against the bound, one panel per setting.

    With four settings the panels form a 2x2 that varies one factor at a time
    from the first: smoothness, curvature, then dimension.
    """
    runs = _load(names)
    if not runs:
        return
    if len(runs) > len(SERIES):
        raise ValueError(f"{len(runs)} configurations but {len(SERIES)} colour slots; "
                         "add validated slots or facet further")
    n = len(runs)
    if n == 4:
        fig, axgrid = plt.subplots(2, 2, figsize=(5.6, 4.6))
        axes = axgrid.ravel()
    else:
        fig, axgrid = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
        axes = np.atleast_1d(axgrid)
    for ax, (cfg, z) in zip(axes, runs):
        t = z["t"][0]
        rhs = z["rhs_strong"] if cfg["lam"] > 0 else z["rhs_convex"]
        bm, blo, bhi = _band(rhs)
        rm, rlo, rhi = _band(z["cum_regret"])
        ax.fill_between(t, blo, bhi, color=SERIES[1], alpha=0.16, linewidth=0)
        ax.plot(t, bm, color=SERIES[1])
        ax.fill_between(t, rlo, rhi, color=SERIES[0], alpha=0.16, linewidth=0)
        ax.plot(t, rm, color=SERIES[0])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(_title(cfg), loc="left", color=INK)
        ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    for ax in axes[len(runs):]:
        ax.set_visible(False)
    if n == 4:
        for ax in axgrid[-1]:
            ax.set_xlabel("round $n$")
        for ax in axgrid[:, 0]:
            ax.set_ylabel("cumulative policy regret")
    else:
        for ax in axes:
            ax.set_xlabel("round $n$")
        axes[0].set_ylabel("cumulative policy regret")
    # label the two curves once; the ordering is the same in every panel
    cfg0, z0 = runs[0]
    t0 = z0["t"][0]
    rhs0 = z0["rhs_strong"] if cfg0["lam"] > 0 else z0["rhs_convex"]
    _endlabel(axes[0], t0[-1], rhs0.mean(axis=0)[-1], "bound", SERIES[1])
    _endlabel(axes[0], t0[-1], z0["cum_regret"].mean(axis=0)[-1], "realised", SERIES[0])
    fig.tight_layout()
    _save(fig, "fig1_regret_vs_bound")


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {FIGS / name}.pdf")


# Base setting first, then the runs that change exactly one factor: smoothness,
# curvature, dimension.  File stems are internal; panels are labelled by
# parameters.
PANEL_ORDER = ["B", "A", "C", "D"]


def main() -> None:
    _style()
    found = []
    for p in sorted(RESULTS.glob("*.npz")):
        if p.stem.startswith(("gate-", "smoke-", "dmc")):
            continue
        cfg = json.loads(str(np.load(p)["config_json"]))
        if cfg["name"] not in found:
            found.append(cfg["name"])
    names = [n for n in PANEL_ORDER if n in found] + \
            [n for n in found if n not in PANEL_ORDER]
    figure1(names)


if __name__ == "__main__":
    main()
