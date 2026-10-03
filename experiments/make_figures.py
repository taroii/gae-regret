#!/usr/bin/env python
"""Build the paper figures from ``results/*.npz``.

    python experiments/make_figures.py

Figure 1 is the main-text figure: (a) regret normalised by the predicted rate, so
a flat or falling curve means the run is consistent with the bound and no
reference line has to be fitted; (b) realised regret against the theorem's own
right-hand side, evaluated with the measured field error and map residual.

Appendix figures: the field error against its predicted decay, with the
exploration-only fit for comparison; the design ratio that controls the bias of
the fit; and the injected-residual study.

Colours are categorical slots 1-3 of the validated reference palette, assigned in
fixed order and never cycled, with direct end-labels so identity never rests on
colour alone.
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


def figure1(names: list[str]) -> None:
    runs = _load(names)
    if not runs:
        return
    if len(runs) > len(SERIES):
        raise ValueError(f"{len(runs)} configurations but {len(SERIES)} colour slots; "
                         "add validated slots or facet instead of cycling")
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(9.6, 2.7))

    for i, (cfg, z) in enumerate(runs):
        t = z["t"][0]
        e = float(z["predicted_regret_exponent"][0])
        mean, lo, hi = _band(z["cum_regret"] / t**e)
        c = SERIES[i]
        axa.plot(t, mean, color=c, label=f"{cfg['name']}  $d={cfg['d']}$, $\\lambda={cfg['lam']:g}$")
        axa.fill_between(t, lo, hi, color=c, alpha=0.16, linewidth=0)
        _endlabel(axa, t[-1], mean[-1], cfg["name"], c)
    axa.set_xscale("log")
    axa.set_xlabel("round $n$")
    axa.set_ylabel(r"$\mathrm{Reg}_n \, / \, n^{\,e_{\mathrm{pred}}}$")
    axa.set_title("(a) regret against the predicted rate", loc="left", color=INK)
    axa.legend(loc="upper left")

    for i, (cfg, z) in enumerate(runs):
        t = z["t"][0]
        rhs = z["rhs_strong"] if cfg["lam"] > 0 else z["rhs_convex"]
        c = SERIES[i]
        reg, _, _ = _band(z["cum_regret"])
        bnd, _, _ = _band(rhs)
        axb.plot(t, reg, color=c, label=f"{cfg['name']}: regret")
        axb.plot(t, bnd, color=c, linestyle="--", linewidth=1.2, label=f"{cfg['name']}: bound")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("round $n$")
    axb.set_ylabel("cumulative policy regret")
    axb.set_title("(b) bound with measured residuals", loc="left", color=INK)
    axb.legend(loc="upper left", ncol=1)

    # (c) the local exponent on every doubling window, which is what a rate claim
    # rests on; a single window is not a rate.
    for i, (cfg, z) in enumerate(runs):
        t = z["t"][0]
        reg = z["cum_regret"].mean(axis=0)
        idx = {int(v): k for k, v in enumerate(t)}
        xs, es = [], []
        for v in t:
            v = int(v)
            if 2 * v in idx and 4 * v in idx:
                d1 = reg[idx[2 * v]] - reg[idx[v]]
                d2 = reg[idx[4 * v]] - reg[idx[2 * v]]
                if d1 > 0 and d2 > 0:
                    xs.append(v); es.append(np.log2(d2 / d1))
        c = SERIES[i]
        axc.plot(xs, es, color=c, marker="o", markersize=3.5, label=cfg["name"])
        axc.axhline(float(z["predicted_regret_exponent"][0]), color=c,
                    linestyle="--", linewidth=1.0)
        if xs:
            _endlabel(axc, xs[-1], es[-1], cfg["name"], c)
    axc.set_xscale("log")
    axc.set_xlabel("window start $n$")
    axc.set_ylabel("local exponent")
    axc.set_title("(c) exponent per doubling window", loc="left", color=INK)
    axc.legend(loc="upper right")

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_regret.{ext}")
    plt.close(fig)
    print(f"  fig1_regret.pdf  ({', '.join(c['name'] for c, _ in runs)})")


def figure2(names: list[str]) -> None:
    runs = _load(names)
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, (cfg, z) in enumerate(runs):
        t = z["t"][0]
        c = SERIES[i]
        mean, lo, hi = _band(np.sqrt(z["eta"] ** 2))
        ax.plot(t, mean, color=c, label=f"{cfg['name']}: all rounds")
        ax.fill_between(t, lo, hi, color=c, alpha=0.16, linewidth=0)
        if np.isfinite(z["eta_explore_only"]).any():
            m2, _, _ = _band(z["eta_explore_only"])
            ax.plot(t, m2, color=c, linestyle=":", linewidth=1.3,
                    label=f"{cfg['name']}: exploration only")
        keep = t >= max(64, t.max() / 30)
        ref = mean[keep][0] * (t[keep] / t[keep][0]) ** float(z["predicted_eta_exponent"][0])
        ax.plot(t[keep], ref, color=INK2, linestyle="--", linewidth=1.0,
                label="predicted slope" if i == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("round $n$"); ax.set_ylabel(r"$\eta_n=\|\widehat s_n-\nabla\bar r\|_{L^2(\pi_{n+1})}$")
    ax.set_title("field error", loc="left", color=INK)
    ax.legend(loc="lower left")
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig2_eta.{ext}")
    plt.close(fig)
    print("  fig2_eta.pdf")


def figure3(names: list[str]) -> None:
    runs = _load(names)
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, (cfg, z) in enumerate(runs):
        t = z["t"][0]
        c = SERIES[i]
        mean, lo, hi = _band(z["ratio_max"])
        keep = t >= 64
        ax.plot(t[keep], mean[keep], color=c, label=cfg["name"])
        ax.fill_between(t[keep], lo[keep], hi[keep], color=c, alpha=0.16, linewidth=0)
        _endlabel(ax, t[-1], mean[-1], cfg["name"], c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("round $n$")
    ax.set_ylabel(r"$\max_a\ \sum_s K / \lambda_{\min}(\widehat\Sigma(a))$")
    ax.set_title("design ratio controlling the fit's bias", loc="left", color=INK)
    ax.legend(loc="upper left")
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig3_design_ratio.{ext}")
    plt.close(fig)
    print("  fig3_design_ratio.pdf")


def figure4() -> None:
    """Injected-residual study: fitted regret exponent against the injected decay."""
    import re

    paths = sorted(RESULTS.glob("gate-*-s*.npz"))
    if not paths:
        return
    ss, fitted, pred = [], [], []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        cfg = json.loads(str(z["config_json"]))
        s = float(re.search(r"s([0-9.]+)$", cfg["name"]).group(1))
        t, reg = z["t"][0], z["cum_regret"].mean(axis=0)
        idx = {int(v): i for i, v in enumerate(t)}
        e = np.nan
        for v in t:
            v = int(v)
            if 2 * v in idx and 4 * v in idx:
                d1 = reg[idx[2 * v]] - reg[idx[v]]
                d2 = reg[idx[4 * v]] - reg[idx[2 * v]]
                if d1 > 0 and d2 > 0:
                    e = float(np.log2(d2 / d1))
        ss.append(s); fitted.append(e)
        pred.append(max(0.0, 1.0 - 2.0 * s) if cfg["lam"] > 0 else max(0.0, 1.0 - s))
    order = np.argsort(ss)
    ss, fitted, pred = np.asarray(ss)[order], np.asarray(fitted)[order], np.asarray(pred)[order]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(ss, pred, color=INK2, linestyle="--", linewidth=1.0, marker="s", markersize=4,
            label="predicted")
    ax.plot(ss, fitted, color=SERIES[0], marker="o", markersize=5, label="fitted")
    ax.set_xlabel("injected decay $s$ in $\\eta_t=c\\,t^{-s}$")
    ax.set_ylabel("regret exponent")
    ax.set_title("optimisation block in isolation", loc="left", color=INK)
    ax.legend(loc="upper right")
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig4_injected.{ext}")
    plt.close(fig)
    print("  fig4_injected.pdf")


def main() -> None:
    _style()
    names = sorted({json.loads(str(np.load(p, allow_pickle=False)["config_json"]))["name"]
                    for p in RESULTS.glob("*.npz")
                    if not p.stem.startswith(("gate", "smoke", "dmc"))})
    print("figures:")
    figure1(names); figure2(names); figure3(names); figure4()


if __name__ == "__main__":
    main()
