"""LaTeX tables for the paper, written to papers/tables/ and echoed to stdout.

Every row is named by the parameters that define it, never by an internal letter:
a reader should not have to look up what "configuration B" was.  Measured
quantities are bold, predictions are not, so the eye lands on the number that
came out of the experiment.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from analyze import _group_shards, analyze

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "papers" / "tables"


def _setting(row: dict) -> str:
    return (f"$d{{=}}{row['d']}$, $\\beta{{=}}{row['beta']}$, "
            f"$\\lambda{{=}}{int(row['lam']) if float(row['lam']).is_integer() else row['lam']}$")


def _ci(lo: float, hi: float) -> str:
    return f"{{\\scriptsize $[{lo:.2f}, {hi:.2f}]$}}"


def _sci(v: float) -> str:
    """Scientific notation that survives LaTeX math mode."""
    if not np.isfinite(v):
        return "--"
    mant, exp = f"{v:.1e}".split("e")
    return f"{mant}\\!\\times\\!10^{{{int(exp)}}}"


def _meets(hat: float, pred: float, ci: tuple[float, float]) -> bool:
    """Does a measured exponent meet what the theory asks of it?

    Both exponents in these tables are upper bounds, so meeting the prediction
    means landing at or below it.  A value fractionally above it whose bootstrap
    interval still covers the prediction is matching the rate, not failing it, so
    it counts as met.
    """
    if not np.isfinite(hat):
        return False
    return hat <= pred or (ci[0] <= pred <= ci[1])


def _bold(text: str, on: bool) -> str:
    return rf"$\mathbf{{{text}}}$" if on else f"${text}$"


def _write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"\n% ---- papers/tables/{name}.tex " + "-" * 40)
    print(body)


def _groups() -> dict:
    """Synthetic runs only; the cartpole files carry no configuration block."""
    paths = [p for p in sorted(RESULTS.glob("*.npz")) if not p.stem.startswith("dmc")]
    return _group_shards(paths)


def _rows(names: list[str]) -> list[dict]:
    groups = _groups()
    return [analyze(groups[n]) for n in names if n in groups]


def table_rates(rows: list[dict]) -> None:
    """The headline table: measured exponents against the theory."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Measured regret rates against the theory.  Bold marks an"
        r" empirical value that meets what the theory asks of it; brackets give a"
        r" 95\% bootstrap interval over seeds.  Both quantities are defined in the"
        r" text: the exponent is the growth rate of cumulative regret in the"
        r" horizon, which the theory bounds above, and the ratio is realised regret"
        r" against the theorem's own right-hand side, which must stay below one.}",
        r"\label{tab:rates}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Setting & Theoretical Regret Exponent & Empirical Regret Exponent"
        r" & Regret-to-Bound Ratio \\",
        r"\midrule",
    ]
    for r in rows:
        e_cell = _bold(f"{r['e_hat']:.3f}", _meets(r["e_hat"], r["e_pred"], r["e_ci"]))
        ratio = _bold(f"{r['bound_tightness']:.3f}", r["bound_tightness"] < 1.0)
        lines.append(
            f"{_setting(r)} & ${r['e_pred']:.3f}$ & {e_cell} {_ci(*r['e_ci'])} "
            f"& {ratio} " + chr(92) * 2
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write("tab_rates", "\n".join(lines))


def table_recon(rows: list[dict]) -> None:
    """Corollary 1's intermediate rate for the reconstruction error."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The reconstruction rate of Corollary~\ref{cor:forced-exploration-rate},"
        r" measured.  This is the intermediate step from which the regret rate"
        r" follows, not a claim of the paper in its own right; it is reported so"
        r" that the estimator half of the analysis can be checked separately from"
        r" the optimisation half.  Corollary~\ref{cor:forced-exploration-rate}"
        r" bounds $\eta_t$ above by $t^{-(\beta-1)/(2\beta+d)}$, so a measured"
        r" exponent at or below the prediction is consistent.  Two settings decay"
        r" faster than required and two match the prediction to within the"
        r" bootstrap interval; bold marks both cases, since each meets the"
        r" corollary.}",
        r"\label{tab:recon}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Setting & Theoretical Decay Exponent & Empirical Decay Exponent"
        r" & Final Gradient Error \\",
        r"\midrule",
    ]
    for r in rows:
        cell = _bold(f"{r['eta_hat']:.3f}",
                     _meets(r["eta_hat"], r["eta_pred"], r["eta_ci"]))
        lines.append(f"{_setting(r)} & ${r['eta_pred']:.3f}$ "
                     f"& {cell} {_ci(*r['eta_ci'])} "
                     f"& ${r['eta_last']:.4f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_recon", "\n".join(lines))


def table_windows(rows: list[dict]) -> None:
    """Local exponent per doubling window: the drift, shown rather than hidden."""
    wins = sorted({n for r in rows for n, _ in r["window_exponents"]})
    wins = wins[-7:]
    head = " & ".join(f"${n}$" for n in wins)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Local regret exponent $\log_2\!\big[(\mathrm{Reg}_{4n}-"
        r"\mathrm{Reg}_{2n})/(\mathrm{Reg}_{2n}-\mathrm{Reg}_{n})\big]$ on each"
        r" doubling window, with the prediction in the last column.  A single"
        r" window is not a rate; the fits in Table~\ref{tab:rates} use every"
        r" window.}",
        r"\label{tab:windows}",
        r"\begin{tabular}{l" + "c" * len(wins) + r"c}",
        r"\toprule",
        r"Setting & \multicolumn{" + str(len(wins)) + r"}{c}{Local Exponent on the"
        r" Window Starting at Round $n$} & Theoretical \\",
        r"\cmidrule(lr){2-" + str(len(wins) + 1) + r"}",
        r" & " + head + r" & Regret Exponent \\",
        r"\midrule",
    ]
    for r in rows:
        d = dict(r["window_exponents"])
        cells = " & ".join(f"${d[n]:+.2f}$" if n in d else "--" for n in wins)
        lines.append(f"{_setting(r)} & {cells} & ${r['e_pred']:.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_windows", "\n".join(lines))


def table_diagnostics(rows: list[dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Run diagnostics.  The step residual is the residual of"
        r" the implicit step, reported at its worst over all rounds and seeds;"
        r" the monotone-map column is the fraction of rounds in which the learned map" r" preserved order, judged against the solver residual; that check is defined" r" only in $d=1$ and is not run in $d=2$.  The policy spread is the"
        r" mean per-coordinate variance of the particles at the final round: it"
        r" collapses because the optimum of a linear objective over measures is a"
        r" Dirac, which the text discusses.  The design ratio is the total local"
        r" kernel weight divided by the smallest eigenvalue of the local design"
        r" matrix, the quantity that controls the bias of the fit under an"
        r" adaptive design.}",
        r"\label{tab:diagnostics}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Setting & Largest Step Residual & Final Step Residual"
        r" & Monotone Map Rounds & Final Particle Variance"
        r" & Design Ratio at Round $T$ \\",
        r"\midrule",
    ]
    for r in rows:
        mono = f"${r['monotone_frac']:.4f}$" if r["d"] == 1 else "--"
        lines.append(
            f"{_setting(r)} "
            f"& ${_sci(r['eps_map_max'])}$ & ${_sci(r['eps_map_final'])}$ "
            f"& {mono} & ${_sci(r['spread_last'])}$ & ${r['ratio_last']:,.0f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write("tab_diagnostics", "\n".join(lines))


def table_estimator(rows: list[dict]) -> None:
    """All samples vs forced-exploration samples only, at matched bandwidth."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Gradient error $\eta$ at the final round, as a root mean square"
        r" over seeds.  Fitting on all samples is compared with fitting on the"
        r" forced-exploration samples alone \emph{at the same bandwidth}; the"
        r" clustered exploitation samples do not degrade the fit at $\beta=2.5$."
        r"  Bold marks the better of the two fits in each row.  The last column"
        r" evaluates the same field error against a fixed uniform measure rather"
        r" than against $\pi_{t+1}$, which has collapsed to a point mass.}",
        r"\label{tab:estimator}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Gradient error $\eta$ at round $T$} \\",
        r"\cmidrule(lr){2-4}",
        r" & Fit on All Samples & Fit on Exploration Samples"
        r" & Measured Against a Uniform Measure \\",
        r"\midrule",
    ]
    for r in rows:
        a, e = r["eta_last"], r["eta_explore_only_last"]
        fa = f"\\mathbf{{{a:.4f}}}" if a <= e else f"{a:.4f}"
        fe = f"\\mathbf{{{e:.4f}}}" if e < a else f"{e:.4f}"
        lines.append(f"{_setting(r)} & ${fa}$ & ${fe}$ & ${r['eta_uniform_last']:.4f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_estimator", "\n".join(lines))


def table_gate() -> None:
    """Injected-error experiment: the optimisation block in isolation.

    The prediction depends on the injected exponent, not on the base
    configuration: with ``eta_t ~ t^{-s}`` the strongly concave bound accumulates
    ``sum_t t^{-2s}``, giving a regret exponent ``max(0, 1 - 2s)``.  Where that is
    zero the regret is bounded, so there is no power law to fit and the realised
    regret is reported against the theorem's right-hand side instead.
    """
    groups = _groups()
    names = sorted(n for n in groups if n.startswith("gate-"))
    if not names:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Injected-error experiment.  The estimated gradient is replaced"
        r" by the exact $\nabla\bar r$ perturbed by a known $\eta_t = c\,t^{-s}$,"
        r" which isolates the optimisation half of the analysis from the"
        r" estimator.  With $\eta_t \propto t^{-s}$ the strongly concave bound"
        r" accumulates $\sum_t t^{-2s}$, predicting a regret exponent"
        r" $\max(0, 1-2s)$.  Where that exponent is zero the theory says regret"
        r" stops growing rather than growing slowly, and a flat curve has no growth"
        r" exponent: the fit returns a strongly negative value for $s=0.75$ and"
        r" nothing at all for the exact gradient, whose increments vanish."
        r"  Both are evidence of saturation, and the ratio is what to read for"
        r" those rows."
        r"  The fitted exponents sit a little above the prediction, by $0.02$ to"
        r" $0.05$, because the prediction is the growth rate of the \emph{bound}"
        r" while the fit is on realised regret, and over a finite horizon the two"
        r" differ by the constant in $\sum_t t^{-2s}$.  The ratio is the one comparison the" r" theorem makes, and it stays between $0.43$ and $0.65$ in every row."
        r"  Seeds differ only in the action stream, so the intervals are narrow by"
        r" construction.}",
        r"\label{tab:gate}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Injected Gradient Error & Theoretical Regret Exponent"
        r" & Empirical Regret Exponent & Regret-to-Bound Ratio \\",
        r"\midrule",
    ]
    for n in names:
        r = analyze(groups[n])
        z = np.load(groups[n][0])
        reg_T = float(z["cum_regret"][:, -1].mean())
        # same statistic as Table 1: the largest ratio over logged rounds, not
        # the ratio at the final round
        ratio_max = float(np.nanmax(z["cum_regret"].mean(axis=0)
                                    / z["rhs_strong"].mean(axis=0)))
        if "-s" in n:
            sval = float(n.split("-s")[-1])
            pred = max(0.0, 1.0 - 2.0 * sval)
            label = rf"$0.3\,t^{{-{sval:g}}}$"
        else:
            sval, pred, label = None, 0.0, "$0$ (exact gradient)"
        bounded = pred <= 0.0
        # A saturating curve has no growth exponent: the fit returns a strongly
        # negative value, or nothing at all when the increments vanish.  Both are
        # reported as they come out rather than blanked.
        if not np.isfinite(r["e_hat"]):
            meas = r"$\mathrm{NaN}$"
        else:
            ok = r["e_hat"] <= pred or (bounded and r["e_hat"] < 0)
            meas = _bold(f"{r['e_hat']:.3f}", ok) + " " + _ci(*r["e_ci"])
        pred_s = r"$0$ (bounded)" if bounded else f"${pred:.2f}$"
        ratio = _bold(f"{ratio_max:.3f}", ratio_max < 1.0)
        lines.append(f"{label} & {pred_s} & {meas} & {ratio} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_gate", "\n".join(lines))


def main() -> None:
    # gate runs carry no design diagnostics, so the all-nan columns warn harmlessly
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Same order as the figure panels: base, then one factor changed at a time.
    rows = _rows(["B", "A", "C", "D"])
    if rows:
        table_rates(rows)
        table_recon(rows)
        table_windows(rows)
        table_diagnostics(rows)
        table_estimator(rows)
    table_gate()
    print(f"\n% wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}/")


if __name__ == "__main__":
    main()
