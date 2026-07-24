"""Figures for the switch-point analysis.

Every matplotlib figure the analysis produces lives here; the numeric results it draws come
from ``hypnose.modelling.switchpoint`` (fits, comparisons, permutation and autocorrelation
outputs), and the fitted curves are evaluated with ``logistic_p`` / ``model_fitted_p`` from
there. Each public ``plot_*`` builds and returns a Figure; it does not call ``plt.show`` or set
an rc context -- the orchestration script wraps them in ``nature_style`` and shows them.

Convention shared across the strategy / model-comparison plots: **SHORT is the lower row and
LONG the upper row** -- the y axis is inverted, so fitted P(SHORT) curves rise downward.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from hypnose.modelling.switchpoint.data import subject_label
from hypnose.modelling.switchpoint.switch import logistic_p
from hypnose.modelling.switchpoint.compare import MODEL_ORDER
from hypnose.modelling.switchpoint.qlearning import (
    QLEARN_VARIANT_ORDER,
    qlearning_generative_band,
)

# --- style constants ----------------------------------------------------------------------
_SESSION_LINE_COLOR = "tab:blue"
_CONSTANT_COLOR = "#3C5488"
_SWITCH_COLOR = "#E64B35"
_SWITCH2_COLOR = "#8491B4"
_LOGISTIC_COLOR = "#00A087"
_DATA_COLOR = "#2b2b2b"

# Q-learning null: one colour per variant.
#
# TWO curves are drawn per variant and they mean different things (see the qlearning module
# docstring). The GENERATIVE one -- the model run forward on its own choices, averaged over
# simulations -- is what the fitted null actually predicts, so it is solid, full weight, with
# its quantile band shaded. The ONE-STEP-AHEAD one is conditioned on the animal's observed
# choices at every trial; it is what the likelihood scores, but it is not a prediction, and
# with a large fitted kappa it degenerates into a one-trial-lagged copy of the data that would
# appear to track any switch perfectly. It is therefore drawn thin and faint, as a reference.
_QLEARN_COLORS = {"qlearn_free": "#7E6148", "qlearn_constrained": "#F39B7F",
                  "qlearn_perseveration": "#4DBBD5"}
_QLEARN_LINESTYLE = (0, (5, 1.5, 1, 1.5))
# The generative band is usually WIDE -- a free-running Q-learner's transition lands at a
# different trial in every simulation, which is itself a result worth seeing. Three of them
# overlap, so each fill is kept very light and sits *below* the data markers; otherwise the
# panel washes out and the trials underneath become unreadable.
_QLEARN_BAND_ALPHA = 0.07
_QLEARN_BAND_ZORDER = 1.5
_QLEARN_ONESTEP_ALPHA = 0.45
# Individual simulated runs, drawn faint next to the mean. For a perseverative fit each run
# steps abruptly at its own trial, so the mean of many is a smooth ramp that no single run
# resembles -- these are what stop that mean being read as "the model predicts gradual change".
_QLEARN_EXAMPLE_ALPHA = 0.30
_QLEARN_EXAMPLE_LW = 0.7
_QLEARN_SHORT_LABELS = {"qlearn_free": "Q free", "qlearn_constrained": "Q constr",
                        "qlearn_perseveration": "Q persev"}

# Parameter-sweep figure: alpha is mapped to colour (a sequential map, since alpha is ordered)
# and b to linestyle, so a line's two coordinates are readable without a 16-entry legend.
_SWEEP_CMAP = "viridis"
_SWEEP_LINESTYLES = ("-", "--", "-.", ":")

# Reward-identity colours. Trials whose identity cannot be resolved are drawn in grey.
_AB_COLORS = {"A": "#E53935", "B": "#00796B"}
_AB_UNKNOWN_COLOR = "#BDBDBD"

# Trials per bin of the empirical P(SHORT) trace drawn under the model fits.
_ROLLING_WINDOW = 21

# Multi-start diagnostic: y positions of the initial/converged midpoint strip, in data
# coordinates above the LONG row (the axis is inverted, so these are negative).
_MS_Y_INIT = -0.16
_MS_Y_CONV = -0.30

__all__ = [
    "plot_strategy",
    "plot_posterior",
    "plot_model_comparison",
    "plot_multistart",
    "plot_qlearning_sweep",
    "plot_residual_autocorr",
    "plot_permutation",
]


def _rolling_mean(s: np.ndarray, window: int = _ROLLING_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving average of ``s``; returns ``(x, y)``, both empty if ``s`` is too short."""
    if s.size < window:
        return np.zeros(0), np.zeros(0)
    y = np.convolve(s.astype(float), np.ones(window) / window, mode="valid")
    return np.arange(window // 2, window // 2 + y.size), y


def _mark_sessions(ax, session_ends: np.ndarray, label: bool = True) -> None:
    """Draw a blue dotted vertical line at the last trial of each session (a sleep marker)."""
    for i, end in enumerate(session_ends):
        ax.axvline(end, color=_SESSION_LINE_COLOR, linestyle=":", linewidth=0.9, alpha=0.8,
                   zorder=1, label="Session end (sleep)" if (label and i == 0) else None)


def plot_strategy(prep: dict, rewarded_only: bool):
    """Binary SHORT/LONG strategy across the continuous trial axis, coloured by reward identity.

    SHORT is the lower row and LONG the upper row (the y axis is inverted). Every trial --
    SHORT or LONG -- takes the colour of the reward it is associated with.
    """
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ab, trials, s = prep["ab"], prep["trial_ids"], prep["s"]
    for letter, color in _AB_COLORS.items():
        mask = ab == letter
        if mask.any():
            ax.scatter(trials[mask], s[mask], marker="|", s=44, linewidths=0.9, color=color,
                       alpha=0.8, zorder=2, label=f"reward {letter} ({int(mask.sum())})")
    unresolved = ~np.isin(ab, list(_AB_COLORS))
    if unresolved.any():
        ax.scatter(trials[unresolved], s[unresolved], marker="|", s=44, linewidths=0.9,
                   color=_AB_UNKNOWN_COLOR, alpha=0.8, zorder=2,
                   label=f"unresolved ({int(unresolved.sum())})")
    _mark_sessions(ax, prep["session_ends"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_ylim(1.35, -0.35)  # inverted: SHORT on the lower row, LONG on the upper row
    ax.set_xlim(-1, max(prep["n_trials"], 1))
    ax.set_xlabel("Trial (continuous across sessions)")
    kept = "rewarded" if rewarded_only else "completed"
    ax.set_title(f"{subject_label(prep)} - strategy per {kept} trial "
                 f"({prep['n_trials']} trials, {len(prep['session_labels'])} sessions)")
    ax.legend(loc="lower left", fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def plot_posterior(prep: dict, fit: dict, likelihood_window: int):
    """Switch-point posterior, windowed to +/- ``likelihood_window`` trials around the peak."""
    posterior, tau, n = fit["posterior"], fit["tau"], prep["n_trials"]
    hdi_lo, hdi_hi = fit["hdi"]
    fwhm_lo, fwhm_hi = fit["fwhm"]
    lo, hi = max(0, tau - likelihood_window), min(n, tau + likelihood_window + 1)
    x = np.arange(lo, hi)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.fill_between(x, posterior[lo:hi], color=_SWITCH_COLOR, alpha=0.25, zorder=2)
    ax.plot(x, posterior[lo:hi], color=_SWITCH_COLOR, linewidth=1.2, zorder=3)
    ax.axvspan(hdi_lo, hdi_hi, color=_SWITCH_COLOR, alpha=0.10, zorder=1,
               label=f"95% HDI [{hdi_lo}, {hdi_hi}], width {hdi_hi - hdi_lo + 1} trials")
    ax.axvline(tau, color=_DATA_COLOR, linestyle="--", linewidth=1.0, zorder=4,
               label=f"tau = {tau}")
    ax.plot([fwhm_lo, fwhm_hi], [0.5 * posterior.max()] * 2, color=_LOGISTIC_COLOR,
            linewidth=1.6, zorder=4,
            label=f"FWHM [{fwhm_lo}, {fwhm_hi}], width {fwhm_hi - fwhm_lo + 1} trials")
    _mark_sessions(ax, prep["session_ends"], label=False)
    ax.set_xlim(lo - 0.5, hi - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Trial (continuous across sessions)")
    ax.set_ylabel("Posterior P(tau)")
    ax.set_title(f"{subject_label(prep)} - switch-point posterior "
                 f"(+/-{likelihood_window} trials around peak)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def _qlearn_label(variant: str, fit: dict) -> str:
    """Legend entry for one fitted Q-learning variant: its two shape parameters and its BIC."""
    kappa = f", k={fit['kappa']:.2f}" if variant == "qlearn_perseveration" else ""
    flag = " [bound]" if fit.get("boundary_hit") else ""
    return (f"{_QLEARN_SHORT_LABELS[variant]} (null): a={fit['alpha']:.3g}, "
            f"b={fit['b']:.3g}{kappa}, BIC {fit['bic']:.0f}{flag}")


def _overlay_qlearning(ax, x: np.ndarray, qlearning_fits: dict,
                       qlearning_bands: dict | None = None) -> list:
    """Draw each fitted Q-learning variant over the data: generative curve, band, one-step-ahead.

    The generative mean (solid, full weight, with its quantile band shaded) is the model's own
    prediction and is what should be read. A few individual simulated runs are drawn faint
    alongside it, because the mean alone misrepresents a perseverative fit: its runs step
    abruptly at a trial that differs each time, and averaging them yields a smooth ramp that no
    single run resembles. The one-step-ahead curve is drawn thin and faint beneath as a
    reference -- see the ``_QLEARN_*`` constants for why the two must not be confused. Returns
    proxy handles explaining the line styles, for the legend.
    """
    bands = qlearning_bands or {}
    n_sims, n_examples, drew_onestep = 0, 0, False
    for variant in QLEARN_VARIANT_ORDER:
        fit = qlearning_fits.get(variant)
        if fit is None:
            continue
        color = _QLEARN_COLORS[variant]
        band = bands.get(variant) or qlearning_generative_band(fit, x.size)
        # Generative: the model on its own choices. The dominant overlay.
        if band["n_sims"] and np.isfinite(band["mean"]).any():
            lo_q, hi_q = band["quantiles"]
            ax.fill_between(x, band["lo"], band["hi"], color=color,
                            alpha=_QLEARN_BAND_ALPHA, linewidth=0,
                            zorder=_QLEARN_BAND_ZORDER)
            examples = np.asarray(band.get("examples", np.zeros((0, x.size))), dtype=float)
            for run in examples:
                ax.plot(x, run, color=color, linewidth=_QLEARN_EXAMPLE_LW,
                        alpha=_QLEARN_EXAMPLE_ALPHA, zorder=5)
            n_examples = max(n_examples, len(examples))
            ax.plot(x, band["mean"], color=color, linewidth=1.9, zorder=9,
                    label=_qlearn_label(variant, fit))
            n_sims = band["n_sims"]
            _band_pct = f"{lo_q:.0%}-{hi_q:.0%}"
        # One-step-ahead: conditioned on the observed choices. Reference only.
        p = np.asarray(fit["p_short"], dtype=float)
        if p.size == x.size and not np.all(np.isnan(p)):
            ax.plot(x, p, color=color, linewidth=0.8, linestyle=_QLEARN_LINESTYLE,
                    alpha=_QLEARN_ONESTEP_ALPHA, zorder=7)
            drew_onestep = True

    handles = []
    if n_sims:
        handles.append(plt.Line2D([], [], color="#666666", linewidth=1.9,
                                  label=f"(null) solid = generative: model's own\n"
                                        f"choices, mean of {n_sims} sims, {_band_pct} band"))
    if n_examples:
        handles.append(plt.Line2D([], [], color="#666666", linewidth=_QLEARN_EXAMPLE_LW,
                                  alpha=_QLEARN_EXAMPLE_ALPHA,
                                  label=f"(null) hairlines = {n_examples} individual\n"
                                        f"simulated runs (the mean is not one)"))
    if drew_onestep:
        handles.append(plt.Line2D([], [], color="#666666", linewidth=0.8,
                                  linestyle=_QLEARN_LINESTYLE, alpha=_QLEARN_ONESTEP_ALPHA,
                                  label="(null) faint = one-step-ahead:\n"
                                        "conditioned on observed choices"))
    return handles


def plot_model_comparison(prep: dict, comparison: dict, qlearning_fits: dict | None = None,
                          qlearning_bands: dict | None = None):
    """Overlay every fitted model on the data, with the five-row AIC/BIC table in-panel.

    SHORT is the lower row, matching the strategy plot: the y axis is inverted, so the fitted
    P(SHORT) curves rise downward.

    ``qlearning_fits`` (as returned by ``fit_qlearning_variants``) additionally overlays the
    three Q-learning variants -- the mechanistic null -- one colour each, labelled "(null)" in
    the legend to keep them visually apart from the descriptive models. Each is drawn twice:
    the **generative** trajectory solid with its quantile band (what the fitted model predicts),
    and the **one-step-ahead** trajectory thin and faint (what the likelihood scores, which is
    conditioned on the animal's choices and so cannot be read as a prediction). The legend names
    both explicitly. Pass ``None`` to omit them; the table row for the ``qlearning`` entry of
    ``comparison`` is drawn either way.

    ``qlearning_bands`` supplies precomputed ``qlearning_generative_band`` results keyed by
    variant, so a caller that already has them does not pay for the simulations twice; they are
    computed here when omitted.
    """
    s, x, n = prep["s"], prep["trial_ids"], prep["n_trials"]
    constant, switch, switch2, logistic = (comparison["fits"][m] for m in
                                           ("constant", "switch", "switch2", "logistic"))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, s, marker="|", linestyle="none", markersize=6, color=_DATA_COLOR, alpha=0.35, zorder=2)
    roll_x, roll_y = _rolling_mean(s)
    if roll_x.size:
        ax.plot(roll_x, roll_y, color="#999999", linewidth=1.0, zorder=3,
                label=f"Empirical P(SHORT), {_ROLLING_WINDOW}-trial mean")
    ax.axhline(constant["p"], color=_CONSTANT_COLOR, linewidth=1.6, zorder=4,
               label=f"Constant: p = {constant['p']:.2f}")
    ax.step([0, switch["tau"], n - 1], [switch["p1"], switch["p2"], switch["p2"]], where="post",
            color=_SWITCH_COLOR, linewidth=1.8, zorder=5,
            label=f"Switch: tau = {switch['tau']}, {switch['p1']:.2f} -> {switch['p2']:.2f}")
    if np.isfinite(switch2["loglik"]):
        ax.step([0, switch2["tau1"], switch2["tau2"], n - 1],
                [switch2["p1"], switch2["p2"], switch2["p3"], switch2["p3"]], where="post",
                color=_SWITCH2_COLOR, linewidth=1.8, linestyle=":", zorder=6,
                label=f"Switch2: tau = ({switch2['tau1']}, {switch2['tau2']}), "
                      f"{switch2['p1']:.2f} -> {switch2['p2']:.2f} -> {switch2['p3']:.2f}")
    grid = np.linspace(0, max(n - 1, 1), 500)
    ax.plot(grid, logistic_p(grid, logistic["midpoint"], logistic["slope"], logistic["lo"], logistic["hi"]),
            color=_LOGISTIC_COLOR, linewidth=1.8, linestyle="--", zorder=7,
            label=f"Logistic: slope = {logistic['slope']:.3f} "
                  f"(start {logistic.get('start_label', '?')})")
    style_handles = _overlay_qlearning(ax, x, qlearning_fits, qlearning_bands) if qlearning_fits else []
    _mark_sessions(ax, prep["session_ends"])

    best_bic = comparison["best_bic"]
    rows = [f"{'model':<10}{'k':>3}{'AIC':>10}{'BIC':>10}"]
    for m in MODEL_ORDER:
        fit, score = comparison["fits"][m], comparison[m]
        mark = " <- BIC" if m == best_bic else ""
        if not fit.get("implemented", True):
            rows.append(f"{m:<10}{score['k_params']:>3}{'n/a':>10}{'n/a':>10}  (not impl.)")
        else:
            rows.append(f"{m:<10}{score['k_params']:>3}{score['aic']:>10.1f}{score['bic']:>10.1f}{mark}")
    rows.append(f"best: AIC {comparison['best_aic']}, BIC {best_bic}")
    # Bottom-left: the SHORT row before the switch, which is empty by construction.
    ax.text(0.015, 0.03, "\n".join(rows), transform=ax.transAxes, va="bottom", ha="left",
            family="monospace", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax.set_ylim(1.45, -0.35)  # inverted: SHORT on the lower row, LONG on the upper row
    ax.set_xlim(-1, max(n, 1))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_xlabel("Trial (continuous across sessions)")
    ax.set_ylabel("P(SHORT)")
    ax.set_title(f"{subject_label(prep)} - model comparison")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, *style_handles],
              labels=[*labels, *(h.get_label() for h in style_handles)],
              loc="upper right", fontsize=6.5 if qlearning_fits else 7, ncol=2)
    fig.tight_layout()
    return fig


def plot_multistart(prep: dict, fits: list[dict], best: int, warm: int):
    """Data plus every converged sigmoid, one colour per multi-start initial condition.

    The winner (highest loglik) is drawn bold and the switch-point warm start dashed -- both,
    if the warm start won. In the margin above the data each start's INITIAL midpoint (down
    triangle) is joined by a faint connector to where it CONVERGED (circle), so starts that
    funnel into one optimum are visibly distinct from starts that split into basins.
    """
    s, x, n = prep["s"], prep["trial_ids"], prep["n_trials"]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(fits), 2))[:len(fits)])
    grid = np.linspace(0, max(n - 1, 1), 600)

    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.plot(x, s, marker="|", linestyle="none", markersize=6, color=_DATA_COLOR, alpha=0.30, zorder=2)
    roll_x, roll_y = _rolling_mean(s)
    if roll_x.size:
        ax.plot(roll_x, roll_y, color="#999999", linewidth=1.0, zorder=3,
                label=f"Empirical P(SHORT), {_ROLLING_WINDOW}-trial mean")

    for i, (fit, color) in enumerate(zip(fits, colors)):
        is_best, is_warm = i == best, i == warm
        ax.plot(grid, logistic_p(grid, fit["midpoint"], fit["slope"], fit["lo"], fit["hi"]),
                color=color, linewidth=2.6 if is_best else 1.2,
                linestyle="--" if is_warm else "-", alpha=1.0 if is_best else 0.75,
                zorder=6 if is_best else 4,
                label=f"{fit['label']}{' [best]' if is_best else ''}{' [warm]' if is_warm else ''}"
                      f"  LL={fit['loglik']:.1f}, slope={fit['slope']:.3g}")
        # Margin above the data: initial midpoint -> converged midpoint.
        ax.plot([fit["initial_midpoint"], fit["midpoint"]], [_MS_Y_INIT, _MS_Y_CONV],
                color=color, linewidth=0.7, alpha=0.45, zorder=5)
        ax.plot(fit["initial_midpoint"], _MS_Y_INIT, marker="v", markersize=5, color=color,
                alpha=0.9, zorder=6)
        ax.plot(fit["midpoint"], _MS_Y_CONV, marker="o", markersize=5.5, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=7)

    _mark_sessions(ax, prep["session_ends"])
    ax.text(0.002, _MS_Y_INIT, "start ", transform=ax.get_yaxis_transform(), va="center",
            ha="right", fontsize=6, color="#666666")
    ax.text(0.002, _MS_Y_CONV, "converged ", transform=ax.get_yaxis_transform(), va="center",
            ha="right", fontsize=6, color="#666666")
    ax.set_ylim(1.45, _MS_Y_CONV - 0.08)  # inverted, with headroom for the midpoint strip
    ax.set_xlim(-1, max(n, 1))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_xlabel("Trial (continuous across sessions)")
    ax.set_ylabel("P(SHORT)")
    ax.set_title(f"{subject_label(prep)} - logistic multi-start diagnostic "
                 f"({len(fits)} initial conditions)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def plot_qlearning_sweep(prep: dict, variant: str, fit: dict, sweep: list[dict],
                         alphas, bs, band: dict | None = None):
    """One Q-learning variant's ``(alpha, b)`` parameter sweep over the binary trial data.

    Every grid line holds ``Q0`` (and ``kappa``) at their maximum-likelihood values and varies
    only ``alpha`` and ``b``, so the figure shows what those two parameters can and cannot make
    a Q-learner do: **alpha sets how fast the curve rises, b how far it travels**. If no
    combination reaches a step, the null cannot describe an abrupt switch, which is the point of
    drawing it.

    When ``sweep`` carries ``p_generative`` (i.e. ``qlearning_parameter_sweep(...,
    generative=True)``) the grid is drawn **generatively** -- each line the mean of that grid
    point's own simulated runs. That is the version to draw for ``qlearn_perseveration``: with a
    large ``kappa`` held across the grid, every one-step-ahead line collapses to roughly
    ``expit(kappa * s_prev)`` and all 16 would step at exactly the animal's switch trial, for a
    reason that has nothing to do with ``alpha`` or ``b``. The one-step-ahead grid is then kept
    as faint hairlines, since it is what each point's ``nll`` scores. Without ``p_generative``
    the grid is one-step-ahead only, and the title says so.

    ``alpha`` maps to colour (a sequential map, since alpha is ordered) and ``b`` to linestyle,
    each with its own compact legend -- 16 individually labelled lines would be unreadable.

    The maximum-likelihood fit is drawn on top: its **generative** trajectory thick and black
    with its quantile band and a few individual simulated runs (what the fit predicts, and how
    little the mean resembles any one run), and its one-step-ahead trajectory thin and dashed.
    ``band`` supplies a precomputed ``qlearning_generative_band``; it is computed here when
    omitted.
    """
    s, x, n = prep["s"], prep["trial_ids"], prep["n_trials"]
    colors = plt.get_cmap(_SWEEP_CMAP)(np.linspace(0.05, 0.9, max(len(alphas), 2)))

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(x, s, marker="|", linestyle="none", markersize=6, color=_DATA_COLOR, alpha=0.30,
            zorder=2)
    roll_x, roll_y = _rolling_mean(s)
    if roll_x.size:
        ax.plot(roll_x, roll_y, color="#999999", linewidth=1.0, zorder=3,
                label=f"Empirical P(SHORT), {_ROLLING_WINDOW}-trial mean")

    # Generative grid where available; the one-step-ahead grid then drops to faint hairlines.
    has_generative = any("p_generative" in point for point in sweep)
    for point in sweep:
        style = dict(color=colors[point["i_alpha"]],
                     linestyle=_SWEEP_LINESTYLES[point["i_b"] % len(_SWEEP_LINESTYLES)])
        if has_generative and "p_generative" in point:
            ax.plot(x, point["p_generative"], linewidth=1.1, alpha=0.85, zorder=5, **style)
        ax.plot(x, point["p_short"], linewidth=0.7 if has_generative else 0.9,
                alpha=_QLEARN_EXAMPLE_ALPHA if has_generative else 0.7, zorder=4, **style)
    ml_band = band if band is not None else qlearning_generative_band(fit, n)
    ml_handles = []
    if ml_band["n_sims"] and np.isfinite(ml_band["mean"]).any():
        lo_q, hi_q = ml_band["quantiles"]
        ax.fill_between(x, ml_band["lo"], ml_band["hi"], color=_DATA_COLOR,
                        alpha=_QLEARN_BAND_ALPHA, linewidth=0, zorder=7)
        handle, = ax.plot(x, ml_band["mean"], color=_DATA_COLOR, linewidth=2.4, zorder=9,
                          label=f"ML fit, generative\na={fit['alpha']:.3g}, b={fit['b']:.3g}\n"
                                f"mean of {ml_band['n_sims']} sims\n({lo_q:.0%}-{hi_q:.0%} band)")
        ml_handles.append(handle)
        examples = np.asarray(ml_band.get("examples", np.zeros((0, x.size))), dtype=float)
        for run in examples:
            ax.plot(x, run, color=_DATA_COLOR, linewidth=_QLEARN_EXAMPLE_LW,
                    alpha=_QLEARN_EXAMPLE_ALPHA, zorder=6)
        if len(examples):
            ml_handles.append(plt.Line2D([], [], color=_DATA_COLOR,
                                         linewidth=_QLEARN_EXAMPLE_LW,
                                         alpha=_QLEARN_EXAMPLE_ALPHA,
                                         label=f"{len(examples)} individual runs\n"
                                               f"(the mean is not one)"))
    onestep = np.asarray(fit["p_short"], dtype=float)
    if onestep.size == x.size and not np.all(np.isnan(onestep)):
        handle, = ax.plot(x, onestep, color=_DATA_COLOR, linewidth=1.0,
                          linestyle=_QLEARN_LINESTYLE, alpha=0.7, zorder=8,
                          label=f"ML fit, one-step-ahead\n(conditioned on observed\n"
                                f"choices) nll={fit['nll']:.1f}")
        ml_handles.append(handle)
    _mark_sessions(ax, prep["session_ends"], label=False)

    # Two proxy legends: one for the alpha colours, one for the b linestyles. Neither draws
    # data -- they only decode the 4 x 4 grid.
    alpha_handles = [plt.Line2D([], [], color=colors[i], linewidth=1.6, label=f"alpha = {a:g}")
                     for i, a in enumerate(alphas)]
    b_handles = [plt.Line2D([], [], color="#666666", linewidth=1.4,
                            linestyle=_SWEEP_LINESTYLES[i % len(_SWEEP_LINESTYLES)],
                            label=f"b = {b:g}") for i, b in enumerate(bs)]
    first = ax.legend(handles=[*ml_handles, *alpha_handles], loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=False,
                      title="alpha: learning rate", title_fontsize=7)
    ax.add_artist(first)
    ax.legend(handles=b_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=7,
              frameon=False, title="b: inverse temp.", title_fontsize=7)

    ax.set_ylim(1.45, -0.35)  # inverted: SHORT on the lower row, LONG on the upper row
    ax.set_xlim(-1, max(n, 1))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_xlabel("Trial (continuous across sessions)")
    ax.set_ylabel("P(SHORT)")
    held = (f"Q0 = ({fit['q0_short']:.2f}, {fit['q0_long']:.2f})"
            + (f", kappa = {fit['kappa']:.2f}" if variant == "qlearn_perseveration" else ""))
    # Name which grid is drawn. The one-step-ahead grid carries the kappa * s_prev term, so with
    # a non-zero kappa held at ML it spikes on every choice flip -- that is the curve tracking
    # the lagged data, not plot noise, and it is exactly why the generative grid exists.
    grid_note = ("grid lines are generative (each the mean of its own sims); hairlines are the "
                 "one-step-ahead grid, which each point's nll scores"
                 if has_generative else
                 "grid lines are one-step-ahead, pairing with each point's nll")
    ax.set_title(f"{subject_label(prep)} - {variant} parameter sweep "
                 f"({len(alphas)} x {len(bs)} grid; {held} held at ML)\n{grid_note}",
                 fontsize=9)
    fig.tight_layout()
    # tight_layout sizes the axes without seeing the two out-of-axes legends, so their titles
    # would be clipped; reserve the right margin for them explicitly afterwards.
    fig.subplots_adjust(right=0.83)
    return fig


def _plot_acf_panel(ax, lags: np.ndarray, acf: np.ndarray, bound: np.ndarray, color: str,
                    title: str) -> None:
    """Stem plot of one ACF with its (per-lag) significance band shaded."""
    finite = np.isfinite(acf)
    ax.fill_between(lags, -bound, bound, color=color, alpha=0.15, linewidth=0,
                    label="~95% band (+/-1.96/sqrt(N))")
    ax.vlines(lags[finite], 0, acf[finite], color=color, linewidth=1.1, zorder=3)
    ax.plot(lags[finite], acf[finite], marker="o", linestyle="none", markersize=3.5,
            color=color, zorder=4)
    ax.axhline(0, color=_DATA_COLOR, linewidth=0.8, zorder=2)
    ax.set_ylabel("Residual ACF")
    ax.set_title(title, fontsize=8)
    ax.legend(loc="upper right", fontsize=7)


def plot_residual_autocorr(prep: dict, best_name: str, lags: np.ndarray, acf_full: np.ndarray,
                           bound_full: np.ndarray, acf_within: np.ndarray,
                           bound_within: np.ndarray):
    """Two stacked ACF panels: all trial pairs, then within-session pairs only."""
    fig, (ax_full, ax_within) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    _plot_acf_panel(ax_full, lags, acf_full, bound_full, _SWITCH_COLOR,
                    f"All trial pairs (residuals from {best_name})")
    _plot_acf_panel(ax_within, lags, acf_within, bound_within, _LOGISTIC_COLOR,
                    "Within-session pairs only (cross-session lags dropped)")
    ax_within.set_xlabel("Lag (trials)")
    ax_within.set_xlim(0, lags[-1] + 1)
    fig.suptitle(f"{subject_label(prep)} - residual autocorrelation "
                 f"(BIC-best model: {best_name})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _box_with_points(ax, groups: dict) -> None:
    """One box per group with its individual points jittered on top."""
    labels = list(groups.keys())
    data = [np.asarray(groups[label], dtype=float) for label in labels]
    ax.boxplot(data, positions=range(1, len(labels) + 1), widths=0.5, showfliers=False,
               medianprops=dict(color=_SWITCH_COLOR, linewidth=1.6),
               boxprops=dict(color=_DATA_COLOR), whiskerprops=dict(color=_DATA_COLOR),
               capprops=dict(color=_DATA_COLOR))
    rng = np.random.default_rng(0)  # jitter only; no effect on the values plotted
    for i, values in enumerate(data, start=1):
        if values.size:
            ax.scatter(i + rng.uniform(-0.11, 0.11, values.size), values, s=22,
                       facecolor=_SESSION_LINE_COLOR, edgecolors=_DATA_COLOR, linewidths=0.6,
                       alpha=0.75, zorder=4)
    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)


def _null_distribution(ax, null_means: np.ndarray, observed_mean: float, p_value: float) -> None:
    """Null distribution of the permutation mean, with the observed mean marked."""
    ax.hist(null_means, bins=40, color=_SESSION_LINE_COLOR, alpha=0.65, edgecolor="white", linewidth=0.4)
    ax.axvline(observed_mean, color=_SWITCH_COLOR, linewidth=1.8,
               label=f"observed {observed_mean:.1f}\np = {p_value:.4f}")
    ax.set_xlabel("Mean f (donated)")
    ax.set_ylabel("Permutations")
    ax.legend(loc="upper right", fontsize=7)


def plot_permutation(real_f: np.ndarray, shuffled_f: np.ndarray, null_means: np.ndarray,
                     observed_mean: float, p_value: float, n_permutations: int,
                     n_included: int, inclusion: str):
    """Two-panel sleep-alignment figure: real vs shuffled ``f`` boxes, and the paired null.

    **Left**: two boxplots with the points overlaid -- real ``f`` (one point per included
    animal) and the span-guarded pool of shuffled ``f`` (one per valid recipient x donor pair).
    **Right**: the paired-permutation null distribution of the mean ``f``, observed mean marked.
    """
    fig, (ax_box, ax_null) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    _box_with_points(ax_box, {"f": real_f, "Shuffled f": shuffled_f})
    ax_box.set_ylabel("Trials from session start (f)")
    ax_box.set_title(f"observed mean f = {observed_mean:.1f}\np = {p_value:.4f} (one-sided)",
                     fontsize=8)
    _null_distribution(ax_null, null_means, observed_mean, p_value)
    ax_null.set_title(f"Paired-permutation null\n({n_permutations} permutations)", fontsize=8)
    fig.suptitle(f"Switch alignment to sleep (n = {n_included} animals, "
                 f"inclusion: {inclusion})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
