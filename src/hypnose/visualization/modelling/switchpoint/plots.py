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

# --- style constants ----------------------------------------------------------------------
_SESSION_LINE_COLOR = "tab:blue"
_CONSTANT_COLOR = "#3C5488"
_SWITCH_COLOR = "#E64B35"
_SWITCH2_COLOR = "#8491B4"
_LOGISTIC_COLOR = "#00A087"
_DATA_COLOR = "#2b2b2b"

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


def plot_model_comparison(prep: dict, comparison: dict):
    """Overlay every fitted model on the data, with the five-row AIC/BIC table in-panel.

    SHORT is the lower row, matching the strategy plot: the y axis is inverted, so the fitted
    P(SHORT) curves rise downward. Unimplemented models (``qlearning``) appear in the table
    but have no curve to draw.
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
    ax.legend(loc="upper right", fontsize=7, ncol=2)
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
