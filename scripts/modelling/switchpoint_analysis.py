#!/usr/bin/env python
"""Switch-point analysis of the LONG -> SHORT strategy change, per animal.

Two independent entry points, each callable from a notebook (import, call, get results and
figure handles back) or from the terminal via the argparse wrapper at the bottom:

- ``run_analysis``    -- per-animal switch-point fit, posterior, and model comparison.
- ``run_permutation`` -- do switches sit closer to *real* sleep boundaries than to other
  animals' donated ones?

Neither depends on the other having run; both build their trial sequences through the same
``_prepare_subject`` helper. All data access, filtering, and plotting live here; the numeric
model is in ``hypnose.models.switchpoint_helpers``.

Trials are read from the ``trial_data.parquet`` written by trial classification, so run that
first. A trial is kept when ``is_aborted == False`` (and, with ``rewarded_only``, when
``response_time_category == "rewarded"``), and it scores 1 (SHORT) when
``hidden_rule_success`` is truthy, else 0 (LONG). Kept trials are re-indexed 0..n-1
continuously across sessions -- note ``trial_data``'s own ``global_trial_id`` restarts at 0
each session, so it is used only to order trials *within* a session.

Examples
--------
  python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 --likelihood-window 100
  python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 41 --date-range 20251201 20251231 --rewarded-only
  python scripts/modelling/switchpoint_analysis.py permutation --subjids 40 41 42 --rewarded-only
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hypnose.io.paths import get_derivatives_root
from hypnose.io.save import nature_style
from hypnose.models.switchpoint_helpers import (
    compare_models,
    distance_to_session_start,
    fit_switchpoint,
    logistic_p,
)
from hypnose.qc.validate import validate_subject
from hypnose.utils.helpers import _filter_session_dirs, _iter_subject_dirs
from hypnose.visualization.visualization_utils import _load_trial_views

# Truthy spellings of hidden_rule_success: bool in parquet, str via the CSV fallback.
# Mirrors the coercion the visualization helpers use.
_HR_TRUE = ("true", "1", "1.0")

_SESSION_LINE_COLOR = "tab:blue"
_CONSTANT_COLOR = "#3C5488"
_SWITCH_COLOR = "#E64B35"
_LOGISTIC_COLOR = "#00A087"
_DATA_COLOR = "#2b2b2b"

# Trials per bin of the empirical P(SHORT) trace drawn under the model fits.
_ROLLING_WINDOW = 21

# Attempts to build a without-replacement donor assignment before allowing replacement.
_ASSIGNMENT_TRIES = 20

# Which animals count as "has a switch" in run_permutation. Keys are the `inclusion` values.
_INCLUSION_RULES = {
    # Strictest: the 3-parameter switch beats BOTH the constant and the 4-parameter
    # logistic on BIC, i.e. the change is real *and* abrupt rather than a slow drift.
    "bic_switch_wins": lambda c: c["best_bic"] == "switch",
    # The change is real, but a gradual (logistic) description may fit it even better.
    "bic_beats_constant": lambda c: c["switch"]["bic"] < c["constant"]["bic"],
    # Same as bic_switch_wins but under the more permissive AIC penalty.
    "aic_switch_wins": lambda c: c["best_aic"] == "switch",
    # No filtering; every animal contributes a tau.
    "all": lambda c: True,
}

__all__ = ["run_analysis", "run_permutation"]


def _normalize_subjids_dates(subjids, dates):
    """Normalize the ``(subjids, date_ranges)`` inputs, supporting a ``{subjid: date_range}``
    dict passed as ``subjids``.

    Reimplements ``hypnose.visualization.sing_rew._normalize_subjids_dates`` (that module
    currently fails to import; see this script's README).
    """
    if isinstance(subjids, dict):
        dates = subjids if (dates is None or not isinstance(dates, dict)) else dates
        subjids = list(subjids.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif not isinstance(subjids, (list, tuple)):
        subjids = [subjids]

    def dates_for(subjid):
        if not isinstance(dates, dict):
            return dates
        if subjid in dates:
            return dates[subjid]
        try:
            if int(subjid) in dates:
                return dates[int(subjid)]
        except (TypeError, ValueError):
            pass
        return dates.get(str(subjid))

    return subjids, dates, dates_for


def _short_mask(td: pd.DataFrame) -> pd.Series:
    """Boolean mask of SHORT-sequence trials (``hidden_rule_success`` truthy)."""
    if "hidden_rule_success" not in td.columns:
        return pd.Series(False, index=td.index)
    return td["hidden_rule_success"].astype(str).str.lower().isin(_HR_TRUE)


def _prepare_subject(
    subjid: int,
    date_range: Optional[Union[Sequence[Union[int, str]], tuple]] = None,
    rewarded_only: bool = False,
    derivatives_dir: Optional[Path] = None,
) -> dict:
    """Build one animal's continuous SHORT/LONG trial sequence and its sleep markers.

    Concatenates the kept trials of every session in ``date_range``, in date order, and
    re-indexes them 0..n-1 so the trial axis is continuous across sessions. Shared by both
    entry points -- all filtering lives here.

    Parameters
    ----------
    subjid : int
        Subject id.
    date_range : None | tuple[start, end] | iterable of dates
        Inclusive ``YYYYMMDD`` range or explicit date list. ``None`` = all sessions.
    rewarded_only : bool
        Additionally require ``response_time_category == "rewarded"``.
    derivatives_dir : Path | None
        Derivatives root; defaults to the resolved project root.

    Returns
    -------
    dict
        ``subjid``, ``trial_ids`` (0..n-1), ``s`` (1 = SHORT, 0 = LONG), ``session_ends``
        (global trial id of the LAST kept trial of each session -- the sleep markers),
        ``session_starts`` (the trial after each sleep period; always starts at 0),
        ``session_labels`` (dates), ``session_index`` (session of each trial),
        ``session_sizes``, ``n_trials``, ``subject_dir``.

    Raises
    ------
    FileNotFoundError
        No derivatives directory for ``subjid``.
    """
    subj_dirs = [d for _, d in _iter_subject_dirs(derivatives_dir, [subjid])]
    if not subj_dirs:
        raise FileNotFoundError(f"No derivatives directory for subject {subjid}")
    subj_dir = subj_dirs[0]

    segments, labels, sizes = [], [], []
    for ses_dir in _filter_session_dirs(subj_dir, date_range):
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue
        views = _load_trial_views(results_dir)
        td = views["rewarded"] if rewarded_only else views["completed"]
        if td.empty:
            continue
        if "global_trial_id" in td.columns:
            td = td.sort_values("global_trial_id")
        segments.append(_short_mask(td).to_numpy(dtype=np.int8))
        labels.append(ses_dir.name.split("_date-")[-1])
        sizes.append(len(td))

    s = np.concatenate(segments) if segments else np.zeros(0, dtype=np.int8)
    sizes_arr = np.asarray(sizes, dtype=int)
    session_ends = np.cumsum(sizes_arr) - 1 if sizes_arr.size else np.zeros(0, dtype=int)
    session_starts = np.concatenate(([0], session_ends[:-1] + 1)) if sizes_arr.size else np.zeros(0, dtype=int)
    return {
        "subjid": subjid,
        "trial_ids": np.arange(s.size),
        "s": s,
        "session_ends": session_ends,
        "session_starts": session_starts,
        "session_labels": labels,
        "session_index": np.repeat(np.arange(sizes_arr.size), sizes_arr),
        "session_sizes": sizes_arr,
        "n_trials": int(s.size),
        "subject_dir": subj_dir,
    }


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


def _plot_strategy(prep: dict, rewarded_only: bool):
    """Binary SHORT/LONG strategy across the continuous trial axis, with sleep markers."""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot(prep["trial_ids"], prep["s"], marker="|", linestyle="none", markersize=7,
            color=_DATA_COLOR, alpha=0.7, zorder=2)
    _mark_sessions(ax, prep["session_ends"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlim(-1, max(prep["n_trials"], 1))
    ax.set_xlabel("Trial (continuous across sessions)")
    kept = "rewarded" if rewarded_only else "completed"
    ax.set_title(f"Subject {prep['subjid']} - strategy per {kept} trial "
                 f"({prep['n_trials']} trials, {len(prep['session_labels'])} sessions)")
    ax.legend(loc="center left", fontsize=7)
    fig.tight_layout()
    return fig


def _plot_posterior(prep: dict, fit: dict, likelihood_window: int):
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
    ax.set_title(f"Subject {prep['subjid']} - switch-point posterior "
                 f"(+/-{likelihood_window} trials around peak)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def _plot_model_comparison(prep: dict, comparison: dict):
    """Overlay the constant, switch, and logistic fits on the data, with AIC/BIC in-panel."""
    s, x, n = prep["s"], prep["trial_ids"], prep["n_trials"]
    constant, switch, logistic = (comparison["fits"][m] for m in ("constant", "switch", "logistic"))

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
    grid = np.linspace(0, max(n - 1, 1), 500)
    ax.plot(grid, logistic_p(grid, logistic["midpoint"], logistic["slope"], logistic["lo"], logistic["hi"]),
            color=_LOGISTIC_COLOR, linewidth=1.8, linestyle="--", zorder=6,
            label=f"Logistic: slope = {logistic['slope']:.3f}")
    _mark_sessions(ax, prep["session_ends"])

    rows = [f"{'model':<9}{'AIC':>10}{'BIC':>10}"]
    rows += [f"{m:<9}{comparison[m]['aic']:>10.1f}{comparison[m]['bic']:>10.1f}"
             for m in ("constant", "switch", "logistic")]
    rows.append(f"best: AIC {comparison['best_aic']}, BIC {comparison['best_bic']}")
    ax.text(0.015, 0.97, "\n".join(rows), transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax.set_ylim(-0.35, 1.45)
    ax.set_xlim(-1, max(n, 1))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_xlabel("Trial (continuous across sessions)")
    ax.set_ylabel("P(SHORT)")
    ax.set_title(f"Subject {prep['subjid']} - model comparison")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def run_analysis(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    likelihood_window: int = 100,
    show: bool = True,
) -> dict:
    """Fit and plot the strategy switch for each subject independently.

    Produces three figures per animal: the binary SHORT/LONG strategy with sleep markers,
    the switch-point posterior windowed around its peak, and the constant/switch/logistic
    model comparison with AIC and BIC in-panel. The peak trial, its session, and the HDI
    width are printed as well as annotated.

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a ``{subjid: date_range}`` dict as a shorthand, in which
        case ``date_ranges`` may be omitted.
    date_ranges : dict | None
        ``{subjid: date_range}``, each value an inclusive ``(start, end)`` ``YYYYMMDD``
        tuple, an explicit date list, or ``None`` for all sessions. A non-dict value is
        applied to every subject.
    rewarded_only : bool
        Keep only ``response_time_category == "rewarded"`` trials (always excludes aborts).
    likelihood_window : int
        Half-width, in trials, of the posterior plot's window around the peak.
    show : bool
        Call ``plt.show()``. Set False in notebooks to hold the figures.

    Returns
    -------
    dict
        Keyed by subjid: ``tau``, ``tau_session``, ``hdi``, ``hdi_width``, ``fwhm``,
        ``fwhm_width``, ``p1``, ``p2``, ``comparison``, ``session_ends``,
        ``session_starts``, ``session_labels``, ``n_trials``, ``prep``, and ``figures``
        (``strategy``, ``posterior``, ``model_comparison``).
    """
    subjids, date_ranges, dates_for = _normalize_subjids_dates(subjids, date_ranges)
    results = {}

    with plt.rc_context(nature_style()):
        for subjid in subjids:
            prep = _prepare_subject(subjid, dates_for(subjid), rewarded_only)
            if prep["n_trials"] < 2:
                print(f"[switchpoint] Subject {subjid}: {prep['n_trials']} kept trial(s); skipping.")
                continue

            fit = fit_switchpoint(prep["s"])
            comparison = compare_models(prep["s"])
            tau = fit["tau"]
            tau_session = prep["session_labels"][int(prep["session_index"][tau])]
            hdi_lo, hdi_hi = fit["hdi"]
            fwhm_lo, fwhm_hi = fit["fwhm"]
            hdi_width, fwhm_width = hdi_hi - hdi_lo + 1, fwhm_hi - fwhm_lo + 1

            print(f"\n[switchpoint] Subject {subjid} ({prep['n_trials']} trials, "
                  f"{len(prep['session_labels'])} sessions)")
            print(f"  tau (global trial id) = {tau}, in session {tau_session}")
            print(f"  p1 = {fit['p1']:.3f} -> p2 = {fit['p2']:.3f}")
            print(f"  95% HDI = [{hdi_lo}, {hdi_hi}], width = {hdi_width} trials")
            print(f"  FWHM    = [{fwhm_lo}, {fwhm_hi}], width = {fwhm_width} trials")
            print(f"  best model: AIC -> {comparison['best_aic']}, BIC -> {comparison['best_bic']}")

            figures = {
                "strategy": _plot_strategy(prep, rewarded_only),
                "posterior": _plot_posterior(prep, fit, likelihood_window),
                "model_comparison": _plot_model_comparison(prep, comparison),
            }
            results[subjid] = {
                "tau": tau, "tau_session": tau_session, "hdi": fit["hdi"], "hdi_width": hdi_width,
                "fwhm": fit["fwhm"], "fwhm_width": fwhm_width, "p1": fit["p1"], "p2": fit["p2"],
                "posterior": fit["posterior"], "comparison": comparison,
                "session_ends": prep["session_ends"], "session_starts": prep["session_starts"],
                "session_labels": prep["session_labels"], "n_trials": prep["n_trials"],
                "prep": prep, "figures": figures,
            }
        if show:
            plt.show()
    return results


def _plot_box_with_points(ax, groups: dict) -> None:
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


def _plot_null_distribution(ax, null_means: np.ndarray, observed_mean: float, p_value: float) -> None:
    """Null distribution of the permutation mean, with the observed mean marked."""
    ax.hist(null_means, bins=40, color=_SESSION_LINE_COLOR, alpha=0.65, edgecolor="white", linewidth=0.4)
    ax.axvline(observed_mean, color=_SWITCH_COLOR, linewidth=1.8,
               label=f"observed {observed_mean:.1f}\np = {p_value:.4f}")
    ax.set_xlabel("Mean f (donated)")
    ax.set_ylabel("Permutations")
    ax.legend(loc="upper right", fontsize=7)


def _pairwise_f(per_subject: dict, candidates: list) -> np.ndarray:
    """``f`` for every ordered (recipient, donor) pair, NaN where the pair is invalid.

    A pair is invalid on the diagonal, and when the recipient's ``tau`` falls beyond the
    donor's trial axis (``tau > donor last trial``). Scoring such a pair would measure ``f``
    from the donor's final session start -- an arbitrarily inflated value that biases the
    null -- so it is dropped. Note the donor's last *session start* is not the cutoff: a
    ``tau`` between it and the donor's last trial still lands inside a real donated session.
    """
    n = len(candidates)
    f = np.full((n, n), np.nan)
    for i, recipient in enumerate(candidates):
        tau = per_subject[recipient]["tau"]
        for j, donor in enumerate(candidates):
            if i == j or tau > per_subject[donor]["last_trial"]:
                continue
            f[i, j] = distance_to_session_start(tau, per_subject[donor]["session_starts"])
    return f


def _sample_assignment(rng: np.random.Generator, recipients: list, valid_donors: dict) -> list:
    """Assign each recipient one span-valid donor, without replacement where possible.

    Recipients are filled in random order, each taking a donor not yet used. A greedy pass
    can strand a later recipient whose only valid donors are all taken, so the whole
    assignment is resampled; after ``_ASSIGNMENT_TRIES`` failures the sampler falls back to
    drawing with replacement, which always succeeds because every recipient here has at
    least one valid donor.
    """
    for _ in range(_ASSIGNMENT_TRIES):
        used, donors = set(), {}
        for i in rng.permutation(recipients):
            choices = [d for d in valid_donors[int(i)] if d not in used]
            if not choices:
                break
            donors[int(i)] = int(rng.choice(choices))
            used.add(donors[int(i)])
        if len(donors) == len(recipients):
            return [donors[i] for i in recipients]
    return [int(rng.choice(valid_donors[i])) for i in recipients]


def _permutation_null_means(f_matrix: np.ndarray, recipients: list, valid_donors: dict,
                            n_permutations: int, seed: int) -> np.ndarray:
    """Null distribution of the mean ``f`` when every recipient gets one donated boundary set."""
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        donors = _sample_assignment(rng, recipients, valid_donors)
        null_means[k] = np.mean([f_matrix[i, j] for i, j in zip(recipients, donors)])
    return null_means


def run_permutation(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    inclusion: str = "bic_switch_wins",
    n_permutations: int = 10000,
    seed: int = 0,
    show: bool = True,
) -> dict:
    """Permutation test of whether strategy switches sit closer to sleep than chance.

    For every included animal, ``f`` is the number of trials from the start of the session
    containing ``tau`` to ``tau`` -- i.e. how deep into a session, and so how long after
    sleep, the switch happened.

    **Statistic**: the mean of ``f`` across included animals.

    **Null**: switches are unrelated to that animal's own sleep timing. It is realized by
    *donating* boundaries across animals -- each recipient keeps its real ``tau`` and its own
    trial axis, but is scored against another included animal's session starts. One
    permutation assigns every recipient exactly one donor (without replacement where
    possible) and takes the mean ``f`` over recipients; ``n_permutations`` of these give the
    null distribution. Pairing donors one-to-one keeps each permutation's statistic on the
    same footing as the observed one; pooling all recipient x donor values instead would
    understate the null's spread.

    **Direction**: one-sided, testing that switches sit *closer* to real sleep than chance:
    ``p = (1 + #{null mean <= observed mean}) / (n_permutations + 1)``. The ``+1`` keeps ``p``
    strictly positive. A small ``p`` means real ``f`` is smaller than donated ``f``.

    Pairs whose donor trial axis does not reach the recipient's ``tau`` are dropped rather
    than scored (see ``_pairwise_f``), both from the null and from the plotted pool.

    Selects its own subjects and recomputes every fit, so it never depends on
    ``run_analysis`` having been called.

    Parameters
    ----------
    subjids, date_ranges, rewarded_only
        As in ``run_analysis``; the subject set may differ.
    inclusion : str
        Which animals count as having a switch:

        - ``"bic_switch_wins"`` (default) -- the switch model has the lowest BIC of the
          three, so the change is both real and abrupt rather than a gradual drift.
        - ``"bic_beats_constant"`` -- the switch model only has to beat the constant model.
        - ``"aic_switch_wins"`` -- as the default, under the milder AIC penalty.
        - ``"all"`` -- no filtering.
    n_permutations : int
        Permutations drawn for the null distribution.
    seed : int
        RNG seed, for a reproducible null.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    dict
        ``real_f`` (one value per included animal), ``shuffled_f`` (every span-valid
        recipient x donor pair, for the boxplot), ``null_means``, ``observed_mean``,
        ``p_value``, ``n_permutations``, ``n_pairs_dropped``, ``included_subjids``,
        ``excluded_subjids`` (no switch), ``excluded_no_donor`` (no donor spans their
        ``tau``), ``per_subject``, and ``fig``.

    Raises
    ------
    ValueError
        Unknown ``inclusion`` rule, or fewer than two animals left to compare.
    """
    if inclusion not in _INCLUSION_RULES:
        raise ValueError(f"inclusion must be one of {sorted(_INCLUSION_RULES)}, got {inclusion!r}")
    subjids, date_ranges, dates_for = _normalize_subjids_dates(subjids, date_ranges)
    keep = _INCLUSION_RULES[inclusion]

    per_subject, excluded = {}, []
    for subjid in subjids:
        prep = _prepare_subject(subjid, dates_for(subjid), rewarded_only)
        if prep["n_trials"] < 2:
            print(f"[permutation] Subject {subjid}: {prep['n_trials']} kept trial(s); excluded.")
            excluded.append(subjid)
            continue
        comparison = compare_models(prep["s"])
        if not keep(comparison):
            print(f"[permutation] Subject {subjid}: no switch under '{inclusion}' "
                  f"(BIC winner: {comparison['best_bic']}); excluded.")
            excluded.append(subjid)
            continue
        tau = comparison["fits"]["switch"]["tau"]
        per_subject[subjid] = {
            "tau": tau, "session_starts": prep["session_starts"],
            "f": distance_to_session_start(tau, prep["session_starts"]),
            "comparison": comparison, "n_trials": prep["n_trials"],
            "last_trial": prep["n_trials"] - 1,
        }

    candidates = list(per_subject)
    if len(candidates) < 2:
        raise ValueError(f"Need >= 2 included animals for the across-animal shuffle, "
                         f"got {len(candidates)} under inclusion='{inclusion}'")

    # Span guard: drop (recipient, donor) pairs whose donor axis is too short for the
    # recipient's tau. A recipient left with no valid donor cannot enter the test, but it
    # still donates its own boundaries to the others.
    f_matrix = _pairwise_f(per_subject, candidates)
    valid_donors = {i: np.flatnonzero(np.isfinite(f_matrix[i])) for i in range(len(candidates))}
    recipients = [i for i in valid_donors if valid_donors[i].size]
    excluded_no_donor = [candidates[i] for i in valid_donors if not valid_donors[i].size]
    n_pairs_dropped = len(candidates) * (len(candidates) - 1) - int(np.isfinite(f_matrix).sum())

    if len(recipients) < 2:
        raise ValueError(f"Need >= 2 animals with at least one span-valid donor, got "
                         f"{len(recipients)}; {n_pairs_dropped} pair(s) failed the span guard")

    included = [candidates[i] for i in recipients]
    real_f = np.array([per_subject[s]["f"] for s in included], dtype=float)
    shuffled_f = f_matrix[np.isfinite(f_matrix)]
    observed_mean = float(np.mean(real_f))
    null_means = _permutation_null_means(f_matrix, recipients, valid_donors, n_permutations, seed)
    p_value = float((1 + np.sum(null_means <= observed_mean)) / (n_permutations + 1))

    print(f"\n[permutation] included {len(included)} animals ({inclusion}): {included}")
    if excluded:
        print(f"[permutation] excluded {len(excluded)} animals (no switch): {excluded}")
    if excluded_no_donor:
        print(f"[permutation] excluded {len(excluded_no_donor)} animals (no span-valid donor): "
              f"{excluded_no_donor}")
    print(f"[permutation] dropped {n_pairs_dropped} of {len(candidates) * (len(candidates) - 1)} "
          f"(recipient, donor) pairs failing the span guard")
    print(f"  real f      : median {np.median(real_f):.1f}, mean {observed_mean:.1f}, n = {real_f.size}")
    print(f"  shuffled f  : median {np.median(shuffled_f):.1f}, mean {np.mean(shuffled_f):.1f}, "
          f"n = {shuffled_f.size}")
    print(f"  observed mean f = {observed_mean:.2f}, null mean = {null_means.mean():.2f} "
          f"({n_permutations} permutations, seed {seed})")
    print(f"  one-sided p (real f closer to sleep than chance) = {p_value:.4f}")

    with plt.rc_context(nature_style()):
        fig, (ax_box, ax_null) = plt.subplots(1, 2, figsize=(10.5, 4.5))
        _plot_box_with_points(ax_box, {"f": real_f, "Shuffled f": shuffled_f})
        ax_box.set_ylabel("Trials from session start (f)")
        ax_box.set_title(f"observed mean f = {observed_mean:.1f}\np = {p_value:.4f} (one-sided)",
                         fontsize=8)
        _plot_null_distribution(ax_null, null_means, observed_mean, p_value)
        ax_null.set_title(f"Paired-permutation null\n({n_permutations} permutations)", fontsize=8)
        fig.suptitle(f"Switch alignment to sleep (n = {len(included)} animals, "
                     f"inclusion: {inclusion})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if show:
            plt.show()

    return {"real_f": real_f, "shuffled_f": shuffled_f, "null_means": null_means,
            "observed_mean": observed_mean, "p_value": p_value, "n_permutations": n_permutations,
            "n_pairs_dropped": n_pairs_dropped, "included_subjids": included,
            "excluded_subjids": excluded, "excluded_no_donor": excluded_no_donor,
            "per_subject": per_subject, "fig": fig}


# --- terminal wrappers (parsing only; all logic stays in the functions above) ------------


def _resolve_dates(args) -> Optional[Union[tuple, list]]:
    """Turn the mutually exclusive --dates / --date-range args into a date_range value."""
    if args.date_range:
        return (args.date_range[0], args.date_range[1])
    if args.dates:
        return list(args.dates)
    return None


def _resolve_date_ranges(args) -> tuple[list[int], dict]:
    """Validate the requested subjects and map each to the same CLI-supplied date range."""
    dates = _resolve_dates(args)
    check_dates = list(args.dates) if args.dates else None
    subjids = [s for s in args.subjids if validate_subject(s, check_dates)["ok"]]
    return subjids, {s: dates for s in subjids}


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subjids", nargs="+", type=int, required=True, help="subject id(s)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dates", nargs="*", type=int, default=None, help="specific date(s) YYYYMMDD")
    group.add_argument("--date-range", nargs=2, type=int, metavar=("START", "END"),
                       help="inclusive YYYYMMDD range")
    parser.add_argument("--rewarded-only", action="store_true",
                        help="keep only rewarded trials (aborts are always dropped)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analysis = subparsers.add_parser("analysis", help="per-animal switch-point fit and figures")
    _add_shared_args(analysis)
    analysis.add_argument("--likelihood-window", type=int, default=100,
                          help="half-width in trials of the posterior plot window (default: 100)")

    permutation = subparsers.add_parser("permutation", help="switch vs sleep-boundary alignment")
    _add_shared_args(permutation)
    permutation.add_argument("--inclusion", default="bic_switch_wins", choices=sorted(_INCLUSION_RULES),
                             help="which animals count as having a switch (default: bic_switch_wins)")
    permutation.add_argument("--n-permutations", type=int, default=10000,
                             help="permutations drawn for the null distribution (default: 10000)")
    permutation.add_argument("--seed", type=int, default=0,
                             help="RNG seed for the permutation null (default: 0)")

    args = parser.parse_args()
    subjids, date_ranges = _resolve_date_ranges(args)
    if not subjids:
        print("Nothing to run after validation.")
        return 1

    if args.command == "analysis":
        run_analysis(subjids, date_ranges, rewarded_only=args.rewarded_only,
                     likelihood_window=args.likelihood_window, show=True)
    else:
        run_permutation(subjids, date_ranges, rewarded_only=args.rewarded_only,
                        inclusion=args.inclusion, n_permutations=args.n_permutations,
                        seed=args.seed, show=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
