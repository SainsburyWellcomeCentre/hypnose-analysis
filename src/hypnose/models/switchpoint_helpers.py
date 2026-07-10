"""Bernoulli switch-point model for a binary strategy sequence.

Closed-form, O(n) profile likelihood over the candidate switch trials -- no MCMC and no
change-point package. Everything here is pure numeric (numpy + scipy): numpy arrays in,
numpy arrays and plain dicts out. No file I/O, no plotting, no pandas, no path handling;
those live in ``scripts/modelling/switchpoint_analysis.py``.

Model
-----
Each solved trial ``i`` is a Bernoulli draw ``s[i] ~ Bern(p_i)``, where ``s[i] == 1`` means
the animal left via the SHORT sequence. Three descriptions of ``p_i`` are compared:

- ``constant``: ``p_i = p`` -- no strategy change (1 parameter).
- ``switch``: ``p_i = p1`` for ``i < tau`` and ``p2`` for ``i >= tau`` -- one abrupt,
  directional change (3 parameters: ``tau``, ``p1``, ``p2``).
- ``logistic``: ``p_i = lo + (hi - lo) * sigmoid(slope * (i - midpoint))`` -- a graded
  change (4 parameters). ``slope`` is the abruptness: large ``slope`` approaches the step
  of the switch model, small ``slope`` is a slow drift.

``tau`` is the index of the FIRST trial of the post-switch regime, so it ranges over
``1 .. n-1``; a switch at ``0`` or ``n`` is just the constant model and is excluded.

Probabilities are clipped away from 0 and 1 so the log-likelihood stays finite when a
segment is all zeros or all ones.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

# Keeps log(p) and log1p(-p) finite for degenerate (all-0 / all-1) segments.
_EPS = 1e-12

# Clamp the logistic argument before exp() so a large fitted slope cannot overflow.
_Z_MAX = 500.0

__all__ = [
    "bernoulli_loglik",
    "switchpoint_loglik_profile",
    "switchpoint_posterior",
    "posterior_hdi",
    "posterior_fwhm",
    "fit_constant",
    "fit_switchpoint",
    "logistic_p",
    "fit_logistic",
    "compare_models",
    "distance_to_session_start",
]


def _clip_p(p):
    """Clip probabilities into ``[_EPS, 1 - _EPS]`` so log-likelihoods stay finite."""
    return np.clip(p, _EPS, 1.0 - _EPS)


def _as_binary(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Coerce ``s`` to a flat int array of 0/1, raising on anything else."""
    arr = np.asarray(s).ravel()
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError("s must contain only 0 and 1")
    return arr.astype(np.int64)


def bernoulli_loglik(s: Sequence[int] | np.ndarray, p) -> float:
    """Log-likelihood of ``s`` under Bernoulli success probability ``p``.

    Parameters
    ----------
    s : array_like of {0, 1}
        Binary outcome sequence.
    p : float or array_like
        Success probability. A scalar applies to every trial; an array must be the
        same length as ``s`` (used for the per-trial logistic probabilities).

    Returns
    -------
    float
    """
    s = _as_binary(s)
    p = _clip_p(np.asarray(p, dtype=float))
    return float(np.sum(s * np.log(p) + (1 - s) * np.log1p(-p)))


def switchpoint_loglik_profile(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Profile log-likelihood of a switch at each candidate trial, in O(n).

    Element ``tau`` is the log-likelihood of the switch model whose post-switch regime
    starts at trial ``tau``, with ``p1`` and ``p2`` set to their (closed-form) maximum-
    likelihood values -- the segment means. Computed from a prefix sum, so no loop over
    trials. Element ``0`` is ``-inf``: a switch at trial 0 has an empty pre-switch
    segment and is the constant model, not a switch.

    Returns
    -------
    np.ndarray
        Length ``n``. All ``-inf`` when ``n < 2``.
    """
    s = _as_binary(s)
    n = s.size
    ll = np.full(n, -np.inf)
    if n < 2:
        return ll
    prefix = np.concatenate(([0], np.cumsum(s)))  # prefix[i] = successes in s[:i]
    tau = np.arange(1, n)
    n1, k1 = tau.astype(float), prefix[tau].astype(float)
    n2, k2 = (n - tau).astype(float), (prefix[-1] - prefix[tau]).astype(float)
    p1, p2 = _clip_p(k1 / n1), _clip_p(k2 / n2)
    ll[1:] = (k1 * np.log(p1) + (n1 - k1) * np.log1p(-p1)
              + k2 * np.log(p2) + (n2 - k2) * np.log1p(-p2))
    return ll


def switchpoint_posterior(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Normalized posterior over the switch trial under a uniform prior on ``tau``.

    Exponentiates the profile log-likelihood with the max subtracted (so the largest
    term is ``exp(0) == 1`` and nothing underflows), then normalizes to sum to 1.

    Returns
    -------
    np.ndarray
        Length ``n``, non-negative, sums to 1 (all zeros when ``n < 2``).
    """
    ll = switchpoint_loglik_profile(s)
    finite = np.isfinite(ll)
    posterior = np.zeros(ll.size, dtype=float)
    if not finite.any():
        return posterior
    posterior[finite] = np.exp(ll[finite] - ll[finite].max())
    total = posterior.sum()
    return posterior / total if total > 0 else posterior


def posterior_hdi(posterior: np.ndarray, mass: float = 0.95) -> tuple[int, int]:
    """Highest-density interval of a discrete posterior, as ``(lo, hi)`` trial indices.

    Takes trials in order of descending posterior mass until ``mass`` is covered, then
    returns the smallest and largest index in that set. The interval is reported as a
    range, so it can contain trials below the density threshold when the posterior is
    multimodal -- ``posterior_fwhm`` is the sharper (secondary) width measure.

    Returns
    -------
    (int, int)
        Inclusive bounds. ``(0, n - 1)`` for an empty/degenerate posterior.
    """
    post = np.asarray(posterior, dtype=float)
    total = post.sum()
    if post.size == 0:
        return (0, 0)
    if total <= 0:
        return (0, post.size - 1)
    order = np.argsort(post)[::-1]
    covered = np.searchsorted(np.cumsum(post[order]), mass * total) + 1
    kept = order[:min(int(covered), post.size)]
    return int(kept.min()), int(kept.max())


def posterior_fwhm(posterior: np.ndarray) -> tuple[int, int]:
    """Full width at half maximum of a discrete posterior, as ``(lo, hi)`` indices.

    The first and last trial whose posterior mass is at least half the peak mass.
    Secondary to ``posterior_hdi``: narrower, and it ignores how much total mass the
    interval actually holds.
    """
    post = np.asarray(posterior, dtype=float)
    if post.size == 0:
        return (0, 0)
    peak = post.max()
    if peak <= 0:
        return (0, post.size - 1)
    above = np.flatnonzero(post >= 0.5 * peak)
    return int(above[0]), int(above[-1])


def fit_constant(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the no-change model: a single Bernoulli rate for the whole sequence.

    Returns
    -------
    dict
        ``loglik``, ``p`` (the sequence mean), ``k_params`` (1).
    """
    s = _as_binary(s)
    p = float(s.mean()) if s.size else 0.5
    return {"loglik": bernoulli_loglik(s, p), "p": p, "k_params": 1}


def fit_switchpoint(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the one-switch model by maximizing the profile log-likelihood over ``tau``.

    Returns
    -------
    dict
        ``tau`` (first trial of the post-switch regime), ``p1``, ``p2``, ``loglik``,
        ``posterior`` (length ``n``), ``hdi`` and ``fwhm`` as ``(lo, hi)`` index pairs,
        and ``k_params`` (3). For ``n < 2`` the fit is degenerate: ``tau`` is 0 and the
        rates are NaN.
    """
    s = _as_binary(s)
    n = s.size
    posterior = switchpoint_posterior(s)
    if n < 2:
        return {"tau": 0, "p1": float("nan"), "p2": float("nan"), "loglik": float("-inf"),
                "posterior": posterior, "hdi": (0, max(n - 1, 0)),
                "fwhm": (0, max(n - 1, 0)), "k_params": 3}
    ll = switchpoint_loglik_profile(s)
    tau = int(np.argmax(ll))  # ll[0] is -inf, so tau lands in 1 .. n-1
    return {"tau": tau, "p1": float(s[:tau].mean()), "p2": float(s[tau:].mean()),
            "loglik": float(ll[tau]), "posterior": posterior,
            "hdi": posterior_hdi(posterior), "fwhm": posterior_fwhm(posterior),
            "k_params": 3}


def logistic_p(x: np.ndarray, midpoint: float, slope: float, lo: float, hi: float) -> np.ndarray:
    """Per-trial success probability of the logistic model (asymptotes ``lo`` and ``hi``).

    Exposed so a caller can draw the fitted curve on an arbitrary trial grid.
    """
    z = np.clip(slope * (x - midpoint), -_Z_MAX, _Z_MAX)
    return lo + (hi - lo) * expit(z)


def fit_logistic(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the graded-change model by maximum likelihood (Nelder-Mead).

    ``p_i = lo + (hi - lo) * sigmoid(slope * (i - midpoint))``. The asymptotes are fitted
    rather than fixed at 0/1, so the model nests the switch model (as ``slope -> inf``)
    and ``slope`` is directly interpretable as abruptness. Initialized from the
    switch-point fit, which keeps the optimizer off the flat parts of the surface.

    Returns
    -------
    dict
        ``midpoint``, ``slope`` (abruptness; negative means SHORT was abandoned), ``lo``,
        ``hi``, ``loglik``, ``k_params`` (4), ``converged``.
    """
    s = _as_binary(s)
    n = s.size
    if n < 2:
        return {"midpoint": 0.0, "slope": 0.0, "lo": 0.5, "hi": 0.5,
                "loglik": float("-inf"), "k_params": 4, "converged": False}
    x = np.arange(n, dtype=float)
    switch = fit_switchpoint(s)
    lo0, hi0 = _clip_p(switch["p1"]), _clip_p(switch["p2"])
    theta0 = np.array([float(switch["tau"]), 1.0, logit(lo0), logit(hi0)])

    def negative_loglik(theta: np.ndarray) -> float:
        p = logistic_p(x, theta[0], theta[1], expit(theta[2]), expit(theta[3]))
        return -bernoulli_loglik(s, p)

    result = minimize(negative_loglik, theta0, method="Nelder-Mead",
                      options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    midpoint, slope, lo_raw, hi_raw = result.x
    return {"midpoint": float(midpoint), "slope": float(slope), "lo": float(expit(lo_raw)),
            "hi": float(expit(hi_raw)), "loglik": float(-result.fun), "k_params": 4,
            "converged": bool(result.success)}


def compare_models(s: Sequence[int] | np.ndarray) -> dict:
    """Fit all three models and score them with AIC and BIC (lower is better).

    ``AIC = 2k - 2 * loglik`` and ``BIC = k * ln(n) - 2 * loglik``. BIC penalizes the
    extra parameters harder, so it is the stricter test of "there really was a switch".

    Returns
    -------
    dict
        ``constant`` / ``switch`` / ``logistic``, each ``{loglik, k_params, aic, bic}``;
        ``best_aic`` and ``best_bic`` (the winning model name); and ``fits``, the full
        fit dict of each model.
    """
    s = _as_binary(s)
    n = max(s.size, 1)
    fits = {"constant": fit_constant(s), "switch": fit_switchpoint(s), "logistic": fit_logistic(s)}
    scores = {}
    for name, fit in fits.items():
        k, loglik = fit["k_params"], fit["loglik"]
        scores[name] = {"loglik": loglik, "k_params": k,
                        "aic": 2 * k - 2 * loglik, "bic": k * np.log(n) - 2 * loglik}
    scores["best_aic"] = min(fits, key=lambda m: scores[m]["aic"])
    scores["best_bic"] = min(fits, key=lambda m: scores[m]["bic"])
    scores["fits"] = fits
    return scores


def distance_to_session_start(tau: float, boundaries: Sequence[float] | np.ndarray) -> float:
    """Trials from the start of the session containing ``tau`` to ``tau`` itself.

    ``boundaries`` are the ordered global trial ids on which sessions *start* (a session
    start is the trial after a sleep period). The containing session is the one with the
    greatest start at or before ``tau``, so ``f = tau - that_start``. Small ``f`` means
    the switch happened soon after sleep.

    ``tau`` at or after the last boundary falls in the final session and is handled by
    the same rule. ``tau`` before the first boundary belongs to no session and returns
    NaN, as does an empty ``boundaries`` -- both are undefined rather than zero, so a
    caller cannot silently average them in.

    Returns
    -------
    float
        Non-negative trial count, or NaN when ``tau`` precedes every session start.
    """
    starts = np.sort(np.asarray(boundaries, dtype=float).ravel())
    if starts.size == 0:
        return float("nan")
    index = int(np.searchsorted(starts, float(tau), side="right")) - 1
    if index < 0:
        return float("nan")
    return float(tau) - float(starts[index])
