"""Fit every model and score them with AIC / BIC, plus per-trial fitted-P reconstruction.

The scoring layer over the model families in ``switch.py`` and ``qlearning.py``: it fits them
all, penalizes their parameter counts, and names the winner. ``_model_fitted_p`` rebuilds the
per-trial P(SHORT) curve a fitted model implies -- used by the residual-autocorrelation
diagnostic (``autocorr.py``) and by the plots.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from hypnose.modelling.switchpoint.switch import (
    _as_binary,
    fit_constant,
    fit_logistic,
    fit_switch2,
    fit_switchpoint,
    logistic_p,
)
from hypnose.modelling.switchpoint.qlearning import fit_qlearning

# Models in increasing flexibility, for tables, plots and the printed logliks. The nesting
# relations constant <= switch and switch <= logistic should hold along it; switch2 is a
# monotone-gated (p1 <= p2 <= p3) special case that need not nest the single switch.
MODEL_ORDER = ("constant", "switch", "logistic", "switch2", "qlearning")

__all__ = ["compare_models", "model_fitted_p", "MODEL_ORDER"]


def compare_models(s: Sequence[int] | np.ndarray) -> dict:
    """Fit all five models and score them with AIC and BIC (lower is better).

    The models, in increasing flexibility: ``constant`` (k=1), ``switch`` (k=3), ``logistic``
    (k=4), ``switch2`` (k=5), and ``qlearning`` (a stub -- see ``fit_qlearning``). Two nesting
    relations hold and are worth checking on any real fit: ``constant <= switch`` and
    ``switch <= logistic`` in loglik. ``switch2`` is monotone-gated (``p1 <= p2 <= p3``), so
    it does *not* nest the single switch and may be ``-inf`` when no monotone split exists.

    ``AIC = 2k - 2 * loglik`` and ``BIC = k * ln(n) - 2 * loglik``. BIC penalizes the extra
    parameters harder, so it is the stricter test of "there really was a switch".

    Models reporting ``implemented = False`` are scored (so they appear in the table) but are
    never eligible to win.

    Caveat -- the parameter counts understate the switch models' flexibility. ``switch``
    searches ~n candidate split points and ``switch2`` ~n^2/2, but they are charged only k=3
    and k=5, as if ``tau`` were an ordinary parameter. AIC and BIC are therefore *generous* to
    them relative to the logistic, and more so to ``switch2`` than to ``switch``. Treat a
    narrow BIC win for a switch model as suggestive, not decisive. The planned fix is a
    cross-validated predictive likelihood, which prices the search honestly; not done yet.

    Returns
    -------
    dict
        One entry per model name, each ``{loglik, k_params, aic, bic}``; ``best_aic`` and
        ``best_bic`` (the winning implemented model); and ``fits``, the full fit dict of each.
    """
    s = _as_binary(s)
    n = max(s.size, 1)
    fits = {"constant": fit_constant(s), "switch": fit_switchpoint(s),
            "logistic": fit_logistic(s), "switch2": fit_switch2(s), "qlearning": fit_qlearning(s)}
    scores = {}
    for name, fit in fits.items():
        k, loglik = fit["k_params"], fit["loglik"]
        scores[name] = {"loglik": loglik, "k_params": k,
                        "aic": 2 * k - 2 * loglik, "bic": k * np.log(n) - 2 * loglik}
    eligible = [name for name, fit in fits.items() if fit.get("implemented", True)]
    scores["best_aic"] = min(eligible, key=lambda m: scores[m]["aic"])
    scores["best_bic"] = min(eligible, key=lambda m: scores[m]["bic"])
    scores["fits"] = fits
    return scores


def model_fitted_p(name: str, fit: dict, n: int) -> Optional[np.ndarray]:
    """Per-trial fitted P(SHORT) of a fitted model over the continuous ``0..n-1`` trial axis.

    Rebuilds the step / curve each model implies at every trial, matching exactly the shapes
    the model-comparison plot draws. Returns ``None`` for a model with no per-trial curve (the
    unimplemented ``qlearning`` stub), so a caller can skip it.
    """
    x = np.arange(n)
    if name == "constant":
        return np.full(n, fit["p"], dtype=float)
    if name == "switch":
        return np.where(x < fit["tau"], fit["p1"], fit["p2"]).astype(float)
    if name == "switch2":
        p = np.full(n, fit["p3"], dtype=float)
        p[x < fit["tau2"]] = fit["p2"]
        p[x < fit["tau1"]] = fit["p1"]
        return p
    if name == "logistic":
        return logistic_p(x.astype(float), fit["midpoint"], fit["slope"], fit["lo"], fit["hi"])
    return None
