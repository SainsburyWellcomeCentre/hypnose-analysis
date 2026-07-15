"""Q-learning account of the strategy change -- NOT YET IMPLEMENTED.

A mechanistic sibling to the descriptive switch-point family in ``switch.py``: rather than
describing the P(SHORT) curve, it derives that curve from a trial-by-trial value update.
``fit_qlearning`` is currently a stub that slots into ``compare_models`` (see ``compare.py``)
with ``loglik = -inf`` and ``implemented = False``, so it appears in every table and plot but
is never selected. When implemented it must be multi-start over a dispersed ``(alpha, beta)``
grid, exactly as the logistic is (see the note in ``fit_qlearning``).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from hypnose.modelling.switchpoint.switch import _as_binary

__all__ = ["fit_qlearning"]


def fit_qlearning(s: Sequence[int] | np.ndarray) -> dict:
    """Placeholder for a Q-learning account of the strategy change -- NOT IMPLEMENTED.

    Returns the same dict shape as the other fits with ``loglik = -inf`` and
    ``implemented = False``, so it slots into ``compare_models`` and the plots without
    breaking them and is never selected as the best model.

    Planned model
    -------------
    A per-trial value update rather than a descriptive curve: the animal holds a value for the
    SHORT option, updates it after every trial with learning rate ``alpha``, and chooses via a
    softmax with inverse temperature ``beta``. Fitted by maximum likelihood on the *same*
    per-trial Bernoulli choices the other models use, so the logliks stay directly comparable
    (``k_params = 2``, plus whatever initial value is fitted rather than fixed).

    It must be fitted with a MULTI-START over a dispersed ``(alpha, beta)`` grid, exactly as
    the logistic is: the likelihood surface in ``(alpha, beta)`` is not guaranteed unimodal
    (small ``alpha`` with large ``beta`` can mimic large ``alpha`` with small ``beta``), and a
    single start would under-fit it -- which would strawman the model against the descriptive
    ones rather than test it.

    Returns
    -------
    dict
        ``alpha``, ``beta`` (NaN), ``loglik`` (``-inf``), ``k_params`` (2),
        ``implemented`` (False).
    """
    _as_binary(s)  # validate the input the same way as the real fits, so callers fail early
    return {"alpha": float("nan"), "beta": float("nan"), "loglik": float("-inf"),
            "k_params": 2, "implemented": False}
