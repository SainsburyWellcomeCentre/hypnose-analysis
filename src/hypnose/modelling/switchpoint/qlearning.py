"""Q-learning account of the strategy change -- the mechanistic NULL to be rejected.

A mechanistic sibling to the descriptive switch-point family in ``switch.py``: rather than
describing the P(SHORT) curve, it derives that curve from a trial-by-trial value update. Its
role here is adversarial. Incremental reinforcement learning produces a *gradual* rise in
P(SHORT) whose steepness is set by the learning rate, so if a Q-learner fits an animal as well
as the step does, the "sudden strategy switch" reading is not supported. The three variants
below are therefore fitted at their best (multi-start, see ``fit_qlearning``) -- a strawmanned
null that loses proves nothing.

Everything here is pure numeric (numpy + scipy): numpy arrays in, numpy arrays and plain dicts
out. No file I/O, no plotting. Figures live in
``hypnose.visualization.modelling.switchpoint.plots``; the orchestration is in
``scripts/modelling/switchpoint_analysis.py``.

Model
-----
Two options, SHORT and LONG, with **fixed** rewards ``r_short = 1``, ``r_long = 0``
(``R_SHORT`` / ``R_LONG``). Only the chosen option updates::

    Q[chosen] += alpha * (r[chosen] - Q[chosen])

and the choice rule is a softmax with a perseveration (choice-history) term::

    P(SHORT at t) = 1 / (1 + exp(-(b * (Q_short - Q_long) + kappa * s_prev)))

where ``s_prev`` is ``+1`` if trial ``t-1`` was SHORT, ``-1`` if it was LONG, and ``0`` at the
first trial. ``kappa = 0`` for the non-perseveration variants. Every trial's probability is
computed from the Q values held *before* that trial's update; the update is then applied using
the route the animal actually took.

Why the rewards are constants and not parameters
------------------------------------------------
Fixing ``(r_short, r_long) = (1, 0)`` is a **choice of units, not a claim that LONG is
unrewarded** -- both routes are rewarded in the real task, and the sequence modelled here is
already conditioned on reward. The true reward advantage ``d`` of SHORT over LONG is
*unidentifiable* from choice data, because it enters the choice rule only through the product
``b * d``: the fitted inverse temperature ``b`` absorbs it.

Concretely, writing the rewards as ``(1, 1 - d)`` and mapping every value through the affine
map ``x -> d * x + (1 - d)`` (which sends ``0 -> 1 - d`` and fixes ``1``) scales every value
difference ``Q_short - Q_long`` by exactly ``d``. So::

    rewards (1, 0)      with (alpha, b,     Q0_short,               Q0_long)
    rewards (1, 1 - d)  with (alpha, b / d, d*Q0_short + (1 - d),   d*Q0_long + (1 - d))

produce the **identical** P(SHORT) trajectory and therefore the identical likelihood. Fitting
``d`` as a free parameter would add a perfectly flat direction to the likelihood surface, not
information. The reward scale is exposed as a keyword on ``qlearning_trajectory`` /
``qlearning_nll`` only so this claim can be checked numerically (see
``src/hypnose/qc/check_qlearning.py``); the fits always use the defaults.

Variants
--------
=========================  =================================  ==========================
variant                    free parameters                    ``Q0`` bounds
=========================  =================================  ==========================
``qlearn_free``            alpha, b, Q0_short, Q0_long        ``[-10, 10]``
``qlearn_constrained``     alpha, b, Q0_short, Q0_long        ``[0, 1]``
``qlearn_perseveration``   alpha, b, Q0_short, Q0_long, kappa ``[-10, 10]``, kappa
                                                              ``[-10, 10]``
=========================  =================================  ==========================

``alpha`` is bounded ``[1e-4, 1]`` and ``b`` ``[1e-3, 50]`` throughout.

``qlearn_constrained`` reads "initial values lie within the range of experienceable outcomes"
-- the animal cannot start out believing an option is worth more than any outcome it could
ever receive. That constraint has a structural consequence worth knowing before reading any
fit: since ``Q_long`` decays to ``0`` within ~``1/alpha`` LONG choices and ``Q0_short >= 0``
keeps ``Q_short >= 0``, the constrained model **cannot hold P(SHORT) below 0.5 in steady
state**. It can only describe an animal that ends up at or above chance for SHORT. The free
version may initialise outside the experienceable range and has no such floor.

Two kinds of trajectory -- do not confuse them
----------------------------------------------
``qlearning_trajectory`` returns the **one-step-ahead** P(SHORT): at every trial the model is
handed the animal's true history, because both the update counts and ``s_prev`` are read off
the observed choices. That is the right quantity for the likelihood, for AIC / BIC, and for the
residual ACF -- and it is *not* a prediction of the animal's trajectory. It cannot be, because
it is conditioned on the trajectory. Plotting it alone is misleading: with a large fitted
``kappa`` the choice rule collapses to roughly ``expit(kappa * s_prev)``, a one-trial-lagged
copy of the data, so the curve will appear to track the switch perfectly no matter how badly
the value-learning part of the model is doing.

``qlearning_generative_band`` returns the **generative** trajectory: the model run forward on
its *own* choices, averaged over many simulations, with a quantile band. That is what the
fitted Q-learner actually predicts an animal would do, and it is the one to read when asking
whether the null can reproduce an abrupt switch. It is the visually dominant overlay in the
figures for exactly this reason.

Closed form
-----------
Because each option's reward is a constant, the update ``Q += alpha * (r - Q)`` is a geometric
recursion with the exact solution ``Q_k = r - (r - Q0) * (1 - alpha)**k`` after ``k`` updates
*of that option*. The Q values entering trial ``t`` therefore depend only on how many SHORT
and LONG choices preceded it, so the whole one-step-ahead trajectory is computed without a
Python loop. ``simulate_qlearning`` still steps trial by trial, because there the choices are
being drawn; the two agree exactly, which is asserted in the qc check. Its loop is vectorized
across simulations rather than across trials, so drawing a 500-simulation generative band costs
one pass over the trials, not 500.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from hypnose.modelling.switchpoint.switch import _as_binary

# Fixed reward constants -- a choice of units, NOT fitted. See the module docstring.
R_SHORT = 1.0
R_LONG = 0.0

# Probability clip, so a saturated softmax cannot produce an infinite log-likelihood.
_P_EPS = 1e-10

# Shared parameter bounds.
_ALPHA_BOUNDS = (1e-4, 1.0)
_B_BOUNDS = (1e-3, 50.0)
_Q0_BOUNDS_FREE = (-10.0, 10.0)
_Q0_BOUNDS_EXPERIENCEABLE = (0.0, 1.0)  # "within the range of experienceable outcomes"
_KAPPA_BOUNDS = (-10.0, 10.0)

# Multi-start: alpha and Q0 trade off against each other (a small alpha with an extreme Q0
# mimics a large alpha with a moderate one), so the surface is not unimodal and a single start
# is not reliable. Starts are drawn uniformly inside the bounds from a seeded generator.
N_STARTS = 32

# An estimate this close to either end of its bound range (as a fraction of the range) is
# flagged: the optimizer stopped at the edge of the parameter space, so the value is a
# constraint artefact rather than an estimate.
_BOUNDARY_RTOL = 0.01

# The variant that represents "qlearning" in compare_models -- the most flexible of the three,
# so the model comparison scores the null at its strongest.
QLEARN_DEFAULT_VARIANT = "qlearn_free"

QLEARN_VARIANTS = {
    "qlearn_free": {
        "params": ("alpha", "b", "q0_short", "q0_long"),
        "bounds": (_ALPHA_BOUNDS, _B_BOUNDS, _Q0_BOUNDS_FREE, _Q0_BOUNDS_FREE),
        "description": "Q0 free to start outside the experienceable range",
    },
    "qlearn_constrained": {
        "params": ("alpha", "b", "q0_short", "q0_long"),
        "bounds": (_ALPHA_BOUNDS, _B_BOUNDS, _Q0_BOUNDS_EXPERIENCEABLE,
                   _Q0_BOUNDS_EXPERIENCEABLE),
        "description": "Q0 within the range of experienceable outcomes [0, 1]",
    },
    "qlearn_perseveration": {
        "params": ("alpha", "b", "q0_short", "q0_long", "kappa"),
        "bounds": (_ALPHA_BOUNDS, _B_BOUNDS, _Q0_BOUNDS_FREE, _Q0_BOUNDS_FREE, _KAPPA_BOUNDS),
        "description": "Q0 free, plus a choice-history (perseveration) term kappa",
    },
}

# Fixed order for tables, plots and the printed logliks.
QLEARN_VARIANT_ORDER = ("qlearn_free", "qlearn_constrained", "qlearn_perseveration")

# Default (alpha, b) grid for the parameter-sweep figure -- 4 x 4, geometrically spaced inside
# the fitted bounds so slow-drift through near-immediate learning is covered.
QLEARN_SWEEP_ALPHAS = (0.005, 0.03, 0.15, 0.75)
QLEARN_SWEEP_BS = (0.25, 1.0, 4.0, 16.0)

__all__ = [
    "R_SHORT",
    "R_LONG",
    "N_STARTS",
    "QLEARN_DEFAULT_VARIANT",
    "QLEARN_VARIANTS",
    "QLEARN_VARIANT_ORDER",
    "QLEARN_SWEEP_ALPHAS",
    "QLEARN_SWEEP_BS",
    "qlearning_trajectory",
    "qlearning_nll",
    "simulate_qlearning",
    "fit_qlearning",
    "fit_qlearning_variants",
    "qlearning_parameter_sweep",
]


def _as_one_animal(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Coerce ``s`` to one animal's flat 0/1 sequence, refusing pooled or averaged input.

    The Q-learning fits are per animal (or per session) by construction. Averaging animals
    that switched at different trials manufactures a gradual curve out of abrupt ones -- which
    is precisely the conclusion the null model is being used to test -- so an aggregate is
    refused loudly rather than fitted.
    """
    arr = np.asarray(s)
    if arr.ndim > 1:
        raise ValueError(
            f"expected ONE animal's 0/1 sequence, got an array of shape {arr.shape}. Fitting "
            f"pooled or stacked sequences is not supported: averaging animals with different "
            f"switch points manufactures a gradual curve out of abrupt ones, which is exactly "
            f"what this null model is meant to test. Fit each animal (or session) separately.")
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError(
            "s must contain only 0 and 1. Group means / rolling averages cannot be fitted: "
            "the likelihood is over individual SHORT/LONG choices of ONE animal.")
    return _as_binary(arr)


def qlearning_trajectory(s: Sequence[int] | np.ndarray, alpha: float, b: float,
                         q0_short: float, q0_long: float, kappa: float = 0.0,
                         rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> dict:
    """Per-trial P(SHORT) implied by a Q-learner that made the choices in ``s``.

    Each trial's probability uses the Q values held *before* that trial's update; the update
    then applies to the option the animal actually took, and only to that option. Evaluated in
    closed form (see the module docstring), so it is O(n) numpy with no Python loop.

    Parameters
    ----------
    s : array_like of {0, 1}
        The observed choices, 1 = SHORT. Only the counts of preceding choices and the previous
        trial's identity enter, so this is the "given the animal's actual history" trajectory,
        not a free-running simulation (that is ``simulate_qlearning``).
    alpha, b, q0_short, q0_long, kappa : float
        Learning rate, inverse temperature, initial values, perseveration weight.
    rewards : (float, float)
        ``(r_short, r_long)``. Exposed only so the identifiability claim in the module
        docstring can be checked numerically; every fit uses the default ``(1, 0)``.

    Returns
    -------
    dict
        ``p_short`` (length n, clipped into ``[1e-10, 1 - 1e-10]``), ``q_short``, ``q_long``
        (the values held *before* each trial's update), and ``s_prev`` (+1/-1/0).
    """
    s = _as_one_animal(s)
    n = s.size
    r_short, r_long = float(rewards[0]), float(rewards[1])
    # Updates of each option BEFORE trial t: the closed form needs only these counts.
    n_short_before = np.concatenate(([0], np.cumsum(s)[:-1])) if n else np.zeros(0, dtype=int)
    n_long_before = np.arange(n) - n_short_before
    decay = 1.0 - float(alpha)
    q_short = r_short - (r_short - float(q0_short)) * decay ** n_short_before
    q_long = r_long - (r_long - float(q0_long)) * decay ** n_long_before
    # +1 after a SHORT trial, -1 after a LONG one, 0 at the first trial (no history yet).
    s_prev = np.concatenate(([0.0], 2.0 * s[:-1] - 1.0)) if n else np.zeros(0)
    p_short = np.clip(expit(float(b) * (q_short - q_long) + float(kappa) * s_prev),
                      _P_EPS, 1.0 - _P_EPS)
    return {"p_short": p_short, "q_short": q_short, "q_long": q_long, "s_prev": s_prev}


def qlearning_nll(s: Sequence[int] | np.ndarray, alpha: float, b: float, q0_short: float,
                  q0_long: float, kappa: float = 0.0,
                  rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> tuple[float, np.ndarray]:
    """Negative summed log-likelihood of the observed sequence, plus its P(SHORT) trajectory.

    ``-sum_t log P(observed choice at t)``, directly comparable with the ``-loglik`` of the
    descriptive models in ``switch.py`` because it scores exactly the same per-trial Bernoulli
    choices.

    Returns
    -------
    (float, np.ndarray)
        ``(nll, p_short)``. ``nll`` is ``0.0`` for an empty sequence.
    """
    s = _as_one_animal(s)
    p = qlearning_trajectory(s, alpha, b, q0_short, q0_long, kappa, rewards)["p_short"]
    if s.size == 0:
        return 0.0, p
    nll = -float(np.sum(np.where(s == 1, np.log(p), np.log1p(-p))))
    return nll, p


def simulate_qlearning(n_trials: int, alpha: float, b: float, q0_short: float, q0_long: float,
                       kappa: float = 0.0, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw a SHORT/LONG sequence from the Q-learner, stepping trial by trial.

    The generative direction of ``qlearning_trajectory``: each trial's choice is sampled from
    the current P(SHORT), and only the sampled option is then updated -- so the values, and
    hence the next trial's probability, depend on what was actually drawn.

    Parameters
    ----------
    n_trials : int
        Length of the sequence to draw.
    alpha, b, q0_short, q0_long, kappa : float
        As in ``qlearning_trajectory``.
    seed : int | None
        Seed for ``np.random.default_rng``, for a reproducible draw.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(s, p_short)`` -- the drawn 0/1 choices, and the P(SHORT) each was drawn from
        (i.e. the value held *before* that trial's update). Both length ``n_trials``.
    """
    n = int(n_trials)
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=np.int64)
    p_short = np.zeros(n, dtype=float)
    q_short, q_long, s_prev = float(q0_short), float(q0_long), 0.0
    for t in range(n):
        p = float(np.clip(expit(b * (q_short - q_long) + kappa * s_prev),
                          _P_EPS, 1.0 - _P_EPS))
        p_short[t] = p
        chose_short = bool(rng.random() < p)
        s[t] = int(chose_short)
        if chose_short:  # only the chosen option updates
            q_short += alpha * (R_SHORT - q_short)
        else:
            q_long += alpha * (R_LONG - q_long)
        s_prev = 1.0 if chose_short else -1.0
    return s, p_short


def _unpack(variant: str, theta: Sequence[float]) -> dict:
    """Map a variant's packed parameter vector onto the full ``(alpha, b, q0, q0, kappa)`` set."""
    values = dict(zip(QLEARN_VARIANTS[variant]["params"], (float(v) for v in theta)))
    values.setdefault("kappa", 0.0)  # 0 for the non-perseveration variants
    return values


def _boundary_params(variant: str, values: dict) -> list[str]:
    """Names of the estimates sitting within ``_BOUNDARY_RTOL`` of either end of their bounds."""
    spec = QLEARN_VARIANTS[variant]
    hit = []
    for name, (lo, hi) in zip(spec["params"], spec["bounds"]):
        tol = _BOUNDARY_RTOL * (hi - lo)
        if values[name] - lo <= tol or hi - values[name] <= tol:
            hit.append(name)
    return hit


def _degenerate_fit(variant: str, n: int) -> dict:
    """The fit dict for a sequence too short to fit (``n < 2``): NaN estimates, ``-inf`` loglik."""
    spec = QLEARN_VARIANTS[variant]
    k = len(spec["params"])
    return {"variant": variant, "alpha": float("nan"), "b": float("nan"),
            "q0_short": float("nan"), "q0_long": float("nan"), "kappa": float("nan"),
            "free_params": spec["params"], "nll": float("inf"), "loglik": float("-inf"),
            "aic": float("inf"), "bic": float("inf"), "n_trials": int(n), "k_params": k,
            "converged": False, "n_starts": 0, "n_starts_converged": 0, "boundary_hit": False,
            "boundary_params": [], "p_short": np.zeros(int(n)), "implemented": True}


def fit_qlearning(s: Sequence[int] | np.ndarray, variant: str = QLEARN_DEFAULT_VARIANT,
                  n_starts: int = N_STARTS, seed: int = 0) -> dict:
    """Fit one Q-learning variant to ONE animal's sequence by multi-start maximum likelihood.

    ``scipy.optimize.minimize`` with ``L-BFGS-B`` (the bounds are the model), started from
    ``n_starts`` points drawn uniformly inside the bounds from a seeded generator; the lowest
    negative log-likelihood wins. Multi-start is not optional here: ``alpha`` and ``Q0`` trade
    off against each other, so the surface has more than one basin and a single start
    under-fits -- which would strawman the null against the descriptive models rather than
    test it.

    Per animal or per session, never pooled: an aggregate raises (see ``_as_one_animal``).

    Parameters
    ----------
    s : array_like of {0, 1}
        One animal's (or one session's) choices, 1 = SHORT.
    variant : str
        One of ``QLEARN_VARIANTS`` -- ``"qlearn_free"``, ``"qlearn_constrained"`` or
        ``"qlearn_perseveration"``.
    n_starts : int
        Random starting points. Must be at least 20.
    seed : int
        Seed for the starting-point draw, so a fit is reproducible.

    Returns
    -------
    dict
        ``variant``; the estimates ``alpha``, ``b``, ``q0_short``, ``q0_long``, ``kappa``
        (``kappa`` is exactly 0 and not free except in ``qlearn_perseveration``);
        ``free_params``; ``nll`` and ``loglik`` (``= -nll``); ``aic``, ``bic``; ``n_trials``,
        ``k_params``; ``converged`` (True iff at least one start converged),
        ``n_starts``, ``n_starts_converged``; ``boundary_hit`` and ``boundary_params`` (any
        estimate within 1% of a bound -- an edge-of-space artefact, not an estimate);
        ``p_short`` (the fitted per-trial P(SHORT) trajectory, for the plots); and
        ``implemented`` (True), so it scores in ``compare_models``.

    Raises
    ------
    ValueError
        Unknown ``variant``, fewer than 20 starts, or a pooled / non-binary sequence.
    """
    if variant not in QLEARN_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(QLEARN_VARIANTS)}, got {variant!r}")
    if n_starts < 20:
        raise ValueError(f"n_starts must be >= 20 (alpha and Q0 trade off, so a small "
                         f"multi-start is unreliable), got {n_starts}")
    s = _as_one_animal(s)
    n = s.size
    if n < 2:
        return _degenerate_fit(variant, n)

    spec = QLEARN_VARIANTS[variant]
    bounds = spec["bounds"]
    k = len(spec["params"])

    def negative_loglik(theta: np.ndarray) -> float:
        nll, _ = qlearning_nll(s, **_unpack(variant, theta))
        return nll

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    starts = rng.uniform(lo, hi, size=(int(n_starts), k))

    best, n_converged = None, 0
    for theta0 in starts:
        result = minimize(negative_loglik, theta0, method="L-BFGS-B", bounds=bounds)
        n_converged += int(bool(result.success))
        if best is None or result.fun < best.fun:
            best = result

    values = _unpack(variant, best.x)
    nll, p_short = qlearning_nll(s, **values)
    loglik = -nll
    boundary = _boundary_params(variant, values)
    return {"variant": variant, **values, "free_params": spec["params"],
            "nll": float(nll), "loglik": float(loglik),
            "aic": float(2 * k - 2 * loglik), "bic": float(k * np.log(n) - 2 * loglik),
            "n_trials": int(n), "k_params": k, "converged": bool(n_converged > 0),
            "n_starts": int(n_starts), "n_starts_converged": int(n_converged),
            "boundary_hit": bool(boundary), "boundary_params": boundary,
            "p_short": p_short, "implemented": True}


def fit_qlearning_variants(s: Sequence[int] | np.ndarray,
                           variants: Sequence[str] = QLEARN_VARIANT_ORDER,
                           n_starts: int = N_STARTS, seed: int = 0) -> dict:
    """Fit every variant to the same sequence, keyed by variant name.

    Convenience over ``fit_qlearning`` for the overlay and sweep figures, which need all three.
    Each variant gets the same ``seed``, so their starting points are identical and any
    difference between the fits comes from the model rather than from the draw.
    """
    return {variant: fit_qlearning(s, variant, n_starts=n_starts, seed=seed)
            for variant in variants}


def qlearning_parameter_sweep(s: Sequence[int] | np.ndarray, fit: dict,
                              alphas: Sequence[float] = QLEARN_SWEEP_ALPHAS,
                              bs: Sequence[float] = QLEARN_SWEEP_BS) -> list[dict]:
    """P(SHORT) trajectories over an ``(alpha, b)`` grid, holding the other estimates at their ML.

    The sweep shows what the *shape* of a Q-learning curve is controlled by: ``alpha`` sets how
    fast it rises and ``b`` how far it travels. ``Q0`` (and ``kappa``, for the perseveration
    variant) stay at ``fit``'s maximum-likelihood values, so every line differs from the ML fit
    in the two swept parameters only.

    Returns
    -------
    list[dict]
        One entry per grid point, in ``alpha``-major order: ``alpha``, ``b``, ``i_alpha``,
        ``i_b``, ``nll``, and ``p_short`` (the trajectory).
    """
    s = _as_one_animal(s)
    out = []
    for i_alpha, alpha in enumerate(alphas):
        for i_b, b in enumerate(bs):
            nll, p_short = qlearning_nll(s, alpha, b, fit["q0_short"], fit["q0_long"],
                                         fit["kappa"])
            out.append({"alpha": float(alpha), "b": float(b), "i_alpha": i_alpha, "i_b": i_b,
                        "nll": float(nll), "p_short": p_short})
    return out
