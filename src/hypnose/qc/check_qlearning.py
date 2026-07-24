#!/usr/bin/env python
"""Self-check for the Q-learning null model -- entirely synthetic, no data mount needed.

The other qc tools fingerprint real pipeline output. This one instead asserts the *structural*
properties of ``hypnose.modelling.switchpoint.qlearning``: the things that must be true of the
model regardless of which animal it is fitted to, and that a plausible-looking fit would hide
if they broke. It generates every sequence it uses, so it runs anywhere, including offline.

Checks
------
1. **closed form == sequential update.** ``qlearning_trajectory`` evaluates the value recursion
   in closed form (see its module docstring); ``simulate_qlearning`` steps trial by trial. Given
   the same choices they must produce the identical P(SHORT) trajectory. This is the check that
   the fast path is the model.
2. **parameter recovery.** Simulate from known parameters, refit with the shipped multi-start,
   and require the estimates and the recovered P(SHORT) trajectory back.
3. **the constrained floor.** ``qlearn_constrained`` *cannot* hold P(SHORT) below 0.5 in steady
   state: ``Q_long`` decays to 0 within ~``1/alpha`` LONG choices and ``Q0_short >= 0`` keeps
   ``Q_short >= 0``, so once ``Q_long`` has decayed the softmax argument cannot be negative.
   This is a property of the bounds, not of any dataset -- an animal that ends up *below*
   chance for SHORT is outside what this variant can describe, and that must not be discovered
   by accident from a fit. The check also confirms ``qlearn_free`` has no such floor.
4. **reward-scale identifiability.** The claim in the qlearning module docstring, numerically:
   rewards ``(1, 0)`` with ``(alpha, b, Q0_short, Q0_long)`` and rewards ``(1, 1-d)`` with
   ``(alpha, b/d, d*Q0_short + 1-d, d*Q0_long + 1-d)`` give the identical nll -- pointwise, at
   the ML fit, and at the optimum of an independent refit. So ``d`` is not estimable and the
   reward scale is rightly fixed rather than fitted.
5. **kappa = 0 reduces perseveration to free.** The perseveration variant nests
   ``qlearn_free``: same nll at ``kappa = 0``, same simulated draw, and a fitted nll that is
   never worse.

Usage
-----
  python src/hypnose/qc/check_qlearning.py
  python src/hypnose/qc/check_qlearning.py --seed 7      # a different synthetic draw

Exit code 0 == GREEN (all checks pass); 1 == RED. Run in the pinned conda env.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from hypnose.modelling.switchpoint.qlearning import (  # noqa: E402
    QLEARN_VARIANTS,
    R_LONG,
    R_SHORT,
    fit_qlearning,
    qlearning_nll,
    qlearning_trajectory,
    simulate_qlearning,
)

# Recovery tolerances. Parameters are recovered from binary choices only, and alpha trades off
# against Q0, so they are deliberately loose on the parameters and tight on the trajectory:
# what the model *predicts* is far better identified than how it is parameterized. The numbers
# are the worst case over 10 synthetic seeds per setting, with headroom -- tightening them
# would make this check flaky rather than stricter.
_N_RECOVERY_TRIALS = 4000
_ALPHA_RTOL = 0.6       # relative
_B_RTOL = 0.4           # relative
_TRAJECTORY_MAE = 0.02  # mean absolute error of the fitted P(SHORT) vs the generating one

# Fast learners finish their transition within a few dozen trials, so alpha is estimated from
# that handful and is NOT recoverable however long the sequence is. The trajectory still is.
# That is a property of the model, so it is asserted rather than avoided by picking parameters.
_FAST_ALPHA = 0.15

# Steady state for check 3: after this many multiples of 1/alpha LONG choices, Q_long has
# decayed by exp(-10) ~ 4.5e-5. With b <= 50 that bounds the softmax argument's shortfall, and
# so P(SHORT) can undershoot 0.5 by at most ~0.25 * 50 * 4.5e-5 ~ 6e-4.
_DECAY_MULTIPLES = 10
_FLOOR_TOL = 1e-3

# Optimizer noise: two ways of computing the same likelihood, or two multi-starts of the same
# problem, may differ by this much without meaning anything.
_NLL_TOL = 1e-6
_REFIT_NLL_TOL = 1e-3


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# --- 1. closed form == sequential update ---------------------------------------------------


def check_closed_form(rng: np.random.Generator) -> bool:
    """The vectorized trajectory must reproduce the trial-by-trial simulation exactly."""
    worst, worst_case = 0.0, None
    for _ in range(20):
        alpha = float(rng.uniform(1e-3, 1.0))
        b = float(rng.uniform(0.1, 20.0))
        q0_short, q0_long = rng.uniform(-2.0, 2.0, 2)
        kappa = float(rng.uniform(-3.0, 3.0))
        seed = int(rng.integers(1 << 30))
        s, p_sim = simulate_qlearning(600, alpha, b, q0_short, q0_long, kappa, seed=seed)
        # Replaying the SAME choices through the closed form must give the same probabilities.
        p_closed = qlearning_trajectory(s, alpha, b, q0_short, q0_long, kappa)["p_short"]
        err = float(np.max(np.abs(p_sim - p_closed)))
        if err > worst:
            worst, worst_case = err, (alpha, b, q0_short, q0_long, kappa)
    return _report("closed form == sequential update", worst < 1e-12,
                   f"max |p_simulated - p_closed_form| = {worst:.3e} over 20 draws "
                   f"(worst at alpha={worst_case[0]:.4g}, b={worst_case[1]:.4g})")


# --- 2. parameter recovery -----------------------------------------------------------------


def _recover(rng: np.random.Generator, alpha: float, b: float, q0_short: float,
             q0_long: float) -> tuple[dict, float, float, float]:
    """Simulate one sequence from known parameters and refit it; returns the fit and the errors."""
    seed = int(rng.integers(1 << 30))
    s, p_true = simulate_qlearning(_N_RECOVERY_TRIALS, alpha, b, q0_short, q0_long, seed=seed)
    fit = fit_qlearning(s, "qlearn_free", seed=seed % 1000)
    return (fit, abs(fit["alpha"] - alpha) / alpha, abs(fit["b"] - b) / b,
            float(np.mean(np.abs(fit["p_short"] - p_true))))


def check_recovery(rng: np.random.Generator) -> bool:
    """Simulate from known parameters, refit, and require the parameters (and curve) back.

    Only in the regime where they *are* identifiable: a learning rate slow enough that the
    transition from LONG to SHORT spans a few hundred trials, so alpha is estimated from a few
    hundred choices rather than from a few dozen. The fast-learner counterpart is asserted
    separately, and only on the trajectory.
    """
    truths = [
        # (alpha, b, q0_short, q0_long): slow learners whose transition occupies a readable
        # fraction of the sequence -- the regime the animals are actually being tested against.
        (0.010, 6.0, 0.0, 0.9),
        (0.015, 4.0, 0.0, 0.8),
        (0.020, 4.0, 0.0, 0.8),
    ]
    ok = True
    for alpha, b, q0_short, q0_long in truths:
        fit, d_alpha, d_b, mae = _recover(rng, alpha, b, q0_short, q0_long)
        good = d_alpha < _ALPHA_RTOL and d_b < _B_RTOL and mae < _TRAJECTORY_MAE
        ok &= _report(
            f"recovery (alpha={alpha:g}, b={b:g})", good,
            f"alpha {alpha:g} -> {fit['alpha']:.4g} ({d_alpha:+.0%}), "
            f"b {b:g} -> {fit['b']:.4g} ({d_b:+.0%}), "
            f"P(SHORT) MAE = {mae:.4f}, {fit['n_starts_converged']}/{fit['n_starts']} starts "
            f"converged")

    # Fast learner: the transition is over within ~1/alpha trials, so alpha and Q0 become
    # interchangeable and only the trajectory is recoverable. Asserting the trajectory (and
    # NOT alpha) is the point -- it records where a fitted alpha may not be read as a rate.
    fit, d_alpha, d_b, mae = _recover(rng, _FAST_ALPHA, 3.0, 0.0, 0.6)
    ok &= _report(
        f"recovery, fast learner (alpha={_FAST_ALPHA:g}): trajectory only", mae < _TRAJECTORY_MAE,
        f"P(SHORT) MAE = {mae:.4f}; alpha {_FAST_ALPHA:g} -> {fit['alpha']:.4g} ({d_alpha:+.0%}) "
        f"is NOT recovered and is not asserted -- with the transition over in ~{1 / _FAST_ALPHA:.0f} "
        f"trials, alpha trades off freely against Q0 (fitted Q0 = "
        f"({fit['q0_short']:.2f}, {fit['q0_long']:.2f}))")
    return bool(ok)


# --- 3. the structural floor of qlearn_constrained -----------------------------------------


def _steady_state_min_p(alpha: float, b: float, q0_short: float, q0_long: float,
                        s: np.ndarray) -> float:
    """Smallest P(SHORT) over the trials at which Q_long has effectively finished decaying."""
    traj = qlearning_trajectory(s, alpha, b, q0_short, q0_long)
    n_long_before = np.arange(s.size) - np.concatenate(([0], np.cumsum(s)[:-1]))
    steady = n_long_before >= _DECAY_MULTIPLES / alpha
    return float(traj["p_short"][steady].min()) if steady.any() else np.inf


def check_constrained_floor(rng: np.random.Generator) -> bool:
    """``qlearn_constrained`` cannot sit below P(SHORT) = 0.5 in steady state; free can."""
    (a_lo, a_hi), (b_lo, b_hi), (q_lo, q_hi), _ = QLEARN_VARIANTS["qlearn_constrained"]["bounds"]
    worst = np.inf
    for _ in range(400):
        # alpha is drawn log-uniformly so the slow end (where the decay takes longest, and the
        # claim is hardest) is sampled properly rather than swamped by fast learners.
        alpha = float(np.exp(rng.uniform(np.log(max(a_lo, 1e-3)), np.log(a_hi))))
        b = float(rng.uniform(b_lo, b_hi))
        q0_short, q0_long = rng.uniform(q_lo, q_hi, 2)
        n = int(60 / alpha) + 200
        # Two adversarial histories. All-LONG is the worst case for the floor: Q_short is never
        # updated, so it stays at Q0_short and the softmax argument is as small as the bounds
        # permit. The sparse one additionally exercises interleaved updates.
        for s in (np.zeros(n, dtype=np.int64), (rng.random(n) < 0.05).astype(np.int64)):
            worst = min(worst, _steady_state_min_p(alpha, b, q0_short, q0_long, s))
    ok = _report(
        "qlearn_constrained floor", worst >= 0.5 - _FLOOR_TOL,
        f"min steady-state P(SHORT) = {worst:.6f} over 400 draws x 2 adversarial histories "
        f"inside the constrained bounds (must be >= {0.5 - _FLOOR_TOL}; exactly 0.5 is reached "
        f"at Q0_short = 0, which is the tight case)")

    # The complement: the same all-LONG history with Q0_short < 0, which only qlearn_free
    # allows, parks P(SHORT) far below chance forever -- so the floor is the constraint's doing
    # and not an artefact of the construction. This is the empirically relevant case: an animal
    # that stays on LONG is outside what qlearn_constrained can describe at all.
    free_min = _steady_state_min_p(0.05, 5.0, -1.0, 0.5, np.zeros(3000, dtype=np.int64))
    ok &= _report("qlearn_free has no such floor", free_min < 0.5,
                  f"min steady-state P(SHORT) = {free_min:.4f} at Q0_short = -1 "
                  f"(outside the constrained bounds, inside the free ones)")
    return bool(ok)


# --- 4. reward-scale identifiability -------------------------------------------------------


def _rescale(d: float, alpha: float, b: float, q0_short: float, q0_long: float) -> dict:
    """The same model written with rewards ``(1, 1-d)``: b scaled by 1/d, Q0 mapped affinely."""
    return {"alpha": alpha, "b": b / d, "q0_short": d * q0_short + (1 - d),
            "q0_long": d * q0_long + (1 - d), "rewards": (R_SHORT, R_SHORT - d)}


def _refit_rescaled(s: np.ndarray, d: float, n_starts: int, seed: int) -> float:
    """Independent multi-start refit under rewards ``(1, 1-d)``, in the transformed bounds.

    Deliberately written here rather than reusing ``fit_qlearning``: the point is that a fit
    carried out entirely in the rescaled parameterization lands on the same likelihood, so it
    must not share the fitting code's reward constants.
    """
    (a_lo, a_hi), (b_lo, b_hi), (q_lo, q_hi), _ = QLEARN_VARIANTS["qlearn_free"]["bounds"]
    bounds = [(a_lo, a_hi), (b_lo / d, b_hi / d),
              (d * q_lo + (1 - d), d * q_hi + (1 - d)),
              (d * q_lo + (1 - d), d * q_hi + (1 - d))]

    def nll(theta):
        return qlearning_nll(s, theta[0], theta[1], theta[2], theta[3],
                             rewards=(R_SHORT, R_SHORT - d))[0]

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    best = np.inf
    for theta0 in rng.uniform(lo, hi, size=(n_starts, 4)):
        best = min(best, float(minimize(nll, theta0, method="L-BFGS-B", bounds=bounds).fun))
    return best


def check_identifiability(rng: np.random.Generator) -> bool:
    """``d`` only ever enters as ``b * d``, so rescaling the rewards cannot change the nll."""
    s, _ = simulate_qlearning(1500, 0.03, 5.0, 0.0, 0.8, seed=int(rng.integers(1 << 30)))
    ok = True

    # (a) pointwise, over random parameters and several reward scales.
    worst = 0.0
    for _ in range(60):
        alpha = float(rng.uniform(1e-3, 1.0))
        b = float(rng.uniform(0.5, 20.0))
        q0_short, q0_long = rng.uniform(-1.0, 1.0, 2)
        d = float(rng.uniform(0.05, 1.0))
        base, _ = qlearning_nll(s, alpha, b, q0_short, q0_long)
        scaled, _ = qlearning_nll(s, **_rescale(d, alpha, b, q0_short, q0_long))
        worst = max(worst, abs(base - scaled) / max(abs(base), 1.0))
    ok &= _report("identifiability (pointwise)", worst < 1e-9,
                  f"max relative |nll(1,0) - nll(1,1-d)| = {worst:.3e} over 60 random "
                  f"(params, d) pairs")

    # (b) at the ML fit, and against an independent refit done in the rescaled parameterization.
    fit = fit_qlearning(s, "qlearn_free", seed=1)
    for d in (0.25, 0.5):
        scaled, _ = qlearning_nll(s, **_rescale(d, fit["alpha"], fit["b"], fit["q0_short"],
                                                fit["q0_long"]))
        ok &= _report(f"identifiability at the ML fit (d = {d})", abs(scaled - fit["nll"]) < _NLL_TOL,
                      f"nll {fit['nll']:.6f} (rewards 1/0, b = {fit['b']:.4g}) vs {scaled:.6f} "
                      f"(rewards 1/{1 - d:g}, b = {fit['b'] / d:.4g})")
        refit = _refit_rescaled(s, d, n_starts=24, seed=2)
        ok &= _report(f"identifiability of the optimum (d = {d})",
                      abs(refit - fit["nll"]) < _REFIT_NLL_TOL,
                      f"independent refit under rewards (1, {1 - d:g}) reached nll "
                      f"{refit:.6f} vs {fit['nll']:.6f}")
    return bool(ok)


# --- 5. kappa = 0 reduces the perseveration variant to qlearn_free -------------------------


def check_perseveration_nests_free(rng: np.random.Generator) -> bool:
    """At ``kappa = 0`` the perseveration variant is exactly ``qlearn_free``."""
    seed = int(rng.integers(1 << 30))
    s, _ = simulate_qlearning(1200, 0.05, 4.0, 0.0, 0.7, seed=seed)
    ok = True

    # Same likelihood at kappa = 0, for random parameters.
    worst = 0.0
    for _ in range(50):
        params = (float(rng.uniform(1e-3, 1.0)), float(rng.uniform(0.5, 20.0)),
                  *rng.uniform(-1.0, 1.0, 2))
        free, _ = qlearning_nll(s, *params)
        persev, _ = qlearning_nll(s, *params, kappa=0.0)
        worst = max(worst, abs(free - persev))
    ok &= _report("kappa = 0 gives the free likelihood", worst == 0.0,
                  f"max |nll(kappa omitted) - nll(kappa = 0)| = {worst:.3e} over 50 draws")

    # Same generative draw at kappa = 0, from the same seed.
    s_free, _ = simulate_qlearning(500, 0.05, 4.0, 0.0, 0.7, seed=7)
    s_persev, _ = simulate_qlearning(500, 0.05, 4.0, 0.0, 0.7, kappa=0.0, seed=7)
    ok &= _report("kappa = 0 gives the free simulation", bool(np.array_equal(s_free, s_persev)),
                  f"{int(np.sum(s_free != s_persev))} of 500 simulated trials differ")

    # And, because it nests the free variant, its fitted nll can never be worse.
    free_fit = fit_qlearning(s, "qlearn_free", seed=3)
    persev_fit = fit_qlearning(s, "qlearn_perseveration", seed=3)
    ok &= _report("qlearn_perseveration nests qlearn_free",
                  persev_fit["nll"] <= free_fit["nll"] + _REFIT_NLL_TOL,
                  f"nll {persev_fit['nll']:.4f} (kappa = {persev_fit['kappa']:+.4f}) <= "
                  f"{free_fit['nll']:.4f} (free)")
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic draws (default: 0)")
    args = ap.parse_args()

    print(f"Q-learning null model self-check (synthetic data, seed {args.seed})\n")
    checks = [
        ("closed form", check_closed_form),
        ("parameter recovery", check_recovery),
        ("constrained floor", check_constrained_floor),
        ("reward identifiability", check_identifiability),
        ("perseveration nesting", check_perseveration_nests_free),
    ]
    results = {}
    for name, check in checks:
        print(f"--- {name} ---")
        results[name] = bool(check(np.random.default_rng(args.seed)))
        print()

    failed = [name for name, ok in results.items() if not ok]
    print("RESULT:", "FAIL -> " + ", ".join(failed) if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
