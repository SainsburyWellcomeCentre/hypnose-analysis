"""Switch-point analysis of the LONG -> SHORT strategy change.

The subsystem's numeric core, one module per role:

- ``data``        -- build the continuous SHORT/LONG sequence for an animal.
- ``switch``      -- the switch-point model family (constant / switch / switch2 / logistic).
- ``qlearning``   -- the mechanistic Q-learning account (stub).
- ``compare``     -- fit all models and score them with AIC / BIC.
- ``permutation`` -- the sleep-alignment permutation test.
- ``autocorr``    -- residual-autocorrelation check for the bootstrap's i.i.d. assumption.
- ``bootstrap``   -- parametric bootstrap null (planned).

Figures live separately in ``hypnose.visualization.modelling.switchpoint``; the orchestration
and CLI live in ``scripts/modelling/switchpoint_analysis.py``.

The most-used names are re-exported here for convenience, e.g.
``from hypnose.modelling.switchpoint import fit_switchpoint, compare_models, prepare_subject``.
"""
from hypnose.modelling.switchpoint.data import (
    AB_LETTERS,
    normalize_subjids_dates,
    prepare_subject,
    subject_label,
    subset_by_ab,
)
from hypnose.modelling.switchpoint.switch import (
    WARM_START_LABEL,
    bernoulli_loglik,
    fit_constant,
    fit_logistic,
    fit_logistic_multistart,
    fit_switch2,
    fit_switchpoint,
    logistic_p,
    logistic_start_points,
    posterior_fwhm,
    posterior_hdi,
    switchpoint_loglik_profile,
    switchpoint_posterior,
)
from hypnose.modelling.switchpoint.qlearning import fit_qlearning
from hypnose.modelling.switchpoint.compare import MODEL_ORDER, compare_models, model_fitted_p
from hypnose.modelling.switchpoint.permutation import (
    distance_to_session_start,
    pairwise_f,
    permutation_null_means,
    sample_assignment,
)
from hypnose.modelling.switchpoint.autocorr import (
    ACF_MATERIAL_THRESHOLD,
    ACF_MAX_LAG,
    acf_bounds,
    residual_acf,
)

__all__ = [
    # data
    "AB_LETTERS", "normalize_subjids_dates", "prepare_subject", "subject_label", "subset_by_ab",
    # switch-point family
    "WARM_START_LABEL", "bernoulli_loglik", "fit_constant", "fit_logistic",
    "fit_logistic_multistart", "fit_switch2", "fit_switchpoint", "logistic_p",
    "logistic_start_points", "posterior_fwhm", "posterior_hdi", "switchpoint_loglik_profile",
    "switchpoint_posterior",
    # qlearning
    "fit_qlearning",
    # comparison
    "MODEL_ORDER", "compare_models", "model_fitted_p",
    # permutation test
    "distance_to_session_start", "pairwise_f", "permutation_null_means", "sample_assignment",
    # autocorrelation diagnostic
    "ACF_MATERIAL_THRESHOLD", "ACF_MAX_LAG", "acf_bounds", "residual_acf",
]
