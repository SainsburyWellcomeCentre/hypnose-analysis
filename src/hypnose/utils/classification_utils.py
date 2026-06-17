"""Back-compat shim: moved to `hypnose.trial_classification.classification_utils`.

Kept so existing imports keep working during the restructuring; remove once all
call sites point at `hypnose.trial_classification.classification_utils`.
"""
import sys
import hypnose.trial_classification.classification_utils as _moved

sys.modules[__name__] = _moved
