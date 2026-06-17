"""Back-compat shim: moved to `hypnose.trial_classification.detect_settings`.

Kept so existing imports keep working during the restructuring; remove once all
call sites point at `hypnose.trial_classification.detect_settings`.
"""
import sys
import hypnose.trial_classification.detect_settings as _moved

sys.modules[__name__] = _moved
