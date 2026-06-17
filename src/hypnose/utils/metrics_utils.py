"""Back-compat shim: moved to `hypnose.metric_analysis.metrics_utils`.

Kept so existing imports keep working during the restructuring; remove once all
call sites point at `hypnose.metric_analysis.metrics_utils`.
"""
import sys
import hypnose.metric_analysis.metrics_utils as _moved

sys.modules[__name__] = _moved
