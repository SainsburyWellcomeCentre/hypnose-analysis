"""Back-compat shim: moved to `hypnose.visualization.movement_analysis_utils`.

Kept so existing imports keep working during the restructuring; remove once all
call sites point at `hypnose.visualization.movement_analysis_utils`.
"""
import sys
import hypnose.visualization.movement_analysis_utils as _moved

sys.modules[__name__] = _moved
