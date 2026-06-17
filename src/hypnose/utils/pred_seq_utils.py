"""Back-compat shim: moved to `hypnose.visualization.pred_seq_utils`.

Kept so existing imports keep working during the restructuring; remove once all
call sites point at `hypnose.visualization.pred_seq_utils`.
"""
import sys
import hypnose.visualization.pred_seq_utils as _moved

sys.modules[__name__] = _moved
