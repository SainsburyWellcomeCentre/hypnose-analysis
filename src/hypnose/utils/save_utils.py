"""Back-compat shim: moved to ``hypnose.io.save``.

Kept so existing ``from hypnose.utils.save_utils import ...`` call sites keep
working during the restructuring. Remove once all imports point at ``hypnose.io.save``.
"""
import sys
import hypnose.io.save as _moved

sys.modules[__name__] = _moved
