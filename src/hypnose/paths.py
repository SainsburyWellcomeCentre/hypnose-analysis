"""Back-compat shim: moved to ``hypnose.io.paths``.

Kept so existing ``from hypnose.paths import ...`` / ``import hypnose.paths``
call sites keep working during the restructuring. Remove once all imports point
at ``hypnose.io.paths``.
"""
import sys
import hypnose.io.paths as _moved

sys.modules[__name__] = _moved
