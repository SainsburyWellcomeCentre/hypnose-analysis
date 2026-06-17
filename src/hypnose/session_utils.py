"""Back-compat shim: moved to ``hypnose.io.readers``.

Kept so existing ``import hypnose.session_utils`` call sites keep working during
the restructuring. Remove once all imports point at ``hypnose.io.readers``.
"""
import sys
import hypnose.io.readers as _moved

sys.modules[__name__] = _moved
