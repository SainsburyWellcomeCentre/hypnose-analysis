"""Back-compat shim: moved to ``hypnose.utils.helpers``.

Kept so existing ``from hypnose.helpers import ...`` call sites (including the
underscore-prefixed helpers) keep working during the restructuring. Remove once
all imports point at ``hypnose.utils.helpers``.
"""
import sys
import hypnose.utils.helpers as _moved

sys.modules[__name__] = _moved
