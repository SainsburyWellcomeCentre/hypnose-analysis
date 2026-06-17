"""Backward-compatibility shim.

The package was renamed ``hypnose_analysis`` -> ``hypnose`` during the
restructuring. Existing imports such as ``from hypnose_analysis.utils.classification_utils
import ...`` keep working by aliasing the old name to the new package. Update
imports to ``hypnose`` at your convenience; this shim will be removed once all
call sites (notebooks, scripts) have migrated.
"""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("hypnose")
