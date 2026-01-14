from pathlib import Path
from functools import lru_cache

RAW_SUBDIR = "rawdata"
DERIV_SUBDIR = "derivatives"


def get_repo_root() -> Path:
    """
    Returns the root of the hypnose-analysis repository,
    assuming standard src/ layout.
    """
    return Path(__file__).resolve().parents[2]
    # paths.py → hypnose_analysis → src → hypnose-analysis


def get_data_root() -> Path:
    return get_repo_root() / "data"

@lru_cache
def get_rawdata_root() -> Path:
    return (get_data_root() / RAW_SUBDIR).resolve()

@lru_cache
def get_server_root() -> Path:
    rawdata_root = get_rawdata_root()
    return rawdata_root.parent if rawdata_root.name == RAW_SUBDIR else rawdata_root

@lru_cache
def get_derivatives_root() -> Path:
    server_root = get_server_root()
    deriv = server_root / DERIV_SUBDIR
    if deriv.exists():
        return deriv
    # fallback for local-only layouts
    return (get_data_root() / DERIV_SUBDIR).resolve()