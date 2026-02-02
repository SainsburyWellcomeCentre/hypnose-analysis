from pathlib import Path
from functools import lru_cache
import os

RAW_SUBDIR = "rawdata"
DERIV_SUBDIR = "derivatives"


def get_repo_root() -> Path:
    """
    Returns the root of the hypnose-analysis repository,
    assuming standard src/ layout.
    """
    return Path(__file__).resolve().parents[2]
    # paths.py → hypnose_analysis → src → hypnose-analysis


def _env_path(var_name: str) -> Path | None:
    val = os.getenv(var_name)
    if not val:
        return None
    return Path(os.path.expanduser(os.path.expandvars(val)))


def get_data_root() -> Path:
    env_root = _env_path("HYPNOSE_DATA_ROOT")
    return env_root if env_root is not None else get_repo_root() / "data"

@lru_cache
def get_rawdata_root() -> Path:
    env_root = _env_path("HYPNOSE_RAWDATA_ROOT")
    path = env_root if env_root is not None else get_data_root() / RAW_SUBDIR
    return path.resolve(strict=False)

@lru_cache
def get_server_root() -> Path:
    env_root = _env_path("HYPNOSE_SERVER_ROOT")
    if env_root is not None:
        return env_root.resolve(strict=False)
    rawdata_root = get_rawdata_root()
    return rawdata_root.parent if rawdata_root.name == RAW_SUBDIR else rawdata_root

@lru_cache
def get_derivatives_root() -> Path:
    env_root = _env_path("HYPNOSE_DERIVATIVES_ROOT")
    if env_root is not None:
        return env_root.resolve(strict=False)
    server_root = get_server_root()
    deriv = server_root / DERIV_SUBDIR
    if deriv.exists():
        return deriv
    # fallback for local-only layouts
    return (get_data_root() / DERIV_SUBDIR).resolve(strict=False)