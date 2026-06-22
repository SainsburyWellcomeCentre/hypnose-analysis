from pathlib import Path
from functools import lru_cache
import os

RAW_SUBDIR = "rawdata"
DERIV_SUBDIR = "derivatives"

# Resolution order for the data roots (highest priority first):
#   1. HYPNOSE_* environment variables  (deliberate override: CI, the QC sandbox, one-offs)
#   2. the active data-location profile  (configs/data_locations.yml + the per-machine
#      git-ignored configs/data_locations.local.yml selecting `active`)
#   3. legacy fallback: the data/rawdata symlink (and server/derivatives) under the repo
# So everyday use is driven by the config; env vars override it temporarily without
# touching the config file; the symlink remains a fallback.


def get_repo_root() -> Path:
    """
    Returns the root of the hypnose-analysis repository,
    assuming standard src/ layout.
    """
    return Path(__file__).resolve().parents[3]
    # paths.py → io → hypnose → src → hypnose-analysis


def _env_path(var_name: str) -> Path | None:
    val = os.getenv(var_name)
    if not val:
        return None
    return Path(os.path.expanduser(os.path.expandvars(val)))


def get_data_root() -> Path:
    env_root = _env_path("HYPNOSE_DATA_ROOT")
    return env_root if env_root is not None else get_repo_root() / "data"


# --- data-location config ---------------------------------------------------

def _configs_dir() -> Path:
    return get_repo_root() / "configs"


def _profiles_path() -> Path:
    """Committed file with the shared profiles (server-mac, server-windows, local_*, ...)."""
    return _configs_dir() / "data_locations.yml"


def _local_path() -> Path:
    """Per-machine, git-ignored file selecting the `active` profile."""
    return _configs_dir() / "data_locations.local.yml"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # lazy: keep paths.py importable even if yaml is unavailable
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def load_profiles() -> dict:
    """Return {profile_name: {'rawdata': ..., 'derivatives': ...}} from the committed config."""
    return _read_yaml(_profiles_path()).get("profiles", {}) or {}


def get_active() -> str | None:
    """The active profile name: the per-machine local override, else the committed default."""
    active = _read_yaml(_local_path()).get("active")
    if active:
        return active
    return _read_yaml(_profiles_path()).get("default_active")


def set_active(name: str) -> None:
    """Write the active profile to the git-ignored local config (used by scripts/set_data_location.py)."""
    import yaml
    _configs_dir().mkdir(parents=True, exist_ok=True)
    _local_path().write_text(
        "# Per-machine data-location selection (git-ignored). Set via scripts/set_data_location.py\n"
        + yaml.safe_dump({"active": name}, sort_keys=False)
    )


def _active_profile() -> dict | None:
    """Resolved active profile: {'name', 'rawdata', 'derivatives'} or None.
    `derivatives` defaults to the sibling of `rawdata` when not given explicitly."""
    name = get_active()
    if not name:
        return None
    prof = load_profiles().get(name)
    if not isinstance(prof, dict) or not prof.get("rawdata"):
        return None
    raw = str(prof["rawdata"])
    deriv = prof.get("derivatives") or str(Path(raw).parent / DERIV_SUBDIR)
    return {"name": name, "rawdata": raw, "derivatives": str(deriv)}


def reload() -> None:
    """Clear cached path lookups so a changed data-location config (or env var) is picked up
    in a running process. Call after `set_data_location` / `set_active` in a live kernel."""
    for fn in (get_rawdata_root, get_server_root, get_derivatives_root):
        try:
            fn.cache_clear()
        except Exception:
            pass


# --- resolved roots ---------------------------------------------------------

@lru_cache
def get_rawdata_root() -> Path:
    env_root = _env_path("HYPNOSE_RAWDATA_ROOT")
    if env_root is not None:
        return env_root.resolve(strict=False)
    prof = _active_profile()
    if prof is not None:
        return Path(prof["rawdata"]).resolve(strict=False)
    return (get_data_root() / RAW_SUBDIR).resolve(strict=False)  # legacy symlink fallback


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
    prof = _active_profile()
    if prof is not None:
        return Path(prof["derivatives"]).resolve(strict=False)
    server_root = get_server_root()
    deriv = server_root / DERIV_SUBDIR
    if deriv.exists():
        return deriv
    # fallback for local-only layouts
    return (get_data_root() / DERIV_SUBDIR).resolve(strict=False)
