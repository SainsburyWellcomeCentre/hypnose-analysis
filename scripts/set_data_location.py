#!/usr/bin/env python
"""Select the active data location (rawdata / derivatives roots) for this machine.

Profiles live in the committed `configs/data_locations.yml`. This writes your choice
to `configs/data_locations.local.yml` (git-ignored), which `hypnose.io.paths` reads —
so the selection persists across kernels / terminals / reboots and is never committed.

Usage
-----
  python scripts/set_data_location.py server-mac      # activate a profile
  python scripts/set_data_location.py local_1
  python scripts/set_data_location.py --show          # print the resolved roots
  python scripts/set_data_location.py --list          # list available profiles

Note: a running kernel caches the paths — after switching, restart the kernel or call
`hypnose.io.paths.reload()`. Terminal runs pick up the new choice automatically.
Any HYPNOSE_* env var still overrides this (that's how the QC sandbox / CI work).
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hypnose.io import paths

# These override the active profile (RAWDATA/DERIVATIVES/SERVER) or change the fallback (DATA_ROOT).
_ENV_VARS = ["HYPNOSE_RAWDATA_ROOT", "HYPNOSE_DERIVATIVES_ROOT", "HYPNOSE_SERVER_ROOT", "HYPNOSE_DATA_ROOT"]


def _report_env_overrides() -> None:
    """If any HYPNOSE_* env var is set, flag it and print the OS-correct removal commands.
    (A script can't unset the parent shell's env, so we tell you exactly what to run.)"""
    set_vars = {v: os.environ[v] for v in _ENV_VARS if os.environ.get(v)}
    if not set_vars:
        return
    print("\n  ⚠ HYPNOSE_* environment variable(s) are set and take precedence over the active profile:")
    for v, val in set_vars.items():
        print(f"      {v} = {val}")
    print("  Remove them so the profile drives the paths (a script can't unset your shell's env):")
    if sys.platform.startswith("win"):
        print("    # this session:")
        for v in set_vars:
            print(f"      Remove-Item Env:{v} -ErrorAction SilentlyContinue")
        print("    # permanently (User scope) — then restart the shell / fully restart VS Code:")
        for v in set_vars:
            print(f'      [Environment]::SetEnvironmentVariable("{v}", $null, "User")')
    else:
        print("    # this session:")
        print("      unset " + " ".join(set_vars))
        print("    # permanently: delete the matching 'export ...' lines from ~/.zshrc / ~/.bashrc")
    print("    (then re-run --show to confirm)")


def _print_resolved() -> None:
    paths.reload()
    active = paths.get_active()
    raw = paths.get_rawdata_root()
    print(f"  active profile : {active or '(none — using legacy symlink/fallback)'}")
    print(f"  rawdata        : {raw}      {'OK' if raw.exists() else 'MISSING ⚠'}")
    print(f"  server         : {paths.get_server_root()}")
    print(f"  derivatives    : {paths.get_derivatives_root()}")
    if not raw.exists():
        print("  ⚠ rawdata path does not exist — wrong profile for this machine, or the drive/mount is not available.")
    if active and any(os.environ.get(v) for v in ("HYPNOSE_RAWDATA_ROOT", "HYPNOSE_DERIVATIVES_ROOT", "HYPNOSE_SERVER_ROOT")):
        print("  ⚠ an env var is overriding your selected profile (see below).")
    _report_env_overrides()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", nargs="?", help="profile name to activate (see --list)")
    ap.add_argument("--show", action="store_true", help="print the currently resolved data roots")
    ap.add_argument("--list", action="store_true", help="list available profiles")
    args = ap.parse_args()

    profiles = paths.load_profiles()

    if args.list:
        active = paths.get_active()
        if not profiles:
            print("No profiles found in configs/data_locations.yml.")
            return 1
        print("Available data-location profiles:")
        for name, prof in profiles.items():
            mark = " (active)" if name == active else ""
            print(f"  - {name}{mark}: rawdata={prof.get('rawdata')}")
        return 0

    if args.show:
        print("Resolved data location:")
        _print_resolved()
        return 0

    if not args.profile:
        ap.error("give a profile name to activate, or use --show / --list")
    if args.profile not in profiles:
        print(f"Unknown profile '{args.profile}'. Available: {', '.join(profiles) or '(none)'}")
        return 1

    paths.set_active(args.profile)
    print(f"Active data location set to '{args.profile}' (written to {paths._local_path()}).")
    _print_resolved()
    print("\n(If a Jupyter kernel is running, restart it or call hypnose.io.paths.reload() to pick this up.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
