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
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hypnose.io import paths


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
