#!/usr/bin/env python
"""Golden-master regression for the hypnose restructuring.

Each restructuring step must leave the pipeline output byte-for-byte identical.
This script fingerprints ``trial_data`` + the metrics dict for a fixed set of
sessions (``sessions.yml``) and compares against stored fixtures.

Usage
-----
  # write fixtures from the current (baseline) code -- do this ONCE on main:
  python src/hypnose/qc/regression.py --generate

  # check the current code against the fixtures -- after every restructuring step:
  python src/hypnose/qc/regression.py

Exit code 0 == GREEN (all match); 1 == RED (mismatch / failure).

Run in the pinned conda env recorded in fixtures/env.json. Reads real rawdata;
writes only to a throwaway temp dir (never the server).
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO = HERE.parents[2]

# Make both the harness module and the src package importable without install.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import _common  # noqa: E402


def _load_sessions() -> list[dict]:
    with open(HERE / "sessions.yml") as fh:
        data = yaml.safe_load(fh)
    return data["sessions"]


def _fixture_path(subjid, date) -> Path:
    return FIXTURES / f"sub-{str(subjid).zfill(3)}_date-{date}.json"


def generate() -> int:
    sessions = _load_sessions()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"Generating fixtures for {len(sessions)} session(s)...\n")
    failures = 0
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        try:
            fp = _common.fingerprint_session(subjid, date)
        except Exception as e:
            print(f"  [FAIL] sub-{subjid} {date} ({label}): {e}")
            failures += 1
            continue
        payload = {"subjid": str(subjid), "date": str(date), "label": label, **fp}
        _fixture_path(subjid, date).write_text(json.dumps(payload, indent=2))
        print(f"  [OK]   sub-{subjid} {date} ({label}): "
              f"trial_data={fp['trial_data'][:8]} metrics={fp['metrics'][:8]}")
    (FIXTURES / "env.json").write_text(json.dumps(_common.env_fingerprint(), indent=2))
    print(f"\nWrote fixtures to {FIXTURES} (env.json recorded).")
    if failures:
        print(f"{failures} session(s) failed to fingerprint -- fix before relying on fixtures.")
        return 1
    return 0


def compare() -> int:
    sessions = _load_sessions()
    print(f"Checking {len(sessions)} session(s) against fixtures...\n")
    red = 0
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        fpath = _fixture_path(subjid, date)
        if not fpath.exists():
            print(f"  [MISSING FIXTURE] sub-{subjid} {date} ({label}) -- run --generate first")
            red += 1
            continue
        expected = json.loads(fpath.read_text())
        try:
            got = _common.fingerprint_session(subjid, date)
        except Exception as e:
            print(f"  [ERROR] sub-{subjid} {date} ({label}): {e}")
            red += 1
            continue
        for key in ("trial_data", "metrics"):
            if got[key] != expected[key]:
                print(f"  [RED]  sub-{subjid} {date} ({label}) {key}: "
                      f"expected {expected[key][:8]} got {got[key][:8]}")
                red += 1
            else:
                print(f"  [green] sub-{subjid} {date} ({label}) {key} ok ({got[key][:8]})")
    print()
    if red:
        print(f"REGRESSION RED: {red} mismatch(es). Output changed -- a move was not pure.")
        return 1
    print("REGRESSION GREEN: all outputs byte-identical to baseline.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true",
                    help="write fixtures from current code (run on baseline only)")
    args = ap.parse_args()
    return generate() if args.generate else compare()


if __name__ == "__main__":
    raise SystemExit(main())
