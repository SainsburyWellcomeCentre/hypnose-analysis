# qc — quality control

Tools to run **after major changes** to confirm the pipeline output is unchanged
(or to mark exactly what changed, if intended). Run them in the pinned conda env
recorded in [`fixtures/env.json`](fixtures/env.json). They read the real,
read-only `rawdata` and redirect all derivatives I/O to a throwaway temp dir
(`HYPNOSE_DERIVATIVES_ROOT`), so they never write to the server.

| Tool | Purpose |
| --- | --- |
| [`regression.py`](regression.py) | Golden-master value check: `trial_data` + metrics vs stored fixtures |
| [`verify_scripts.py`](verify_scripts.py) | Same, but through the actual `scripts/` CLIs (covers arg wiring) |
| [`check_imports.py`](check_imports.py) | Static check: flag any referenced global that isn't imported |
| [`validate.py`](validate.py) | `validate_subject()` — pre-flight data-existence check used by the scripts |

## What `regression.py` checks

For each session in [`sessions.yml`](sessions.yml) it fingerprints the two
outputs that must not change:

- **`trial_data`** — md5 of the canonical CSV (sorted columns, no index; not
  parquet bytes, not the manifest/summary with their timestamps).
- **metrics** — md5 of the metrics dict returned by `run_all_metrics`.

Each fixture also stores a **per-column** md5 (trial_data) and a **per-metric-key**
md5 (metrics). The overall md5 is the pass/fail signal; on a mismatch these let
the report say exactly *what* changed:

```
[RED]  sub-053 20260528 metrics: expected 6ac8e236 got a1b2c3d4
      + added metric: false_response_rate
      ~ changed metric: choice_accuracy
```

## Standard workflow

```
1. (baseline) add/keep the sessions you care about in sessions.yml, then capture
   the current known-good output:
       python src/hypnose/qc/regression.py --generate

2. make your major change(s)

3. compare against the baseline:
       python src/hypnose/qc/regression.py        # exit 0 = GREEN, 1 = RED

4. GREEN  -> commit.
   RED    -> read the +/-/~ lines:
              * unintended change -> fix it.
              * intended change   -> regenerate the affected fixtures deliberately
                                     (step 1) in the SAME commit, so the new
                                     baseline and the change land together.
```

Run `verify_scripts.py` and `check_imports.py` as additional gates the same way.

## Adding / removing sessions

- **Add a session:** put `{subjid, date, label}` in `sessions.yml`, then generate
  its fixture. You can limit `--generate` to specific `subjid:date` keys so you
  don't recompute the others:
  ```
  python src/hypnose/qc/regression.py --generate 060:20260601
  ```
  Generate on a **known-good baseline** (before your changes) — `--generate`
  records whatever the current code produces.
- **Remove a session:** delete it from `sessions.yml`. Its fixture file is left in
  `fixtures/` but ignored (compare/generate only act on sessions in `sessions.yml`,
  intersected with any keys you pass). Delete the orphan fixture by hand if you like.

## Notes

- All pipeline imports live in one block in [`_common.py`](_common.py) — the single
  place to update if module locations change; the md5s must not change as a result.
- Fixtures are only valid in the environment recorded in `fixtures/env.json`
  (float formatting depends on pandas/numpy versions). Regenerate if you change env.
- `check_imports.py` skips the runnable qc tools in its default scan; pass a module
  or file path to check anything specifically.
