# restructure_2 — optimization & consolidation plan (target: v2.0.0)

Hand-off plan for the next round of work on `hypnose-analysis`. Branch: `restructure_2`
(off `main` @ v1.0.0, the restructured codebase). Goal: make the code faster, cleaner,
and reusable across repos — **without accidentally changing analysis output**.

---

## 0. Context a fresh chat needs

**What the v1.0.0 restructure delivered** (already done — do NOT redo):
- Package `hypnose` under `src/hypnose/`: `io/` (paths, readers, loaders, save, save_results),
  `trial_classification/` (classification_utils [core], detect_trials/stage/settings, merge,
  summary, run), `metric_analysis/` (metrics_utils), `visualization/` (visualization_utils,
  pred_seq_utils, movement_analysis_utils, valve_poke_plots), `utils/` (helpers), `qc/`.
- Terminal entry points in `scripts/` (run_trial_classification, run_metrics_analysis, batch_process).
- No back-compat shims remain; all imports are canonical.

**The QC safety net — use it after every change** (`src/hypnose/qc/`, see `qc/README.md`):
- `regression.py` — golden-master: fingerprints `trial_data` (canonical CSV) + metrics dict for
  6 coverage sessions in `sessions.yml`, md5-compares to `fixtures/`. On mismatch it reports the
  **added/removed/changed columns and metric keys** (so an intended change is easy to confirm).
  `--generate` writes baselines (run on a known-good state, before changing); optional `subjid:date`
  args limit which sessions.
- `verify_scripts.py` — same, but through the actual CLI scripts (covers arg wiring).
- `check_imports.py` — static check: flags any referenced global that isn't imported.
- `validate.py` — `validate_subject()` used by the scripts.

**Operating rules:**
- Env: run everything with `~/miniconda3/envs/hypnose-analysis-test/bin/python` (fixtures are only
  valid in the env recorded in `fixtures/env.json`).
- **Byte-identical philosophy:** refactors (#1, #6 moves, #5) must keep regression GREEN. Intended
  output changes (#2, #3 schema, #4 vectorization numerics) get fixtures **regenerated deliberately
  in the same commit**, with the column/metric diff confirming only the intended fields changed.
- Commit per logical step; keep the working tree GREEN between commits where possible.

**Out of scope (explicit):** do NOT change protocol detection — the `"odourdiscrimination" in name`
string matching stays as-is for now (no protocol changes desired).

**File sizes for orientation:** `visualization_utils.py` 5,087 lines (largest!),
`classification_utils.py` 3,580, `metrics_utils.py` 1,789, `save_results.py` 489.

---

## 1. Streamline `classification_utils` — collapse the 3-way outcome duplication  *(highest priority)*

**Problem:** rewarded/unrewarded/timeout is derived **three times independently**:
`classify_trials` (the `completed_sequence_*` frames), `analyze_response_times` (the
`response_time_category` column), and `save_results._derive_outcome` (re-derives from supply/poke
counts). They share no code and can drift; the same false-response/false-alarm latency-bucket logic
is also duplicated. The trial loop in `classify_trials` is ~1000 lines with deeply nested helpers.

**Approach:**
1. Extract a **pure** `classify_completed_trial(record) -> outcome` (and the FR/FA latency-bucket
   helper) that takes a small per-trial record (await_reward_time, supply pulses, port-poke windows,
   response window, sequence_rewarded) and returns the label. No `data`/`events` dict inside.
2. Have all three sites call it. Delete the duplicated branches.
3. Then clean the trial-processing loop: one pass builds the per-trial record, classification is pure
   on the record.
4. **Modularize the whole `trial_classification/` package, not just the dedup:** break long,
   multi-purpose functions into **short, single-responsibility** ones (each does *one* aspect of trial
   processing — e.g. detect the cue poke, resolve the odor sequence, compute the poke/valve windows,
   classify the outcome — rather than several at once). The ~1000-line `classify_trials` loop is the
   prime case, but apply the same to `detect_trials`, `merge`, `run` where functions do several things
   at once. Short pure functions are far easier to unit-test (see cross-cutting) and reuse.

**Risk:** high (core logic) — but fully guarded by regression + the new unit tests (build those first).
**Effort:** High. **Progress:** 0% (deferred from v1). **Done:** regression GREEN, 3 sites → 1,
no single function in `trial_classification/` is a giant multi-purpose block, unit tests for the
extracted pure functions pass, files meaningfully shorter.

## 2. Provenance in `manifest.json`  *(quick win)*

Add the **git commit** (`git rev-parse --short HEAD` via subprocess, fall back to "unknown") and the
**package version** (`importlib.metadata.version("hypnose-analysis")`) alongside the existing
`created_at` date. Keep these in the **manifest only** (the regression already ignores it) so they
never enter the fingerprint.

**Risk:** low. **Effort:** Low (~½ day). **Progress:** ~40% (date exists, commit/version missing).
**Done:** manifest carries commit + version + date; regression unaffected.

## 3. Save formats & cleaner schema

`trial_data` already saves **parquet + CSV**. Decisions:
- Standardize on **parquet for tables + JSON for metadata**. **Do not use pickle** for saved outputs
  (version-fragile). Keep a CSV of `trial_data` only if you still want human-readability; if dropped,
  update `qc/_common.py` to read parquet → canonical form.
- **Bigger win — flatten the JSON-blob columns** (`position_valve_times`, `position_poke_times`,
  `presentations`) into a tidy long-format side-table (`position_data`: trial_id × position with
  odor/valve/poke fields). See the dataclass note below. This is an **intended schema change** →
  regenerate fixtures; do it **phased** (add the side-table additively, keep blobs during transition,
  drop blobs last). Couples tightly with #1.

**Risk:** med (schema change touches downstream readers). **Effort:** Low–Med. **Progress:** ~50%.

### Dataclass + schema (the design behind #1/#3) — adopt BOTH parts (complementary, not either/or)
- **(a) Typed `@dataclass TrialRecord`** for the *flat* trial table: replace the free-form ~60-key
  trial dict (with its singular/plural aliases) with explicit fields + types, validation in
  `__post_init__`, and a `.to_row()` for the DataFrame.
- **(b) Flatten the JSON-blob columns into a tidy `position_data` side-table** (one row per
  `trial_id × position`: odor, valve_start/end, poke_time_ms, …), removing the nested blobs from
  the trial table.

These work together: (a) governs the flat per-trial table, (b) replaces the per-position blobs that
don't belong in it. Advantages: queryable, type-safe, self-documenting, smaller/faster parquet, kills
the alias hacks, friendly to cross-repo consumers (#A). Both are an intended schema change → phase
them in (add the side-table additively, keep blobs during transition, drop blobs last) and regenerate
fixtures deliberately.

## 4. Profile, then vectorize the hotspots

**Profile first** (don't guess): run one session through `analyze_session_multi_run_by_id_date`
(+ `run_all_metrics`) under `cProfile` (visualize with `snakeviz`); use `line_profiler` on the top
function. Do it on **local data** so I/O variance doesn't dominate. Likely finding: **data loading**
(harp/aeon `.bin` reads, timestamp interpolation, `concat`) dominates, not the 158 classification
loops — so optimize what the profile shows (I/O batching, fewer `concat`s, vectorized event-window
math) rather than vectorizing sequential event logic for its own sake.

**Risk:** med — vectorization can produce *almost* (not byte-) identical floats → expect some intended
RED; the per-column diff localizes it; decide tolerance per case. **Effort:** Med–High. **Progress:** 0%.

## 5. Validation with clear errors

Currently **0 asserts** in classification/metrics. Add checks that "function X succeeded before Y
starts," with clear messages for troubleshooting. **Prefer explicit `raise ValueError(msg)` for
production preconditions** (bare `assert` is stripped under `python -O`); reserve `assert` for internal
invariants. (Optional later: swap `print`/`vprint` for the `logging` module with levels.)

**Risk:** low (additive). **Effort:** Low–Med, spread out. **Progress:** 0%.

## 6. Metrics: single source of truth + modular `metric_analysis`  *(do early — big de-bloat)*

### 6a. Strip `visualization/` of all metric calculation
`visualization_utils.py` (5,087 lines) both **imports** `metric_analysis.metrics_utils` **and
re-defines/recomputes metrics** (~27 metric/accuracy/rate-ish functions).
1. Audit every function in `visualization/` (esp. `visualization_utils.py`): does it *compute a
   metric* or only *plot*?
2. **Any metric computed inside `visualization/` must be moved into `metric_analysis`** (the
   appropriate definitions file from 6b) and computed there — add it if it doesn't already exist
   (don't lose any metric); if it recomputes something `metric_analysis` already has, delete the
   recompute and call the canonical one.
3. `visualization/` then only **fetches** (reads the saved metric, or calls the `metric_analysis`
   function) and **plots** — no metric math left in `visualization/`.

### 6b. Modularize `metric_analysis` — separate plumbing from definitions
`metrics_utils.py` (1,789 lines) mixes I/O, orchestration, merging, saving, and the metric
definitions. Split it so `metric_analysis/` mirrors `trial_classification/`:
- **Move the plumbing out of `metrics_utils`:**
  - `load_session_results`, `parse_json_columns` → **`io/`** (loading saved results; pairs with
    `save_results.py`). *Coordinate with #3:* `parse_json_columns` shrinks/disappears once the blob
    columns are flattened.
  - `run_all_metrics`, `batch_run_all_metrics_with_merge` → **`metric_analysis/run.py`** (orchestration
    + batch; parallels `trial_classification/run.py`).
  - `pool_results_dicts` → **`metric_analysis/merge.py`**.
  - `save_merged_metrics_txt` → **`metric_analysis/summary.py`** (the report; parallels
    `trial_classification/summary.py`); `merged_results_output_dir`, `merged_metrics_filename` →
    **`io/`** (keep all derivatives-output path conventions in one place).
- **Split the metric DEFINITIONS by type** into a `metric_analysis/metrics/` subpackage —
  grouping related metric definitions. **Suggest** a new grouping before moving any of the definitions to a new file. After confirmation, move into new files. Each file holds **short, single-purpose** metric functions. Add a small **registry**
  (a list, or a `@metric` decorator) so `run_all_metrics` discovers and runs them — then adding a new
  metric is a one-file change.

Result: every metric defined **once**, plumbing separated from definitions, `metric_analysis/`
consistent with `trial_classification/`, and all metrics available to the public API (#A).

**Risk:** med (6a: map every recompute to a canonical metric; 6b: mostly pure moves of plumbing/defs
→ metric *values* unchanged, regression GREEN). Coordinate the load/parse moves with #3. **Effort:**
Med–High. **Progress:** 0%. **Done:** no metric math in `visualization/`; `metrics_utils.py` split
into definitions-by-type + run/merge/summary (+ io); regression + a metrics-parity check GREEN.

## A. Cross-repo public API  *(after #6)*

A thin public facade so other repos (ephys, etc.) use `hypnose-analysis` as an editable install:
`hypnose.behavior.accuracy(subjid, date)` → load the saved metric, else compute it, else clearly
report it's missing. Build on `load_session_results` + the consolidated metrics from #6. Keep it a
small, documented namespace (`hypnose/behavior.py` or `hypnose/api.py`).

**Effort:** Med. **Progress:** ~20% (loaders + `run_all_metrics` exist; no clean public namespace).

## B. Time-base audit for ephys/movement alignment  *(foundational — can run in parallel)*

Ensure every saved event carries a **canonical, documented timestamp** suitable for aligning with
electrophysiology and movement data. The pipeline already does harp timestamp interpolation + a
real-time (UK tz) offset; audit that (a) the time base is consistent and documented, (b) ideally tied
to a hardware sync signal, (c) saved outputs expose it. This unblocks multi-modal alignment later.

**Effort:** Med–High. **Progress:** ~40%.

---

## Cross-cutting (do alongside)

- **Unit tests** (`tests/` or `src/hypnose/qc/`-adjacent): fast, mount-free tests for the outcome
  classifier, FR/FA buckets, `_get_single_reward_info`, `_parse_date_input`, `validate_subject`.
  Build the outcome-classifier tests **before** #1 so the refactor is guarded at fine grain.
- **Lightweight CI** (optional): run `check_imports` + unit tests in GitHub Actions on PRs (regression
  stays local — CI can't reach the data).

## Suggested order

1. **Foundations:** #2 provenance · #3 format/schema decision · #4 profiling pass · core unit tests.
2. **#1** dedup the 3-way outcome logic + clean trial processing (typed record).
3. **#6** metrics consolidation → **#A** public API.
4. **#4** targeted vectorization on profiled hotspots · **#5** validation woven throughout.
5. **#B** time-base audit (parallelizable).

After each step: `qc/regression.py` (+ `verify_scripts.py`, `check_imports.py`). GREEN ⇒ commit;
intended change ⇒ regenerate fixtures in the same commit and confirm via the +/-/~ diff. Tag the
finished round **v2.0.0**.
