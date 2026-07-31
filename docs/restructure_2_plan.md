# restructure_2 — consolidation & reuse plan (target: v2.0.0)

Hand-off plan for the next round of work on `hypnose-analysis`.

**This document lives on `main`.** Branch from `main` when you start the work — the old
`restructure_2` branch is far behind and should not be resumed; treat it as abandoned. This
plan supersedes the version at `restructure_2:docs/restructure_2_plan.md`.

Goal: make the code tidier, faster and reusable across the growing repo family — **without
accidentally changing analysis output**.

All measurements in this document were taken **2026-07-31**. Re-measure before trusting
them; `visualization_utils.py` grew 43% in the month before that date.

---

## 0. Context a fresh session needs

**What v1.0.0 delivered** (done — do NOT redo): package `hypnose` under `src/hypnose/` with
`io/`, `trial_classification/`, `metric_analysis/`, `visualization/`, `utils/`, `qc/`;
terminal entry points in `scripts/`; no back-compat shims, all imports canonical.

**The repo family this now lives in:**

| repo | package | role |
|---|---|---|
| hypnose-analysis → **hypnose-behavior-analysis** | `hypnose` → `hypnose_behavior` | behavioural analysis (this repo) |
| hypnose-somnotate | `hypnose_somnotate` | EEG sleep scoring (done, v1.0.0) |
| hypnose-eeg-analysis | `hypnose_eeg` | EEG analysis (coming) |
| neuropixel analysis | `hypnose_ephys` | ephys (planned) |
| **hypnose-helpers** | `hypnose_helpers` | shared, modality-agnostic utilities (to create) |

**Current scale** (2026-07-31): 49 py files, 30,676 lines.

| file | lines |
|---|---|
| `visualization/` **total** | **16,044** |
| ├ `visualization_utils.py` | 7,283 |
| ├ `movement_analysis_utils.py` | 4,497 |
| ├ `pred_seq_utils.py` | 1,886 |
| ├ `sing_rew.py` | 1,291 |
| └ `valve_poke_plots.py` | 646 |
| `trial_classification/classification_utils.py` | 3,703 |
| `metric_analysis/metrics_utils.py` | 1,839 |
| `io/save_results.py` | 491 |

---

## 1. The QC safety net — use it after every change

`src/hypnose/qc/`, see `qc/README.md`:

- **`regression.py`** — golden-master. Fingerprints `trial_data` + metrics dict for the 6
  coverage sessions in `sessions.yml`, md5-compares to `fixtures/`. On mismatch it reports
  **added/removed/changed columns and metric keys**, so an intended change is easy to confirm.
  `--generate` writes baselines; optional `subjid:date` args limit scope.
- **`verify_scripts.py`** — the same, through the actual CLI scripts (covers arg wiring).
- **`check_imports.py`** — static check for referenced-but-not-imported globals.
- **`validate.py`** — `validate_subject()`, used by the scripts.

**Operating rules**

- Run everything with `~/miniconda3/envs/hypnose-analysis-test/bin/python`. Fixtures are only
  valid in the env recorded in `fixtures/env.json` — as of 2026-07-31 that is
  `py3.12.13 / pandas 3.0.1 / numpy 1.26.4`, and the test env still matches.
- **Byte-identical philosophy.** Pure refactors and moves must keep regression GREEN.
  Intended output changes (schema, vectorisation numerics) get fixtures **regenerated
  deliberately in the same commit**, with the column/metric diff confirming only the
  intended fields changed.
- Commit per logical step; keep the tree GREEN between commits where possible.

**Out of scope (explicit):** do NOT change protocol detection — the
`"odourdiscrimination" in name` string matching stays as-is.

---

## Phase 0 — Decisions to make before touching code

**0.1 Package name.** Repo and distribution become `hypnose-behavior-analysis`; the import
package becomes **`hypnose_behavior`** (not `hypnose_behavior_analysis` — you type it
constantly and "analysis" adds nothing inside an import). Repo name and package name need
not match; `pyproject` already does this today (dist `hypnose-analysis`, package `hypnose`).

**0.2 What goes in hypnose-helpers.** Decide the boundary before extracting, using one test:

> **Would this need to change if you added a third modality?** If yes, it is not a helper.

| belongs in helpers | stays in the modality repo |
|---|---|
| figure styles + `save_figure` | metric definitions |
| data-location resolution *mechanism* | knowledge of what a "session" contains |
| subject/date selector parsing | harp/aeon readers, EDF readers |
| session/subject iteration; `sub-XXX/ses-YY_date-…` path conventions | anything importing a modality repo |
| generic parquet/JSON read-write | |

**Hard constraint: `hypnose_helpers` imports nothing from the family.** Strictly one-way, or
you get cycles the first time someone is lazy.

---

## Phase 1 — Rename  *(~½ day)*

`hypnose-analysis` → `hypnose-behavior-analysis`, `hypnose` → `hypnose_behavior`.

Scope measured 2026-07-31: `hypnose.` ×161, `from hypnose` ×104, `import hypnose` ×8,
`hypnose-analysis` ×14. Mechanical, but do it **before** the helpers split so the new repo is
extracted from correctly-named source, and **before** the big refactors so v2.0.0 isn't
tagged straddling a rename.

Also update: `pyproject.toml` (name, packages.find, scripts), `environment.yml`, README,
notebooks, `qc/fixtures/env.json` if it records the dist name, and the GitHub repo name.

**Risk:** low (mechanical). **Done:** regression GREEN, `check_imports` clean, no
`hypnose.`/`from hypnose ` references remain outside historical notes.

---

## Phase 2 — Extract `hypnose-helpers`  *(1–2 days)*

New repo, minimal dependencies (roughly `matplotlib`, `pyyaml`, `pandas`), installable on any
Python the family uses.

**Seed contents:**

- **Figure styles + `save_figure`** — from `io/save.py` (570 lines). Already proven shared:
  hypnose-somnotate uses it today.
- **Data-location resolution** — from `io/paths.py` (165 lines). The *mechanism* (profiles,
  env vars, precedence) is shared; decide whether the `configs/data_locations.yml` config
  itself moves too or stays per-repo.
- **Subject/date selectors** — normalising `66` / `"066"` / `"sub-066"` / `"66,67"` /
  date ranges. **This is already written twice**: `hypnose_somnotate/io/selectors.py` (with
  tests) and this repo's `_parse_date_input`. Take the somnotate version, it's tested.
- **Session/subject iteration** and derivatives path conventions.
- **Generic parquet/JSON read-write.**

**Two design corrections to make during the move** (both learned from the somnotate integration):

1. **Drop `set_figure_dir_resolver`.** It exists only because `save_figure` hardcodes one
   dataset's layout, so consumers need a hook to escape it. A helpers library owned by no
   dataset should take `fig_dir` as a plain argument and let each consumer resolve its own.
   Keep the hook deprecated during the transition, remove it once both consumers pass `fig_dir`.
2. **Never mutate `rcParams` at import.** `io/save.py` currently runs
   `mpl.rcParams.update(nature_style())` at module scope, which collides with somnotate's
   vendored `configuration.py` writing the same four keys at *its* import — whoever lands last
   wins. Export the styles and an explicit `use_style()`; let consumers apply it at
   figure-creation time (hypnose-somnotate's `io/style.ensure_style()` is the pattern).

**Cost note:** 17 files inside this repo import `io/paths` or `io/save`, plus 4 sites in
hypnose-somnotate. The move is mechanical but touches all of them.

**Risk:** low–med (pure moves + import rewiring). **Done:** regression GREEN in this repo;
hypnose-somnotate green against helpers; helpers has no family dependencies.

---

## Phase 3 — Re-baseline QC  *(~1 hour, do not skip)*

41 commits landed between the previous plan being written and 2026-07-31; fixtures date from
3 July. Run `qc/regression.py` **before any refactor**. If it is already RED from accumulated
drift, regenerate deliberately and commit that separately — otherwise every subsequent diff is
measured against a stale baseline and you lose the signal the whole plan depends on.

---

## Phase 4 — Metrics: single source of truth  *(the big de-bloat — do this before trial classification)*

### 4a. Strip `visualization/` of all metric calculation

`visualization_utils.py` both **imports** `metric_analysis.metrics_utils` **and
re-defines/recomputes metrics** (~27 metric/accuracy/rate-ish functions).

1. Audit every function in `visualization/` — **all 16,044 lines, not just
   `visualization_utils.py`**; `movement_analysis_utils.py` (4,497) needs identical treatment.
   Does it *compute a metric* or only *plot*?
2. Any metric computed inside `visualization/` **moves into `metric_analysis`** (the
   appropriate definitions file from 4b) — add it if it doesn't exist yet (lose no metric); if
   it recomputes something `metric_analysis` already has, delete the recompute and call the
   canonical one.
3. `visualization/` then only **fetches** (reads the saved metric, or calls the
   `metric_analysis` function) and **plots**. No metric math remains.

**Why this comes first:** `visualization_utils.py` has **88 functions, median 17 lines** — it
is not uniformly bloated. The mass is **16 functions over 200 lines** (longest 483), and they
are long because they do data prep *and* metric computation *and* plotting in one. Stripping
the metric math is what actually shortens them. It is also lower-risk than the trial-classification
work and a bigger line win, so it proves the QC loop before the risky phase.

### 4b. Modularise `metric_analysis` — plumbing apart from definitions

`metrics_utils.py` (1,839 lines) mixes I/O, orchestration, merging, saving and definitions.
Split so `metric_analysis/` mirrors `trial_classification/`:

- **Plumbing out:**
  - `load_session_results`, `parse_json_columns` → **`io/`** (pairs with `save_results.py`).
    *Coordinate with Phase 7* — `parse_json_columns` shrinks or disappears once blob columns
    are flattened.
  - `run_all_metrics`, `batch_run_all_metrics_with_merge` → **`metric_analysis/run.py`**.
  - `pool_results_dicts` → **`metric_analysis/merge.py`**.
  - `save_merged_metrics_txt` → **`metric_analysis/summary.py`**;
    `merged_results_output_dir`, `merged_metrics_filename` → **`io/`** (all derivatives-path
    conventions in one place).
- **Definitions split by type** into `metric_analysis/metrics/`. **Propose a grouping and get
  it confirmed before moving anything.** Each file holds short, single-purpose functions. Add
  a small **registry** (list, or a `@metric` decorator) so `run_all_metrics` discovers them —
  then adding a metric is a one-file change.

**Risk:** med for 4a (map every recompute to a canonical metric), low for 4b (pure moves →
metric values unchanged, regression GREEN). **Done:** no metric math in `visualization/`;
`metrics_utils.py` split; regression + a metrics-parity check GREEN.

---

## Phase 5 — Visualization: primitives, then thin plotters

Only meaningful **after 4a**, since the metric math has to be out before real plotting
duplication is visible.

**Measured primitive usage across `visualization/` (2026-07-31):**

```
.plot(  64     .scatter( 69     .errorbar( 20     rolling( 20
.legend( 53    .set_xlabel( 55
.boxplot( 0    .hist( 0    .barh( 0    .bar( 1
```

Two things follow. First, **there are no boxplots today** — a common boxplot helper is new
capability, not deduplication; worth adding, but don't expect it to shrink anything. Second,
the largest real repetition is **axis decoration**: 53 legends and 55 axis labels.

**Target shape:**

```python
# primitives (thin, no metric knowledge)
line(ax, df, x, y, **style)
scatter(ax, df, x, y, **style)
boxplot(ax, df, by, value, **style)
rolling_mean(series, window)          # + SEM/CI band helper
style_axis(ax, xlabel=…, ylabel=…, legend=…, title=…)

# per-metric plotters stay thin and explicit
plot_accuracy(ses, kind="line")   ->  load_metric(...) + primitives
```

**Deliberately avoid** a single `plot_metric(kind, ses)` dispatcher — it accumulates kwargs
for every plot type it supports and becomes a god-function. Thin primitives plus one small
function per metric give the same ergonomics without that.

**Risk:** low–med (plot-only changes don't affect the regression fingerprint, which covers
`trial_data` + metrics — visual output needs eyeballing). **Done:** no metric math in
`visualization/`; primitives used by all plotters; no plot function over ~100 lines.

---

## Phase 6 — Trial classification: dedup + modularise  *(highest risk — do after the QC loop is proven)*

**Problem:** rewarded/unrewarded/timeout is derived **three times independently** —
`classify_trials` (the `completed_sequence_*` frames), `analyze_response_times` (the
`response_time_category` column), and `save_results._derive_outcome` (re-derived from
supply/poke counts). They share no code and can drift. The false-response/false-alarm
latency-bucket logic is duplicated too. The trial loop in `classify_trials` is ~1000 lines
with deeply nested helpers.

**Approach:**

1. **Write the unit tests first** (see cross-cutting) so the refactor is guarded at fine grain.
2. Extract a **pure** `classify_completed_trial(record) -> outcome` (and the FR/FA
   latency-bucket helper) taking a small per-trial record (await_reward_time, supply pulses,
   port-poke windows, response window, sequence_rewarded). No `data`/`events` dicts inside.
3. Point all three sites at it; delete the duplicated branches.
4. Clean the trial loop: one pass builds the per-trial record, classification is pure on it.
5. **Modularise the whole `trial_classification/` package.** Break long multi-purpose functions
   into single-responsibility ones — detect the cue poke, resolve the odor sequence, compute
   poke/valve windows, classify the outcome — rather than several at once. Apply to
   `detect_trials`, `merge`, `run` as well.

**On "shorter functions":** length is the symptom, not the goal. Extract at seams that are
**pure and independently testable**; shortness follows. Splitting purely to reduce line count
produces functions taking twelve arguments, which is worse than what you started with.

**Risk:** high (core logic) — guarded by regression + new unit tests. **Done:** regression
GREEN, 3 sites → 1, no giant multi-purpose blocks left in `trial_classification/`, unit tests
pass.

---

## Phase 7 — Schema & save formats

`trial_data` already saves parquet + CSV. Decisions:

- Standardise on **parquet for tables, JSON for metadata**. **No pickle** for saved outputs
  (version-fragile — the somnotate work is a live example of pickle/version coupling biting).
  Keep a CSV of `trial_data` only for human-readability; if dropped, update `qc/_common.py` to
  read parquet → canonical form.
- **Typed `@dataclass TrialRecord`** for the flat trial table: replace the free-form ~60-key
  dict (with its singular/plural aliases) with explicit typed fields, validation in
  `__post_init__`, and `.to_row()` for the DataFrame.
- **Flatten the JSON-blob columns** (`position_valve_times`, `position_poke_times`,
  `presentations`) into a tidy long-format side-table `position_data` — one row per
  `trial_id × position` with odor / valve_start / valve_end / poke_time_ms.

These two are complementary, not alternatives: the dataclass governs the flat per-trial table,
the side-table replaces the per-position blobs that don't belong in it. Queryable, type-safe,
smaller/faster parquet, kills the alias hacks.

**Intended schema change → regenerate fixtures deliberately.** Phase it: add the side-table
additively, keep blobs during transition, drop blobs last. Couples tightly with Phase 6.

**Risk:** med (touches downstream readers). **Done:** no pickle outputs; `position_data`
side-table exists; blobs removed; fixtures regenerated with only the intended diff.

---

## Phase 8 — Profile, then vectorise

**Profile first — do not guess.** Run one session through
`analyze_session_multi_run_by_id_date` (+ `run_all_metrics`) under `cProfile` (`snakeviz` to
view), then `line_profiler` on the top function. Use **local data** so I/O variance doesn't
dominate.

Likely finding: **data loading** (harp/aeon `.bin` reads, timestamp interpolation, `concat`)
dominates rather than the classification loops — in which case optimise I/O batching, fewer
`concat`s and vectorised event-window math, not sequential event logic for its own sake.

**Risk:** med — vectorisation can produce *almost* (not byte-) identical floats, so expect
some intended RED; the per-column diff localises it and you decide tolerance per case.

---

## Phase 9 — Validation with clear errors

Currently **0 asserts** in classification/metrics. Add checks that "function X succeeded before
Y starts", with messages that aid troubleshooting.

Prefer explicit **`raise ValueError(msg)`** for production preconditions — bare `assert` is
stripped under `python -O`. Reserve `assert` for internal invariants. Optionally later: swap
`print`/`vprint` for the `logging` module with levels.

**Risk:** low (additive). **Effort:** low–med, spread throughout.

---

## Parallel track — time-base audit for ephys/movement alignment

Ensure every saved event carries a **canonical, documented timestamp** suitable for aligning
with electrophysiology and movement data. The pipeline already does harp timestamp
interpolation plus a real-time (UK tz) offset; audit that (a) the time base is consistent and
documented, (b) it is ideally tied to a hardware sync signal, (c) saved outputs expose it.

Unblocks multi-modal alignment. Independent of the rest — can run in parallel.
**Progress:** ~40%.

---

## Cross-cutting

**Unit tests** (`tests/`, or adjacent to `src/hypnose/qc/`): fast, mount-free tests for the
outcome classifier, FR/FA buckets, `_get_single_reward_info`, `_parse_date_input`,
`validate_subject`. **Build the outcome-classifier tests before Phase 6.**

**Lightweight CI** (optional): `check_imports` + unit tests in GitHub Actions on PRs. The
regression stays local — CI can't reach the data.

---

## Afterthought — cross-repo API

Previously planned as a facade (`hypnose.behavior.accuracy(subjid, date)`). **Demoted**: the
repos do largely independent work and don't obviously need to call each other's analysis.

What they *do* need is **well-defined tidy DataFrame loaders** — the
`hypnose_somnotate.io.load_scores()` pattern: forgiving selectors in, one tidy DataFrame out,
identifier columns prepended, fast enough to call across a cohort, downstream computation left
to the caller. If this repo grows an equivalent `load_trials(...)` / `load_metrics(...)` with
the same shape, that probably covers the real cross-repo need without a facade.

Revisit only if a concrete consumer appears.

---

## Suggested order

```
Phase 0   decisions: hypnose_behavior, helpers boundary        blocks everything named
Phase 1   rename                                               ~½ day
Phase 2   extract hypnose-helpers                              1–2 days
Phase 3   re-baseline QC                                       ~1 hour, do not skip
Phase 4   metrics single source of truth (4a then 4b)          the real de-bloat
Phase 5   visualization primitives + thin plotters             only after 4a
Phase 6   trial classification dedup + modularise              highest risk, tests first
Phase 7   schema & formats                                     couples with Phase 6
Phase 8   profile, then vectorise                              evidence-led
Phase 9   validation                                           woven throughout
∥         time-base audit                                      parallelisable
```

After each step: `qc/regression.py` (+ `verify_scripts.py`, `check_imports.py`). GREEN ⇒
commit. Intended change ⇒ regenerate fixtures in the same commit and confirm via the +/−/~
diff. Tag the finished round **v2.0.0**.
