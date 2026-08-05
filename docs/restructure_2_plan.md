# restructure_2 — consolidation & reuse plan (target: v2.0.0)

Hand-off plan for the next round of work on `hypnose-analysis`.

**The restructure is planned on the new branch `hypnose-restructure`.** 

Goal: make the code tidier, faster and reusable across the growing repo family — **without
accidentally changing analysis output**.

All measurements in this document were taken **2026-07-31**. .

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

**Current scale** — re-measured **2026-08-04**, after Phase 2 (the 2026-07-31 figures the
rest of this document quotes are superseded by these): 54 py files, 31,140 lines.

| file | lines | 2026-07-31 |
|---|---|---|
| `visualization/` **total** | **16,627** | 16,044 |
| ├ `visualization_utils.py` | 7,264 | 7,283 |
| ├ `movement_analysis_utils.py` | 4,460 | 4,497 |
| ├ `pred_seq_utils.py` | 1,886 | 1,886 |
| ├ `sing_rew.py` | 1,291 | 1,291 |
| ├ `valve_poke_plots.py` | 646 | 646 |
| ├ `modelling/switchpoint/plots.py` | 638 | (not counted) |
| └ `movement_analysis/sing_rew_movement.py` | 433 | (not counted) |
| `trial_classification/classification_utils.py` | 3,703 | 3,703 |
| `metric_analysis/metrics_utils.py` | 1,817 | 1,839 |
| `metric_analysis/sing_rew_metrics.py` | 440 | (not counted) |

The `visualization/` total *rose* despite Phase 2 removing code: the original 16,044 omitted
`modelling/switchpoint/plots.py` and `movement_analysis/sing_rew_movement.py`. **Phase 4a
must audit those two as well** — they were invisible in the original count.

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

- **Invoke the QC tools by absolute path, with no `cd`** (learned the hard way, 2026-08-04):

  ```bash
  PY=~/miniconda3/envs/hypnose-analysis-test/bin/python
  QC=~/repos/harris_lab/hypnose/hypnose-behavior-analysis/src/hypnose_behavior/qc

  $PY -u $QC/regression.py            # optional: subjid:date ... to limit scope
  $PY -u $QC/check_imports.py
  $PY -u $QC/verify_scripts.py
  ```

  Two reasons, both of which have already cost a session:

  - The `cd <repo> && $PY src/...` form gets stopped at the agent permission layer, and
    the tool reports it as a rejected call — indistinguishable from a hung or slow run.
    None of the three tools needs a working directory: `regression.py` derives
    `HERE`/`REPO` from `__file__` and puts `src/` on `sys.path` itself.
  - `-u` matters when the output is redirected (backgrounded runs, tee to a log). Without
    it Python block-buffers stdout, so a 5-minute run shows nothing at all until it exits
    and there is no way to tell progress from a stall.
- **Byte-identical philosophy.** Pure refactors and moves must keep regression GREEN.
  Intended output changes (schema, vectorisation numerics) get fixtures **regenerated
  deliberately in the same commit**, with the column/metric diff confirming only the
  intended fields changed.
- Commit per logical step; keep the tree GREEN between commits where possible.

**Out of scope (explicit):** do NOT change protocol detection — the
`"odourdiscrimination" in name` string matching stays as-is.

---

## 2. How to work through this plan

### One phase per chat

**Do not attempt this in a single long session.** Start a fresh chat for each phase,
using this document as the handoff. Section 0 exists precisely so a cold session can
pick up without prior context.

Long sessions degrade in exactly the way this work cannot tolerate: dropped items when
summarising, repeated identical tool mistakes, and mislabelled results where the
narration disagrees with the output. On work whose entire premise is *do not silently
change the output*, that failure mode matters more than the convenience of continuity.

Commit at every phase boundary so each new chat starts from a green, known state.

### Progress

Update this table at the end of each phase, in the same commit as the work.

| phase | status | commit | notes |
|---|---|---|---|
| Step 0 — re-baseline QC fixtures | **done** 2026-08-03 | `481110b` | 9 sessions (8 re-run + sub-053 20260520 kept for seqLen 2 & the singrew-name guard). Old sub-040 20251124 fixture was **stale, not drifted** — cb724d5's own code reproduces the new md5. regression / verify_scripts / check_imports all green |
| 0.1 package name decision | **done** 2026-08-03 | `9aad717` | Decided `hypnose_behavior` (dist `hypnose-behavior-analysis`); realised in Phase 1 |
| 0.2 helpers boundary decision | **done** 2026-08-03 | `9793cbc`..`b840ba1` | Decided by the "knows the data vs knows the layout" test; realised in 2a/2b/2c |
| 0.3 collapse loaders/readers | **done** 2026-08-03 | `5d9c14a` | `readers.py` is now the single definition site for the 8 primitives (tolerant bodies); `loaders.py` re-exports them. Kept as two files: deleting `readers.py` would make `loaders → detect_settings → loaders` a cycle. Dead `create_unique_series` / `find_session_roots` deleted. regression GREEN |
| 1 rename | **done** 2026-08-03 | `9aad717` | `hypnose` → `hypnose_behavior`, dist/repo → `hypnose-behavior-analysis`. 210 anchored replacements + `git mv`. No reinstall needed (editable install is a static-path `.pth` onto `src/`). `HYPNOSE_*` env vars, ceph data paths, Jupyter kernel names and this doc deliberately untouched. GitHub repo renamed, remote URL updated, local folder renamed to `hypnose-behavior-analysis`. That move invalidated the editable-install `.pth` in 6 conda envs; only `hypnose-analysis-test` was repointed — **`hypnose`, `hypnose-analysis`, `hypnose-somnotate`, `sleap`, `sleap-2` still need `pip install -e .`** |
| 2a helpers extraction | **done** 2026-08-03 | `9793cbc`..`b840ba1` (+ helpers `1333955`..`d11d0dc`, somnotate `9e3c155`..`7de20a5`) | `hypnose-helpers` created (local only, no remote yet). Moved: `io/paths` → `DataLocations(config_dir=…)` (could NOT move whole — `__file__`-derived repo root), `io/save` → `viz/styles` + `viz/save` (`save_figure` takes `fig_dir`), `io/save_results` serialisation → `io/serialize`, somnotate `io/selectors` + `ensure_style`. rcParams no longer mutated at import. Behaviour repo −1226 lines. Regression GREEN at each step |
| 2b canonical session discovery | **done** 2026-08-04 | `312854d`..`2fbd5f5` (+ helpers `8dadcf8`, somnotate `80afbda`) | `hypnose_helpers.io.layout` owns the walking; each repo binds its own roots. All 17 session lookups and ~30 subject globs repointed. **`session_index` ships but is deliberately unused by the plotters** — see "State at the end of 2b". regression / verify_scripts / check_imports all green |
| 2c figure provenance | **done** 2026-08-04 | `7d117b8` (+ helpers `059c652`, somnotate `892e56c`) | `hypnose_helpers.provenance` + `viz/metadata`. Every `save_figure` PDF carries commit/version/caller/params; `read_figure_metadata` reads it back **without pypdf**. Phase 7a can call `provenance()` directly. regression / verify_scripts / check_imports all green |
| 3 re-baseline QC | **done** — superseded by Step 0 | `481110b` | Step 0 did it first: 9 sessions re-baselined 2026-08-03. Nothing further to do; do NOT re-run `--generate` |
| 4a strip metrics from visualization | **audit done** 2026-08-05; moves not started | `58387ce`..`a25ba1c` | Audit written to `docs/metric_audit.md` — all 7 files, ~220 functions. **32 sites of metric math in 4 files**, resolving to **24 metrics with no canonical version** (checklist in the audit is the "lose no metric" gate), 8 exact duplicates, 9 granularity variants. Biggest single item: `compute_speed_analysis` (591 lines, 7 movement metrics, no plotting) moves wholesale. `modelling/switchpoint/plots.py` needs nothing; `movement_analysis/sing_rew_movement.py` needs only dedup — both are files the original count missed. **Both blocking decisions settled 2026-08-05** — see "D0 resolution" and "Q5 resolution" in the audit. **D0 (metric signature):** every metric gets a pure `f(frame) -> value` core + thin `f(results)` wrapper, in 4a; four tiers, not one shape; tier-1 metrics store numerator/denominator contributions separately (storing a single per-trial value reproduces finding 12); `position_data` is derived at **load** time so legacy sessions need no compat branch; tier-3 bias metrics take an optional `reference`, which gives plotters a `baseline="session"|"window"` option for free. **Q5 (reached@p):** two helpers — `sequence_depth` never filtered, `sampled_positions(only_true_pokes=)` filterable — unfiltered by default, definition B deleted; unblocked for filtering by the §7b TODO. **One deliberate output change:** non-initiated trials leave the metric set (`non_initiated_FA_rate`, `fa_port_ratio_by_odor.with_non_initiated`, `avg_sampling_time_initiation_abortion`, `non_initiation_odor_bias`), requiring `--generate` in that commit — confirm first. Everything else in 4a is value-neutral. Plus 5 local judgement calls at the end of the audit. Docs only: regression / check_imports green, no source touched |
| 4b modularise metric_analysis | not started | | |
| 5 visualization primitives | not started | | after 4a only |
| 6 trial classification dedup | not started | | unit tests first |
| 7a manifest provenance | not started | | |
| 7b schema & formats | not started | | intended output change |
| 8 profile, then vectorise | not started | | |
| 9 validation | not started | | |
| ∥ time-base audit | not started, deferred | | parallelisable |

### Model and reasoning effort

Use **Opus 5 throughout** — the failure modes here are subtle correctness, not
throughput. Vary the effort by phase:

| phase | effort | why |
|---|---|---|
| 0.3 collapse loaders/readers | **max** | judging which of 4 diverged functions is correct; a wrong pick is invisible and regression may not catch it |
| 1 rename | standard | mechanical over ~265 references; needs care, not reasoning |
| 2a extraction | standard–high | the inventory is already decided; mostly execution |
| 2b session API | high | new API, **17** lookup sites (re-measured), duplicate-`ses` semantics |
| 2c figure provenance | high | value summariser has the edge cases |
| 4a metrics audit | **max** | mapping ~27 recomputes to canonical metrics across 16k lines without losing one |
| 4b, 5 | standard–high | mostly moves once 4a has decided what goes where |
| **6 classification dedup** | **max** | riskiest item in the plan — 3 divergent implementations of one rule, ~1000-line function |
| 7–9 | high | schema is deliberate-change territory; profiling is evidence-led |
| any unexpected RED | **max** | always — diagnose before touching anything |

Max effort on the rename is just slow. Standard effort on Phase 6 is how a subtle
behaviour change passes regression on 8 sessions and breaks on the 9th.

### Context strategy for the large audits

Phase 4a spans **16,044 lines** of `visualization/`. That is a context-capacity problem,
not a reasoning one — no effort setting fixes files that do not fit.

Work file by file and **write the audit into the repo as you go**, e.g.
`docs/metric_audit.md`: every function, whether it computes a metric or only plots, and
where the canonical version lives. The moves then happen in a later chat reading that
file instead of re-reading 16k lines — and you get something reviewable before any code
changes. The same approach suits any phase whose inputs exceed one context.

### Handoff prompt for each new chat

Adapt the phase name and paste:

```
I'm continuing a planned restructure of this repo, on branch `hypnose-restructure`.

FIRST: read `docs/restructure_2_plan.md` in full before doing anything. It is the
authoritative plan. Check the Progress table in section 2 for what is already done —
do not redo completed phases.

This chat covers exactly one phase: <PHASE>. Do not start any other phase.

Hard constraints:
1. The ceph mount `/Volumes/harris` is STRICTLY READ-ONLY. Never write, move, rename,
   chmod or delete anything under it. The only thing that may read it is the QC
   regression harness. If you think you need to write there, stop and ask me.
2. Do not explore ceph — no browsing subject folders, no inventorying sessions, no
   `find` over the mount. If it is unavailable, stop and tell me.
3. Run everything with `~/miniconda3/envs/hypnose-analysis-test/bin/python`. Do not
   install, upgrade or remove packages in any conda env — fixtures are only valid in
   the env recorded in `qc/fixtures/env.json`. If something is missing, tell me.
4. All work is in the repo: edit files, run the QC tools, commit.
5. Invoke the QC tools by ABSOLUTE path with `-u` and no `cd` — see "Operating rules"
   in section 1. The `cd <repo> && python src/...` form is blocked at the permission
   layer and surfaces as a rejected call that looks like a hang.

Workflow:
- Tell me what you are about to do and what regression result you expect, before doing it.
- After the change, run `qc/regression.py` (plus `check_imports.py`) and show me the result.
- GREEN → commit. Unexpected RED → stop and diagnose, do not regenerate fixtures to
  make it pass.
- An intended output change gets fixtures regenerated in the same commit, with the
  +/-/~ diff confirming only the intended fields moved — and ask me first.
- At the end: update the Progress table in the plan, and commit that with the work.

Ask me rather than guessing if a decision is not settled in the plan.
```

---

## Phase 0 — Decisions to make before touching code

### 0.1 Package name

Repo and distribution become `hypnose-behavior-analysis`; the import
package becomes **`hypnose_behavior`** (not `hypnose_behavior_analysis` — you type it
constantly and "analysis" adds nothing inside an import). Repo name and package name need
not match; `pyproject` already does this today (dist `hypnose-analysis`, package `hypnose`).

### 0.2 What goes in hypnose-helpers

Decide the boundary before extracting, using one test:

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

**A sharper form of the test**, easier to apply per function:

> **Does it know what the data *is*, or only where it lives and what format it's in?**
> Knows the data (harp streams, odors, EDF channels, sleep stages, trials) → **stays**.
> Knows only layout/format (`sub-XXX/` dirs, parquet-vs-CSV, JSON serialisation) → **helper**.

### 0.3 Collapse `io/loaders.py` vs `io/readers.py`  *(blocks Phase 2)*

These two files **both define the same 8 names**, and they have already diverged
(measured 2026-07-31):

```
identical   SessionData, TimestampedCsvReader, Video, concat_digi_events
DIFFERENT   load, load_csv, load_json, load_video
readers-only: create_unique_series, find_session_roots
imported by:  loaders.py ×8    readers.py ×2
```

Four functions sharing a name with different bodies in the same package. This must be
resolved **before** deciding what moves to helpers — extracting from the wrong copy would
silently freeze the wrong behaviour.

Work out which `load*` variant is canonical, collapse to one file, delete the other, repoint
the imports. Regression will tell you whether the two behaved differently. This is a latent
bug in its own right, independent of the helpers work.

**Risk:** med (two live code paths). **Done:** one definition per name in `io/`; regression
GREEN (or an understood, deliberate diff).

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

## Phase 2 — Extract `hypnose-helpers`  *(3–4 days total)*

### 2a. The extraction  *(1–2 days)*

New repo, minimal dependencies (roughly `matplotlib`, `pyyaml`, `pandas`), installable on any
Python the family uses.

**Extract functions, not files.** Almost nothing moves whole — every candidate file splits
along the 0.2 test. Both repos keep a thinner `io/paths.py` and `io/loading.py` holding only
what knows the data.

**Inventory — `hypnose-analysis`** (line counts 2026-07-31):

| file | verdict |
|---|---|
| `io/paths.py` (165) | **moves whole.** Profile mechanism — `load_profiles`, `get_active`, `set_active`, `get_rawdata_root`, `get_server_root`, `get_derivatives_root`. Pure layout; the cleanest win in either repo. |
| `io/save.py` (570) | **splits ~90/10.** → helpers: `nature_style`, `poster_style`, `presentation_style`, `use_style`, `_resolve_style`, `_presentation_active`, `nice_x_locator`, `set_size`, `strip_legends`, `_format_span`, `_coerce_list`, `_unique_sorted`, `save_figure`. **Stays:** `_resolve_subject_dir`, `_resolve_session_dir`, `resolve_figure_dir` — they glob `sub-{id:03d}_id-*`, i.e. the behaviour layout. |
| `io/save_results.py` (491) | **splits.** → helpers: `_json_safe`, `_json_default`, `_normalize_df_for_io` (generic serialisation). **Stays:** `save_session_analysis_results` (takes a classification dict + data/events). `resolve_derivatives_output_dir` / `_find_rawdata_root` are a judgement call — probably helpers if the derivatives convention is family-wide. |
| `io/loaders.py` (906) | **stays.** `load_all_streams`, `load_experiment_events`, `load_odor_mapping` are harp/aeon and know about odors. |
| `io/readers.py` (145) | **stays** — after the 0.3 collapse. `find_session_roots` may be generic enough to move. |

**Inventory — `hypnose-somnotate`:**

| file | verdict |
|---|---|
| `io/selectors.py` (100) | **moves whole.** Pure parsing, already unit-tested. Supersedes this repo's `_parse_date_input`. |
| `io/style.py` (62) | **moves whole.** `ensure_style` / `active_style` — this *is* the lazy-application pattern helpers should own (see correction 2 below). |
| `io/paths.py` (260) | **splits.** → helpers: `normalize_subjid`, `_iter_subject_dirs`, `_find_subject_dir`, `_parse_session_dir`, `_date_in_filter` (generic `sub-XXX` / `ses-YY_date-…` walking). **Stays:** `RecordingRef`, `find_recordings` (globs `.edf`), `get_eeg_root`. |
| `io/loading.py` (579) | **mostly stays.** → helpers: `load_signal_table` (parquet/CSV dispatch), `list_csv_files`. Everything else is somnotate predictions, stage vectors, `ScoredRef`. |
| `io/save.py` (113) | **mostly disappears.** `resolve_eeg_figure_dir` stays; the `save_figure` wrapper and `_register_resolver` evaporate once helpers' `save_figure` takes `fig_dir` (correction 1). |

**Two more decisions:**

- **`scripts/set_data_location.py`** moves with `io/paths.py` — it is the CLI for that
  mechanism, and becomes `hypnose-helpers`' own entry point, usable from any repo.
- **`configs/data_locations.yml` stays per-repo; only the loader moves.** The mechanism
  (format, precedence, `set_active`) is shared; the *contents* are per-dataset. Note
  hypnose-somnotate currently derives its EEG root from the behaviour profile's `server_root`
  — a coupling worth loosening rather than enshrining in helpers.
- **Canonical session discovery** — see 2b below. This is the single biggest de-duplication
  in the whole plan.
- **Figure provenance** — new capability, see 2c below.

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

### State at the end of 2a *(2026-08-03)* — read this before starting 2b

Five things differ from what this document assumed when it was written. All are done and
committed; they change the ground 2b starts from.

1. **`io/paths.py` did NOT "move whole".** Everything in it derived from `get_repo_root()`
   = `Path(__file__).parents[3]`, so relocating the file silently repointed the config at
   `hypnose-helpers/configs` (absent) and fell through to a wrong rawdata root — no error.
   The mechanism moved as `DataLocations(config_dir=…, data_root=…, env_prefix=…)`; each
   repo instantiates one and re-exports the bound methods. **Any later "move whole" in this
   plan deserves the same suspicion: check for `__file__`-derived state first.**
2. **hypnose-somnotate no longer imports the behaviour repo at all.** It owns
   `configs/data_locations.yml` (EEG profiles, `env_prefix: HYPNOSE_EEG`) and installs with
   hypnose-helpers alone. The plan's "4 sites in hypnose-somnotate" was correct at the time;
   it is now genuinely 0.
3. **`rcParams` are no longer mutated at import**, and `set_figure_dir_resolver` is gone —
   both design corrections from 2a are complete. Two notebooks (`sing_rew_visualization`,
   `trial_classification`) still rely on the old implicit styling and need a `use_style()`
   call; deliberately deferred.
4. **hypnose-helpers exists** at `hypnose/hypnose-helpers` with `io/{paths,selectors,
   serialize}`, `viz/{styles,save}`, `cli/set_data_location`. It imports nothing from the
   family — verify that still holds after every 2b move.

**Environment:** hypnose-helpers must be `pip install -e`'d into any env that runs either
repo. As of 2026-08-03 only `hypnose-analysis-test` has it; `hypnose`, `hypnose-somnotate`,
`sleap` and `sleap-2` might still need it. To be done by user. 

### 2b. Canonical session discovery  *(~1 day — the biggest single de-duplication)*

**Re-measured 2026-08-03** (the counts below supersede the pre-rename figures further down —
those line numbers are stale after the rename and the 2a extractions):

| | plan said | actually, now |
|---|---|---|
| session-directory lookups | 11 | **17** (16 in the behaviour repo, 1 in somnotate) |
| `sub-NNN` zero-pad formatting | "4+ implementations" | **66 sites** |
| files touched by a full repoint | — | **15** |

The `sub-NNN` figure is the surprising one: `f"sub-{str(subjid).zfill(3)}"` appears 66 times,
mostly in `visualization/`. Most are *label* formatting rather than directory discovery, so
they do not all become `find_sessions()` calls — but they do all want the same
`normalize_subjid()` helper, and deciding which is which is part of the phase.

**The layout contract is identical everywhere** — behaviour and EEG both use
`sub-0XX_id-XXX/ses-0XX_date-YYYYMMDD/<modality>/`. Yet "find the session directory for this
subject + date" is currently implemented **11 times independently** (measured 2026-07-31):

```
HA visualization_utils.py:121      HA metric_analysis/metrics_utils.py:50
HA movement_analysis_utils.py:116  HA metric_analysis/metrics_utils.py:189
HA movement_analysis_utils.py:241  HA qc/verify_scripts.py:66
HA io/save.py:379                  HA qc/_common.py:130
HA io/loaders.py:218               HA debug/debug.py:44
HS io/save.py:37
```

Plus **4+ separate implementations** of `sub-*_id-*` subject discovery (`io/save.py:371`,
`utils/helpers.py:82`, `metrics_utils.py:516`, `trial_classification/run.py:460`) and
`sub-{n:03d}` formatting scattered throughout both repos.

**One function replaces all of them:**

```python
find_sessions(subjid, *, ses=None, date=None,
              ses_range=None, date_range=None) -> list[SessionRef]
```

`SessionRef` carries `subject`, `subject_dir`, `ses` (int), `date` (str), `path`, and
`session_index`. Selection is forgiving in the same way the CLI already is —
`66` / `"066"` / `"sub-066"`, single values, lists, `"66,67"`.

**`ses` and `date` are interchangeable selectors.** Both resolve to the same session, because
the directory name carries both. `find_sessions(66, ses="03-09")` works; so does
`find_sessions(66, date_range="20260707-20260718")`. This is what makes
`sub 66 ses 03-09` usable across every function in the family.

**Duplicate `ses` within a subject raises**, naming both candidate directories. Silently
taking the first is exactly the failure that surfaces months later as an unexplained result.
Only `sub-036` (`ses-60` twice, known human error) triggers this today; once fixed it never
fires.

#### `session_index` — the plotting ordinal

`ses` is a good *identifier* but **not** a gap-free ordinal. Measured across all 48 subjects /
2045 sessions:

| | subjid ≤ 35 | subjid ≥ 36 |
|---|---|---|
| duplicates / out-of-order | 8 subjects | **1** (`sub-036`, known) |
| **gaps in `ses` numbers** | 6 subjects | **8 subjects (29%)** |

Gaps are irrelevant for selection — `ses-38` is still unique whether or not `ses-37` exists —
but they break `ses` as an x-axis. `sub-038, 045, 046, 047, 048, 057, 058, 062` all have holes,
and these are *current* subjects, not legacy.

So helpers also provides:

```python
session_index(subjid, date_or_ses) -> int   # rank by date within subject: 1..N, gap-free
```

Use `ses`/`date` to *select*, `session_index` to *plot*. They answer different questions and
neither replaces the other. Add `session_index` as a column at load time so plotters never
recompute it.

**Do this in the same pass as collapsing the 11 sites.** Adding a second lookup key to one
resolver costs about an hour; retrofitting it to 11 call sites later costs days.

**Done:** one session-discovery function in helpers; all 11 date-lookup sites and all subject-
discovery sites repointed; `ses`, `date`, and both range forms accepted; duplicate `ses`
raises with both paths named; `session_index` available and gap-free; regression GREEN.

### State at the end of 2b *(2026-08-04)* — read this before starting 2c or 5

1. **The API is bound, not free-floating.** `hypnose_helpers.io.layout.SessionLayout` takes
   the root as a **callable**, and each repo binds its own:
   `hypnose_behavior.io.layout` exposes `rawdata` and `derivatives`; there is deliberately
   **no bare `find_sessions()`** defaulting to one tree, because rawdata holds every
   recorded session and derivatives only the analysed ones. Passing a resolved `Path`
   would freeze the root and break `qc/_common`'s per-session redirect — the same
   `__file__` trap as 2a, one layer along.

2. **`session_index` is a selector, never a plot axis.** It is on every `SessionRef`,
   gap-free and correct, and since 2026-08-04 it is also a **selector**:
   `find_sessions(62, index_range=(1, 9))` — "this animal's first nine sessions",
   comparable across cohorts recorded months apart, which `ses` cannot express (see
   Phase 5). Use it for that.

   The 8 plotters were *not* retrofitted onto it as an **x-axis**, by decision: they
   count `enumerate(ses_dirs, 1)` **within the filtered selection**, so every plot's x
   starts at 1 no matter which sessions were requested. `session_index` is the animal's
   full-history rank, so a filtered call would plot at x=12,27,33 with a mostly empty
   axis. The plan's original premise — that gaps in `ses` break the x-axis — did not
   apply: no plotter ever used `ses` as x. **Do not "finish the retrofit" in Phase 5.**

   *Selection and positioning are different jobs.* `index` answers "which sessions";
   `enumerate` answers "where on the axis". Conflating them is what would make the plots
   look wrong.

   **A caveat on the gap statistics quoted in this document.** The "29% of current
   subjects have gaps in `ses`" figure was measured 2026-08-03 and is already dated: the
   gaps are mostly in retired subjids, and recent subjects are contiguous bar one. So
   `ses` and `index` usually agree — which is precisely why the difference must stay
   explicit rather than being papered over. Re-measure before relying on that number.

3. **The two central helpers absorbed most of the work.** `utils/helpers._iter_subject_dirs`
   (21 call sites) and `_filter_session_dirs` (36) became thin wrappers, so 57 sites needed
   no edit; only the ~22 that bypassed them were touched. `_filter_sessions` is the new
   `SessionRef`-returning form — prefer it in new code.

4. **Ambiguity now raises.** Duplicate `ses`, duplicate date, or two `sub-NNN_id-*` dirs
   raise `DuplicateSessionError` naming both paths. Consequence: a batch run over *all*
   subjects hard-fails on `sub-036` (duplicate `ses-060`) instead of skipping it. Tolerant
   callers pass `missing_ok=True` or catch.

5. **`qc/_common.py` and `qc/verify_scripts.py` keep their own `**/ses-*_date-*` glob**,
   deliberately: a harness must not share an implementation with the code it validates.
   They are the only remaining hand-rolled session lookups, and that is correct.

6. **Of the 66 `sub-NNN` sites, ~26 were directory tokens** (now `normalize_subjid`); the
   other ~40 are display labels (`Subject 040`, `Sub 040`) and QC fixture filenames, left
   alone. The count was never 66 lookups.

**Testing note:** regression cannot see any of this — it fingerprints `trial_data` and the
metrics dict, not session ordering, selectors, or `session_index`. The real guards are
`hypnose-helpers/tests/test_layout.py` (20 tests, mount-free, runs without pytest) and a
throwaway old-vs-new parity harness (93 comparisons) that caught two silent divergences
before they landed: `dates=[]` flipping from "no sessions" to "all sessions", and
`iter_subjects` widening `sub-*_id-*` to `sub-*`.

### 2c. Embedded figure provenance  *(~½–1 day, new capability)*

Goal: open any saved PDF months later and recover **what it shows and how it was made**, from
the file itself — no sidecar, still one PDF.

**Metadata to embed:** creation timestamp · git commit (+ dirty flag) · package version ·
subjids · dates/session ids · the function that made the figure and the file it is defined in ·
the parameters it was called with.

**Verified constraints** (tested 2026-07-31, matplotlib PDF backend):

- The PDF info dictionary accepts **only** `Title`, `Author`, `Subject`, `Keywords`, `Creator`,
  `Producer`, `CreationDate`, `ModDate`, `Trapped`. Custom keys are **silently dropped** with
  `UserWarning: Unknown infodict keyword`. `CreationDate` wants a `datetime`, not a string.
- So: put a **JSON blob in `Subject`**, and a short human-readable summary in `Title`
  (e.g. `"accuracy sub-040,045,066 20251124–20260203"`). A 244-char blob wrote fine and
  survived round-trip; a realistic provenance record serialises to ~200 chars.

**Auto-capture works** — `inspect.stack()` + `inspect.getargvalues(frame)` recovers the calling
function, its file/lineno, its named arguments *and* its `**kwargs` extras, with no effort at
the call site. Prototyped output:

```json
{"function": "plot_accuracy", "file": "visualization_utils.py", "lineno": 28,
 "params": {"df": "<DataFrame 50x2>", "subjids": [40,45,66],
            "dates": ["20251124","20260203"], "window": 30,
            "kind": "line", "ax": null, "color": "red", "alpha": 0.5}}
```

**Where the work actually is:** not the plumbing — the **value summariser**. Arguments must
become JSON-safe without exploding: scalars/str pass through (truncated), list/tuple/set/dict
truncate at ~10 items, DataFrame/Series/ndarray become descriptors (`<DataFrame 50x2>`,
`<ndarray (100,3) float64>`), anything else becomes `<TypeName>`. Expect a few more cases from
real call sites; budget most of the day here.

**Three design points:**

1. **Frame-walking is fragile.** It takes the first frame not on a skip-list — but after
   Phase 5 introduces thin plotter primitives the real function may be several frames up.
   Give `save_figure` an explicit `provenance=` argument that overrides introspection, and
   treat auto-capture as the fallback.
2. **Record dirty state.** A bare commit hash on a dirty tree points at code that is not what
   ran. Append `-dirty` when `git status --porcelain` is non-empty.
3. **Ship the reader too.** Writing needs no new dependency; reading the info dict needs
   `pypdf` or `pikepdf`. Provide `read_figure_metadata(path) -> dict` with that dependency
   **optional**, or the metadata is write-only in practice.

**Share one provenance helper with the manifest work.** Phase 7's manifest also wants git
commit + package version — write `provenance()` once in helpers and call it from both, rather
than two implementations that drift.

**Done:** every `save_figure` PDF carries recoverable provenance; `read_figure_metadata`
round-trips it; the same helper feeds the manifest.

### State at the end of 2c *(2026-08-04)* — read this before Phase 5 or 7a

0. **`function` is only ever "the nearest frame we did not skip".** In real plotting code
   that is frequently a *local closure* — `movement_analysis_utils` has four nested
   `_save_fig` helpers, and a figure saved through one reports `function: _save_fig`
   rather than the analysis that made it. Found by reading an actual saved figure, not
   by testing. The record therefore also carries **`chain`**, the enclosing function
   names, which is what makes the real caller recoverable. Read `chain` before `function`.

1. **A thin `save_figure` wrapper must skip its own module, or it names itself.** Both
   consumer repos wrap helpers' `save_figure`; the frame walk stops at the *first*
   non-helpers frame, which is the wrapper. Capturing inside the wrapper does not fix
   this — `capture_call()` still returns the wrapper's own frame. The fix is
   `provenance(skip_modules=(__name__,))`, which both wrappers now pass. **Phase 5's
   plotting primitives will reintroduce this**: once `plot_accuracy` calls `line(ax, …)`
   which calls `save_figure`, the primitive's module needs skipping too, or pass
   `provenance=` explicitly. There is a regression test for the wrapper case.

2. **The record is ASCII-only, deliberately.** matplotlib writes a PDF string containing
   any non-ASCII character as UTF-16BE with a BOM. The truncation markers were `…`,
   which silently switched the encoding and broke the reader for exactly the records
   large enough to need truncating. Markers are now `...`, `json.dumps` uses
   `ensure_ascii=True`, and the reader honours a BOM anyway.

3. **Reading needs no dependency.** The plan expected `pypdf` and worried the metadata
   would be "write-only in practice". matplotlib writes the info dictionary
   uncompressed, so a ~15-line stdlib parser recovers it; `pypdf` is used when present
   but is not required. The `pdf` extra remains for PDFs written by other tools.

4. **`package_version` needs `packages_distributions()`.** The import package and the
   distribution differ here (`hypnose_behavior` ← `hypnose-behavior-analysis`), so
   guessing by swapping `_` for `-` returns None. Worth remembering in Phase 7a, which
   wants the same version string in `manifest.json`.

5. **Cost is ~40 ms per saved figure**, almost all of it the two `git` calls. Not cached:
   a stale commit hash in a provenance record is worse than the milliseconds.

6. **`read_figure_metadata` returns `{}` for a figure with no provenance**, so
   `if not read_figure_metadata(p)` is meaningful. matplotlib stamps its own `Creator` on
   every PDF, and returning that alone would have made the emptiness test useless.

**Phase 7a is now mostly done in advance:** call
`hypnose_helpers.provenance.provenance()` and put `commit`/`version` in the manifest.

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

**The registry's precondition is settled — 4a delivers it.** All 28 canonical metrics were
`f(results) -> one session-level number` printing to stdout; a registry over that shape does
not work, and it was also why `visualization/` re-derived the same formulas at other
granularities (the audit's `VARIANT` class). **Decision 2026-08-05: every metric gets a pure
`f(frame) -> value` core plus a thin `f(results)` wrapper, done in 4a** — see "D0 resolution"
in `docs/metric_audit.md`. The registry can therefore assume that shape, with one
qualification: the signature is **not uniform**, it has four tiers (trial-reducible;
`position_data`-grouped; `reference`-normalised; and the non-initiated set, which is being
removed). The registry needs to declare which frame each metric consumes.

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
These thin helpers should live in extra visualization helper py files, so they are shared across
visualization code and can be re-imported in different files. 

**Deliberately avoid** a single `plot_metric(kind, ses)` dispatcher — it accumulates kwargs
for every plot type it supports and becomes a god-function. Thin primitives plus one small
function per metric give the same ergonomics without that.

#### Thread the session selectors through the plotters *(decided 2026-08-04)*

Public plotting functions currently accept **`subjids` and `dates` only**. Phase 2b built
three interchangeable selectors in `hypnose_helpers.io.layout`; the plotters expose none of
them. Widening that interface is a Phase 5 job, because `dates` reaches ~36
`_filter_session_dirs` call sites and changing it earlier would touch files 4a/5 rewrite
anyway.

**Accept all three, and pass them straight through:**

```python
def plot_accuracy(subjids, *, dates=None, ses=None, index=None,
                  date_range=None, ses_range=None, index_range=None, ...):
    for subjid in subjids:
        for session in derivatives.find_sessions(subjid, dates=…, ses=…, index=…):
            ...
```

`find_sessions` already takes exactly these six keywords, so a plotter should forward them
rather than reinterpret them. `utils/helpers._filter_sessions` is the `SessionRef`-returning
form to build on; `_filter_session_dirs` (paths only) is the legacy shim.

**Semantics — none is required, and they combine.** Verified against the implementation:

| given | result |
|---|---|
| all of `dates` / `ses` / `index` are `None` | **every session for the subject** — unchanged from today's `dates=None` |
| exactly one | filter on that key |
| two or more | **intersection** — a session must satisfy *all* of them |
| `[]` (empty list) rather than `None` | **matches nothing** |

So they are *not* mutually exclusive alternatives, and a plotter must not treat them as
"pick one". `find_sessions(66, ses_range=(1, 9), index_range=(3, 5))` legitimately means
"of ses 1-9, the 3rd to 5th sessions chronologically". No validation should reject a
combination.

The `None` vs `[]` distinction is load-bearing rather than incidental: callers build a
per-subject date list and pass it straight through, so a subject with no requested dates
must yield no sessions rather than its whole history. There is a regression test for it.

**Why all three rather than picking one:**

| key | question it answers | when it is the right one |
|---|---|---|
| `date` | "what happened on 7 July" | a specific session you can name |
| `ses` | "session 40 of this animal" | the number in the lab book; stable, quotable |
| `index` | "its first nine sessions" | **comparing subjects across cohorts** |

`ses` and `index` are *not* interchangeable, and the difference is silent. Measured on a
three-subject tree: `ses="01-09"` returns **9, 3 and 0** sessions for a contiguous subject,
one with gaps, and one whose numbering carried over from an earlier protocol — while
`index_range=(1, 9)` returns **9 each**, spanning cohorts months apart. A subject numbered
from `ses-038` yields *nothing* for `ses 1-9` and does not error.

Most current subjects are contiguous (the gaps are mostly in retired subjids — see the note
in "State at the end of 2b"), so `ses` usually behaves like `index`. That is exactly why the
distinction has to be explicit: it works until the one animal where it does not.

**Do not make `index` the plotting x-axis.** It is the animal's full-history rank, so a
filtered call would plot at x=12,27,33. The x-axis stays `enumerate(…, 1)` over the selected
sessions — see "State at the end of 2b" §2. `index` selects; it does not position.

**Risk:** low–med (plot-only changes don't affect the regression fingerprint, which covers
`trial_data` + metrics — visual output needs eyeballing). **Done:** no metric math in
`visualization/`; primitives used by all plotters; no plot function over ~100 lines; every
public plotter accepts `dates`/`ses`/`index` and their range forms.

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

## Phase 7 — Schema, save formats & manifest provenance

### 7a. Manifest provenance  *(quick win, ~½ day)*

Add the **git commit** (`git rev-parse --short HEAD` via subprocess, `"unknown"` on failure,
`-dirty` suffix when `git status --porcelain` is non-empty) and the **package version**
(`importlib.metadata.version("hypnose-behavior-analysis")`) alongside the existing `created_at`
date in `manifest.json`.

Keep these **in the manifest only** — the regression already ignores it, so they never enter
the fingerprint and never cause spurious RED.

**Use the same `provenance()` helper as Phase 2c** (figure metadata) rather than a second
implementation; both want commit + dirty flag + version.

**Risk:** low. **Progress:** ~40% (date exists; commit/version missing).
**Done:** manifest carries commit + version + date; regression unaffected.

### 7b. Schema & save formats

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

- **TODO, from the Phase 4a audit (Q5) — record every presented position, and mark how it
  was derived.** Two data-writing bugs currently make the position record incomplete and
  ambiguous; both surface as per-position metrics that cannot be defined consistently
  (`docs/metric_audit.md`, "Q5 resolution").

  1. **Write the 0 ms / no-poke positions.** A position whose poke registers as ~0 ms is
     currently omitted from `position_poke_times`, `presentations` *and* `num_odors`, even
     though the odor was presented and the sequence advanced through it. Write the position
     with `poke_time_ms = 0` and null `poke_odor_start` / `poke_odor_end`. Until this lands,
     "reached" has to be reconstructed as contiguous `1..max`, which is why the audit
     mandates that form.
  2. **Add `poke_source`** to every position entry: `"poke"` for a genuine poke inside the
     odor window, `"grace"` for one synthesised by the `PRE_ODOR_GRACE_MS` path
     (`classification_utils:1281-1293`, where the poke ended *before* the valve opened),
     `"none"` for a 0 ms / no-poke position. Today a grace entry is indistinguishable from a
     real short poke except by the fragile tell `poke_first_in == poke_odor_start`, and
     animals genuinely poke for under 20 ms — so the marker is the only reliable way to
     separate them. Direct measurement (grace period set to 0) puts grace-derived entries at
     ~2-10 odors per session.

  This unblocks `only_true_pokes` on the sampling metrics. Consumers must treat an **absent**
  `poke_source` as "unknown" and omit the filtered variant, never as "all real pokes" — older
  sessions will never carry the field. Alters `trial_data` ⇒ deliberate fixture regeneration,
  with the diff confirming only the intended columns moved.

  *Coordinate with Phase 6:* the writing happens in `classify_trials`, so it lands naturally
  with the trial-loop cleanup; `position_data` above is where `poke_source` becomes a column.

These two are complementary, not alternatives: the dataclass governs the flat per-trial table,
the side-table replaces the per-position blobs that don't belong in it. Queryable, type-safe,
smaller/faster parquet, kills the alias hacks.

**Intended schema change → regenerate fixtures deliberately.** Phase it: add the side-table
additively, keep blobs during transition, drop blobs last. Couples tightly with Phase 6.

**Risk:** med (touches downstream readers). **Done:** no pickle outputs; `position_data`
side-table exists; blobs removed; fixtures regenerated with only the intended diff.

--- OPTIONAL BONUS, NOT PART OF THE CORE CHANGES: Phase 8 and Phase 9

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

## Parallel track — time-base audit for ephys/movement alignment - Keep this as a note! This will be checked later, as part of it lives within sleap-hypnose (where tracking is done)

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

## Afterthought — cross-repo API - Done later, kept as a TODO

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
Phase 0   decisions + collapse loaders/readers            blocks Phases 1-2
Phase 1   rename                                          ~½ day
Phase 2   extract hypnose-helpers, session API, provenance 3–4 days
Phase 3   re-baseline QC                                  ~1 hour, do not skip
Phase 4   metrics single source of truth (4a then 4b)     the real de-bloat
Phase 5   visualization primitives + thin plotters        only after 4a
Phase 6   trial classification dedup + modularise         highest risk, tests first
Phase 7   manifest provenance, schema & formats           couples with Phase 6
Phase 8   profile, then vectorise                         evidence-led
Phase 9   validation                                      woven throughout
∥         time-base audit                                 parallelisable
```

After each step: `qc/regression.py` (+ `verify_scripts.py`, `check_imports.py`). GREEN ⇒
commit. Intended change ⇒ regenerate fixtures in the same commit and confirm via the +/−/~
diff. Tag the finished round **v2.0.0**.
