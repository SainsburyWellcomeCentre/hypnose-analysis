# DECISIONS — settled rules and standing traps

**Read this at the start of every phase.** These are the decisions and traps that outlive the
phase that produced them: each one either prevents a silently wrong number or constrains a
choice a later phase would otherwise make freely. Each entry carries the measurement it rests
on, so you never need the narrative — that is in `git log`.

Nothing here is optional reading because the failure mode is silent. A deleted trap costs a
wrong result that no gate reports.

---

## 1. Metric shape

### D0 — four tiers, not one signature *(4a, 2026-08-05)*

Every metric is a pure `f(frame) -> value` core plus a thin `f(results)` wrapper. The core is
what the registry and the resolvers (`by_group`, `over_windows`) consume. But the signature is
**not uniform** — there are four tiers, and treating them as one is how a variant gets
mis-resolved:

| tier | shape |
|---|---|
| 1 — trial-reducible (13) | `f(trials)`; every resolver works |
| 2 — grouping key inside a JSON blob (8) | `f(trials)` + `f(position_data)` for per-position/odor grouping |
| 3 — normalised by a whole-frame quantity (3) | `f(trials, *, reference=None)`; no `reference` on a rolling call means each window normalises by *itself* |
| 4 — non-initiated trials (4) | **removed, not ported** — dropped from the metric set in 4a |

### Store numerator/denominator contributions, never a per-trial value

**The rule that is easiest to get wrong, and the reason two rolling accuracies disagreed for
years.** A rate is not a per-trial quantity:

```python
num = (rtc == "rewarded")                    # numerator contribution
den = rtc.isin(["rewarded", "unrewarded"])   # denominator contribution
value_over(sl) = num[sl].sum() / den[sl].sum()
```

Storing one number per trial and taking a rolling mean gives `rewarded / window_size` — a
denominator silently containing timeouts and aborts. Any consumer that collects a metric's
contributions must reduce them the way the metric does: `metrics.common.reduce_rate` is public
for exactly that. Mean-type metrics store `(value, included)` and reduce to `sum/count`.

### Three tier-2 traps that change values invisibly

1. **Summation style is part of the metric.** The two pooled `avg_sampling_time_*` metrics
   accumulate `total += x` left to right; `avg_sampling_time_odor_x` calls `np.mean`, which sums
   pairwise. They disagree in the last ULP over a few hundred values — enough to move the
   metrics md5. `_sequential_mean` exists to reproduce the first. **Do not tidy either onto
   `Series.mean()`**, and do not tidy `_mean_sd_by` back onto the pandas reductions (the same
   trap, resolved to `np.mean`/`np.std` so panels 1-4 of `plot_sampling_times_analysis` stay
   byte-identical).
2. **`last_event_index` vs `is_last_event`.** `avg_sampling_time_aborted_sequence` excludes the
   entry whose `index_in_trial` equals `last_event_index`. `presentations` also carries an
   `is_last_event` flag which **agrees on all 9 fixture sessions** — but it is a *different
   rule*, and the code reproduces today's values rather than a rule that happens to match.
3. **`np.int64` counts serialise to JSON as strings.** `json.dumps(..., default=str)` writes
   them as `"3"`, not `3`. Any count returned from a metric must be cast to `int`, or the
   fingerprint moves for a reason that looks like nothing.

---

## 2. `position_data` — filter on the provenance flag matching your blob

`build_position_data` (in `metric_analysis/frames.py`) builds one row per `trial × position`
from the **union** of `position_poke_times`, `presentations` and `position_valve_times` —
because the three do not carry the same positions:

- on a **completed** trial, `position_valve_times` holds every position with a valve activation,
  including ones whose poke registered as ~0 ms, which the other two and `num_odors` all drop;
- on an **aborted** trial, all three are restricted to positions with a poke.

So every row records which blobs it came from: **`in_poke_times` / `in_presentations` /
`in_valve_times`**.

> **Every per-position metric must filter on the flag matching the blob it reads today.**
> Without it, `manual_vs_auto_stop_preference` — which counts valve durations — gains the 0 ms
> positions and changes value.

`poke_source` is deliberately **not** synthesised. Its absence is how `sampled_positions` knows
to omit the `only_true_pokes` variants rather than return the unfiltered value. Treating "no
marker" as "all real pokes" would make old and new sessions look comparable when they are not.

---

## 3. `frames.py` must stay a leaf

`io/load_results.py` calls `build_position_data`, which lives in `metric_analysis/frames.py`.
That looks like `io/ → metric_analysis/` and therefore like a cycle. It is not, for one reason
only:

> **`frames.py` imports nothing from the package** — only `json`, `re`, `typing` and `pandas`.
> Both `io/__init__.py` and `metric_analysis/__init__.py` are docstring-only, so importing a
> submodule triggers no package-level side effects.

`io → metric_analysis.frames` is a one-way edge into a leaf. **The day a metric — or anything
else in the package — is imported into `frames.py`, `io/load_results.py` becomes a real cycle.**
Keep its imports to the standard library and pandas.

*(Settled 2026-08-06, do not re-open: `build_position_data` performs no I/O, so `io/` is the
wrong home by the 0.2 "knows the data vs knows the layout" test, and it shares four helpers with
`sequence_depth` / `reached_counts` / `sampled_positions`. Promoting `frames.py` to a schema
layer below both is the honest fix — revisit only if it grows.)*

---

## 4. The 4b registry contract

`@metric(frame=…)` on the core, `@session_metric(core)` on the printing wrapper. 43 registered,
25 reported.

- **The frame is a decorator argument, not a file boundary** (`frame="trials" |
  "position_data" | "trials+position_data"`). Grouping under `metrics/` is by behavioural
  construct, which is why `fa_latency_from_pokeout` sits with the false alarms and not with the
  other latencies. `MetricSpec.call(results)` is what makes the declaration load-bearing rather
  than decorative.
- **`run.REPORT` holds the report order explicitly**, *because* registration order would make it
  a function of import order — i.e. of a file layout that has already changed twice. Being in
  `REGISTRY` makes a metric discoverable; being in `REPORT` is the separate decision to save it.
- **Only `f(frame) -> value` is registrable.** `fa_port_ratio(n_a, n_b)` and
  `get_fa_ratio_a_stats(subjid, dates)` are deliberately absent — useful functions, not metrics
  over a frame.
- **Re-registering the same function is a reload, not a clash.** The notebooks run under
  `%autoreload 2`, which re-executes a module body on every edit; raising there would make the
  registry unusable exactly where metrics get written. A *different* function claiming a
  registered name still raises, as does an unknown `frame=`.

---

## 5. Load vs compute — `metrics_*.json` is not a plotting input *(2026-08-07)*

**Decision: plotters compute through the registry. `metrics_*.json` / `.txt` stay as the export
and the record of an analysis run.** This deletes the staleness problem rather than managing it:
no provenance stamp, no invalidation rule, no backfill, and no way for two plots to disagree.

Measured before deciding (warm, sub-040 20251124):

| path | per session |
|---|---|
| read `trial_data.parquet` (205 KB) | 6.7 ms |
| `build_position_data` → 1022 rows | 21.9 ms |
| **compute total** | **29 ms** |
| read `metrics_*.json` (17 KB) | 4.2 ms |

7× the time and 12× the bytes — but **both paths already paid the expensive part**. The caller's
walk over the mount happened either way, and that is what costs seconds (14.6 s for one
`derivatives.find_session` on a cold mount, against 0.2 s on rawdata). The cache saves one small
file read and 25 ms of CPU.

**Why the end state "save everything and only load" is unreachable:** three unreported metrics
take a `window` and two take an `fa_types` filter — properties of the *figure*, not the session.
Nine more return per-trial or per-poke tables (`poke_durations` is 739 rows for one session)
that belong with Phase 7b's `position_data` side-table. Only `false_response_ratio` is a
genuinely missing parameter-free scalar.

**Why a provenance stamp was rejected:** stamping the JSON with the commit that wrote it
invalidates the cache on every unrelated commit — a docstring fix would force a re-analysis of
the whole server. The correct key is a hash of the *metric definitions* plus an mtime check
against `trial_data.parquet`. That works, and it is machinery in service of a cache worth 25 ms.

**The real defect it fixes:** `decision_accuracy`, `avg_response_time` and
`FA_avg_response_times` are each obtained **both** ways in `visualization/` today. Two plots can
show the same quantity and disagree.

**Items:**

1. **Phase 5** — route the three dual-sourced quantities through one path. Pick compute.
2. **Phase 5 or 7b** — make `position_data` lazy (22 of the 29 ms, and most metrics never touch
   it), then convert the remaining JSON readers.
3. **Phase 7b** — decide where the nine per-trial tables live; they ride with `position_data`.
4. **Anytime** — `false_response_ratio` into `run.REPORT` if it should be saved (new key on
   every session ⇒ `--generate` in its own commit).

**Caveat:** switching a plotter from load to compute *can* move a curve, for any session whose
saved JSON predates a metric change. That is the staleness surfacing, which is the point — so it
is a `plot_regression`-gated change with a deliberate look at the diffs.

### The trap inside it

**`plot_abortion_and_fa_rates` reads both the numeric and the legacy string form of
`fa_abortion_stats`.** 4b made the metric numeric (counts `int`, rates `float`, positions `int`);
**every `metrics_*.json` on the server still holds the legacy `"3/10 (0.30)"` form**, and that
plotter reads those files directly.

> Tidying the legacy reader away before the tree is re-analysed makes that plot draw **nothing**
> for every session — silently, because the plotter skips what it cannot parse.

The reader may go only once the plotter no longer reads the JSON at all (item 1 above) or the
whole derivatives tree has been re-analysed. Not before, and not as a cleanup.

---

## 6. One truthiness rule

There is exactly one: **`metrics/common._is_truthy`**, widened in 4b so a string that parses as a
non-zero number is truthy. It previously accepted the float `1.0` and rejected its string form
`"1.0"`, while `hr_odor_associations` accepted both — a latent divergence reachable only through
the CSV fallback, where a float column renders `True` as `"1.0"` (measured: 0 disagreements on
all 9 fixture sessions, because both flag columns arrive as native `bool` through parquet).

**This matters the moment anyone adds a flag column.** Use `_is_truthy`; do not write a second
rule, and do not narrow it — widening strictly cannot lose a row, narrowing silently can.

---

## 7. Figures are gated by `qc/plot_regression.py`, not by `regression.py`

`regression.py` fingerprints `trial_data` + the metrics dict and **never sees a figure**. Every
change inside `visualization/` is invisible to it — a plotting refactor can be silently wrong and
stay GREEN.

`qc/plot_regression.py` (added 2026-08-06) runs 32 plotter cases under Agg against a git revision
*and* the working tree, then diffs every line's xy data, collection offsets, patch geometry, axis
decoration and **stdout**. Deliberately a two-tree diff, not a golden master: figures are meant to
change, and the question is always whether *this* change moved a curve.

What it sees that `regression.py` cannot, demonstrated rather than claimed: a `pd.concat` over a
variable a refactor had deleted, swallowed by a bare `except Exception: continue` — every session
would have returned an empty frame silently. And a metric value that is *printed*, not drawn.

Three properties to know before relying on it:

- **It resolves each case's function across an ordered `MODULES` list**, so *moving* a plotter is
  invisible to the diff while a change in what it draws is not. Add new plotter modules to
  `MODULES` or their cases become "not found", which reads as untestable, not as green.
- **It seeds the global RNG (`np.random.seed(0)`) before each call**, because several plotters
  jitter points and never seed it. It also pins `PYTHONHASHSEED=0` and applies `use_style("nature")`
  in the child. All three are *workarounds*: two of them hide real defects (see the plan's Phase 5
  section), and a "both raise, unchanged" case is an ungated one, not a green one.
- **Two non-zero diffs are accepted and recorded:** a sub-nanosecond recovery the trial-timing
  metrics inherit from `e9516e4` (max rel 2.2e-07) and one ULP choice in
  `plot_sampling_times_analysis`.

---

## 8. `session_index` selects; it does not position

`session_index` is on every `SessionRef`, gap-free, and is also a selector
(`find_sessions(62, index_range=(1, 9))` — "this animal's first nine sessions", comparable across
cohorts recorded months apart, which `ses` cannot express).

**Do not make it a plot x-axis, and do not "finish the retrofit" in Phase 5.** The 8 plotters
count `enumerate(ses_dirs, 1)` *within the filtered selection*, so every plot's x starts at 1
whichever sessions were requested. `session_index` is the animal's full-history rank, so a
filtered call would plot at x=12,27,33 with a mostly empty axis. The premise that gaps in `ses`
break the x-axis did not apply — no plotter ever used `ses` as x.

*Selection and positioning are different jobs.* Same distinction as `sequence_depth` vs
`sampled_positions`, and it fails the same silent way if merged.

---

## 9. A `save_figure` wrapper must pass `skip_modules=(__name__,)`

Provenance capture walks the stack and stops at the **first** non-helpers frame — which, for a
repo that wraps `hypnose_helpers.viz.save_figure`, is the wrapper itself. Capturing *inside* the
wrapper does not fix it; `capture_call()` still returns the wrapper's own frame. Both consumer
repos pass `skip_modules=(__name__,)`, and there is a regression test for it.

**Two things reintroduce the hazard:**

- **Phase 5's plotting primitives.** Once `plot_accuracy` calls `line(ax, …)` which calls
  `save_figure`, the primitive's module needs skipping too — or pass `provenance=` explicitly,
  which overrides introspection and is the robust form.
- **The proposed Phase 10 `visualization_utils.py` split.** Moving a plotter between modules
  changes `file` and `chain` in every saved figure's provenance record.

Related: `function` is only ever "the nearest frame we did not skip", frequently a local closure
(`movement_analysis_utils` has four nested `_save_fig` helpers). **Read `chain` before
`function`.**

---

## 10. Phase 7b TODO — the 0 ms positions and `poke_source`

Two data-writing bugs make the position record incomplete and ambiguous, and both surface as
per-position metrics that cannot be defined consistently.

1. **Write the 0 ms / no-poke positions.** A position whose poke registers as ~0 ms is currently
   omitted from `position_poke_times`, `presentations` *and* `num_odors`, even though the odor was
   presented and the sequence advanced through it. Write it with `poke_time_ms = 0` and null
   `poke_odor_start` / `poke_odor_end`.
2. **Add `poke_source`** to every position entry: `"poke"` for a genuine poke inside the odor
   window, `"grace"` for one synthesised by the `PRE_ODOR_GRACE_MS` path
   (`classification_utils:1281-1293`, where the poke ended *before* the valve opened), `"none"`
   for a 0 ms / no-poke position. Today a grace entry is indistinguishable from a real short poke
   except by the fragile tell `poke_first_in == poke_odor_start` — and animals genuinely poke for
   under 20 ms, so the marker is the only reliable separator. Direct measurement (grace set to 0)
   puts grace-derived entries at **~2-10 odors per session**.

Consumers must treat an **absent** `poke_source` as "unknown" and omit the filtered variant, never
as "all real pokes" — older sessions will never carry the field. Alters `trial_data` ⇒ deliberate
fixture regeneration with the diff confirming only the intended columns moved. The writing happens
in `classify_trials`, so it lands naturally with Phase 6's trial-loop cleanup.

### What it unblocks, and why `sequence_depth` looks wrong until then

`only_true_pokes` on the sampling metrics becomes computable, and `sequence_depth` collapses to a
one-line change.

**`sequence_depth` deliberately reproduces *today's* rule, not the `presentations`-sourced target.**
The target says the source is `presentations` and the set is `1..max(presented position)` for every
trial; the canonical metrics instead walk `1..last_odor_position` for an **aborted** trial. Measured
on the 9 fixture sessions, **10 of 1731 trials disagree**, moving `reached` counts on 3 sessions:

```
sub-048  today={1:181, 2:152, 3:117, 4:91, 5:65}   presentations={1:181, 2:153, 3:118, 4:92, 5:65}
sub-057  today={1:338, 2:283, 3:226}               presentations={1:339, 2:287, 3:227}
sub-059  today={1:221, 2:208, 3:139}               presentations={1:221, 2:209, 3:140}
```

The disagreeing trials are precisely the grace artifact. **Switching now would not be more correct
— it would bake that artifact into the denominators of `abortion_rate_positionX` and
`fa_abortion_stats`**, because nothing yet distinguishes a genuine short poke from a synthesised
one. Only after `poke_source` exists can the two sources agree, at which point the
aborted/completed branch collapses. The reasoning and the numbers are in the docstring.

### And why the two position helpers must stay separate

`sequence_depth` ("how far the sequence got") is **never** filtered; `sampled_positions` ("was this
position sampled") **is**. A single filtered `reached_positions` produces physically impossible
sets: dropping a non-`poke` entry from the middle of a trial credits it with reaching position 5
but not position 3, which makes any per-position denominator non-monotonic. A gap is meaningless
for *reached* and perfectly natural for *sampled*.

The contiguous `1..max` fill is doing real work until the fix lands — `sub-057 gid=108` has
`position_poke_times` keys `[2, 3]` and `num_odors=2`, but position 1 *was* presented and its 0 ms
poke was never written. `1..max` recovers it; plain membership loses a real position.
