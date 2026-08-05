# Phase 4a — metric audit of `visualization/`

Working document for restructure_2 Phase 4a (see `docs/restructure_2_plan.md`).

**Goal of 4a:** `visualization/` ends up only *fetching* and *plotting*. Every piece of
metric math either moves into `metric_analysis` (if it is a metric that does not exist
there yet — lose no metric) or is deleted in favour of the canonical version.

**This document is the audit, not the move.** It records, for every function in all seven
`visualization/` files, whether it computes a metric or only plots, and where the canonical
version lives. The moves happen in a later chat reading this file.

Audited against the tree at `f72d201` (Phase 2 complete). Line numbers are from that commit.

---

## How this was produced

1. AST inventory of every function in `visualization/` (name, span, args, call targets,
   aggregation/plot signals) — 7 files, 16,627 lines, ~220 functions including nested ones.
2. Full read of `metric_analysis/metrics_utils.py` + `sing_rew_metrics.py` to build the
   canonical catalog below.
3. Per function, the computation-bearing lines were extracted and read at full fidelity.
   Pure styling/layout was skimmed. Anything with a division, an aggregation, or a
   category mask was read in full.

---

## Taxonomy

Every function gets one **verdict**:

| verdict | meaning |
|---|---|
| `PLOT` | draws only; no data math. Stays. |
| `FETCH` | loads sessions / reads the saved metrics JSON. No math. Stays. |
| `PREP` | parsing, label/odor normalisation, colour maps, date resolution. No metric math. Stays. |
| `DERIVE` | derives a **non-metric** data fact (odor↔reward mapping, trial→display-category, port inference, time axis). Not a 4a move on its own; flagged where it duplicates something. |
| `METRIC` | computes a metric and does not plot. Moves wholesale. |
| `MIXED` | computes a metric **and** plots. Must be split: math out, plotting stays. |

and, where there is metric math, one **action**:

| action | meaning |
|---|---|
| `DEDUP → f` | same definition as canonical `f`. Delete the recompute, call `f`. |
| `VARIANT of f` | **same formula, different granularity or different inputs.** A canonical version exists but is *not* a drop-in substitute. Needs a parameterised canonical, and the divergence must be preserved. |
| `NEW → …` | genuine metric with no canonical version anywhere. Must be **added** to `metric_analysis`, or it is lost. |
| `DISPLAY-AGG` | mean / SEM / cumsum **across subjects or sessions of an already-canonical value**, purely to draw a summary line or error bar. |

### Two standing rules, decided during the audit

**1. `DISPLAY-AGG` is not a 4a move.** Taking the mean±SEM of one metric across the
subjects on a plot is a property of the figure, not of the data. Those belong to Phase 5's
primitives (`rolling_mean(series, window)` + the SEM/CI band helper the plan already
names), not to `metric_analysis`. They are listed so the later chat does not mistake them
for metric math, but they stay in `visualization/`.

**2. `VARIANT` is the dangerous class — never swap one for its canonical blindly.**
Every canonical metric in `metrics_utils.py` takes a whole-session `results` dict and
returns one session-level number. Much of `visualization/` computes the *same formula* at
a different granularity (rolling window, per position, per odor, per day, per HR position)
or off a *different source column*. Replacing such a call with the canonical function
would silently change the figure, and the QC regression **cannot see it** — it fingerprints
`trial_data` and the metrics dict, never a plot. Each `VARIANT` row below spells out the
exact divergence. Preserve it, or change it deliberately and say so.

---

## Where each verdict lands *(decided 2026-08-05)*

`visualization/` should end up containing **only plotting**. That is a stronger target than
"no metric math", and it gives three of the six verdicts a destination of their own:

| verdict | lands in |
|---|---|
| `PLOT` | the plot file |
| `FETCH` | **`visualization/io/metric_loader.py`** (new) — session discovery, `metrics_*.json` loading and protocol lookup come out of the plot files entirely |
| `PREP` | a **shared** module under `visualization/` whenever more than one file needs it; stays local only when genuinely single-use |
| `DISPLAY-AGG` | **one shared helper per pattern**, in the Phase 5 primitives module — the cross-session mean±SEM is currently written out longhand at least six times in `visualization_utils.py` alone |
| `DERIVE` | case by case; noted per row |
| `METRIC` / `MIXED` math | `metric_analysis/` |

The `FETCH` and `DISPLAY-AGG` moves are strictly Phase 5 work (they are not metric math),
but they are recorded here because this audit is what the Phase 5 chat will read.

## Should `metric_analysis` grow a general time-resolved helper?

**Yes — as a signature change plus two small composable resolvers, not as a dispatcher.**

The `VARIANT` class exists for one structural reason: every canonical metric is written as
`f(results) -> one session-level number`, so nothing can reuse it at another granularity and
each plotter re-derives the formula by hand. Give every metric a pure core that takes a
**trial frame** instead of a `results` dict:

```python
def decision_accuracy(trials) -> float:          # pure: no I/O, no printing
    ...
def decision_accuracy_session(results):          # thin wrapper, keeps run_all_metrics working
    return decision_accuracy(results["trial_data"])
```

and every granularity in this audit becomes free, with no new metric definitions:

```python
by_group(decision_accuracy, trials, "last_odor")            # per odor
by_group(decision_accuracy, trials, "date")                 # per day
over_windows(decision_accuracy, trials, window=30, step=1)  # rolling
```

That collapses most `VARIANT` rows below, and it is the same uniform signature 4b's metric
registry needs anyway — so 4a should refactor *towards* it rather than adding parallel
per-granularity functions.

**Three limits, all real, all load-bearing:**

1. **It cannot absorb a variant whose *denominator* changes with granularity.** The rolling
   reward fraction divides by the window size, not by rewarded+unrewarded, so
   `over_windows(decision_accuracy, …)` would draw a visibly different curve. It stays a
   separately named metric. Same for the per-position rates, whose denominator is "reached"
   (finding 2), not the group size.
2. **The position/odor metrics need the tidy table first.** `avg_sampling_time_*` and
   `abortion_rate_positionX` reach into JSON blobs, so they only become `by_group`-able once
   `position_poke_times` / `presentations` are expanded into long format — finding 5, which
   is also Phase 7b's `position_data` side-table. Sequence it: extractor → resolvers.
3. **Do not build a name-dispatched `get_metric(name, granularity=…)`.** That is the same
   failure mode the plan already rejects for `plot_metric(kind, ses)`: it accumulates kwargs
   for every metric it supports and becomes a god-function. A uniform signature plus
   `by_group` / `over_windows` gives the same reach with none of that.

> **Settled 2026-08-05 — yes, as part of 4a.** The shape below is right but not uniform:
> the 28 canonical metrics fall into four tiers, and one of them needs a rule that is easy
> to get wrong. See **"D0 resolution"** for the tiering, the numerator/denominator rule, and
> what stays unchanged.

---

## Canonical catalog — what `metric_analysis` already provides

`metric_analysis/metrics_utils.py` — 28 public metrics, all `f(results) -> …`, all
session-level, all printing to stdout:

| function | returns | definition |
|---|---|---|
| `decision_accuracy` | `(n, denom, acc)` | rewarded / (rewarded + unrewarded) |
| `global_choice_accuracy` | `(n, denom, acc)` | rewarded / (rewarded + unrewarded + FA_time_in) |
| `decision_accuracy_by_odor` | DataFrame by odor | per-odor `decision_accuracy_ab` and `_total` (total includes timeout) |
| `premature_response_rate` | `(n, denom, rate)` | FA_time_in among aborted / n_aborted |
| `response_contingent_FA_rate` | `(n, denom, rate)` | FA_time_in / (FA_time_in + rewarded + unrewarded) |
| `global_FA_rate` | `(n, denom, rate)` | FA_time_in / n_initiated |
| `FA_odor_bias` | dict | per-odor `(FA@odor / aborts@odor) / (total FA / total aborts)` |
| `FA_position_bias` | Series | same, by `last_odor_position` |
| `sequence_completion_rate` | `(n, denom, rate)` | completed / initiated |
| `odorx_abortion_rate` | Series | aborts@odor / presentations@odor |
| `hidden_rule_performance` | `(n, denom, rate)` | (HR success & rewarded) / hit_hidden_rule |
| `hidden_rule_detection_rate` | `(n, denom, rate)` | (not aborted & HR success) / hit_hidden_rule |
| `hidden_rule_counts_by_odor` | dict | per-HR-odor counts + `performance` + `detection_rate` |
| `choice_timeout_rate` | `(n, denom, rate)` | timeout_delayed / completed |
| `avg_sampling_time_odor_x` | Series | mean `poke_time_ms` per odor, from **`position_poke_times`**, completed trials, **no `>0` filter** |
| `avg_sampling_time_completed_sequence` | float | pooled mean over completed positions |
| `avg_sampling_time_aborted_sequence` | float | pooled mean over aborted `presentations`, excluding the abort event |
| `avg_sampling_time_initiation_abortion` | float | mean poke time on non-initiated attempts |
| `abortion_rate_positionX` | Series | aborts@pos / reached@pos (**"reached" definition A**, below) |
| `avg_response_time` | dict | mean `response_time_ms` by category + `"…(Rewarded + Unrewarded)"` |
| `FA_avg_response_times` | dict | mean `fa_latency_ms` per FA subtype |
| `response_rate` | `(n, denom, rate)` | (rewarded + unrewarded) / (rewarded + unrewarded + timeout) |
| `manual_vs_auto_stop_preference` | dict | valve durations ≤1000 vs ≥1000 ms |
| `non_initiated_FA_rate` | `(n, denom, rate)` | FA_time_in among non-initiated |
| `non_initiation_odor_bias` | Series | per-first-odor non-initiation rate / global rate |
| `odor_initiation_bias` | Series | per-odor initiation-abortion share / global share |
| `fa_abortion_stats` | 3 DataFrames | by odor / by position / by odor×position. **Values are pre-formatted strings** (`"3/10 (0.30)"`) |
| `fa_port_ratio_by_odor` | dict | per-odor **(A − B) / (A + B)** over `fa_port`, `fa_type` selectable, non-initiated optional |

`metric_analysis/sing_rew_metrics.py` — single-reward protocol:

| function | returns |
|---|---|
| `is_singrew_session(results)` | bool |
| `_classify_trial(row)` | `(category, subcategory)` for one trial — Hit / Miss / FA / CR / premature |
| `compute_sing_rew_metrics(results)` | per-(sub)category counts + `global_trial_id` lists + validation block |
| `compute_sing_rew_rates(categories)` | hit/fa rate, H′/F′, d′, criterion, balanced accuracy, earned-reward, port accuracy, efficient/early rejection, anticipatory, forfeit, omission, impulsivity, impatience |

**Nothing canonical is time-resolved.** There is no rolling, cumulative, per-day or
per-window metric anywhere in `metric_analysis`. Every such thing found in
`visualization/` is therefore `NEW` by construction, even when it shares a formula with a
session-level canonical.

---

## Headline findings

**1. The FA port ratio is implemented 8 times.** `(A − B) / (A + B)` (or its rescaling
`A / (A + B)`) over `fa_port ∈ {1, 2}` appears in the canonical `fa_port_ratio_by_odor`
plus **7 independent recomputes** across `visualization_utils.py`. They differ only in the
FA-label filter and the slicing key:

| site | formula | FA filter | sliced by |
|---|---|---|---|
| `metrics_utils.fa_port_ratio_by_odor` | (A−B)/(A+B) | `fa_type` arg (default `FA_time_in`) or `all` | odor |
| `plot_abortion_and_fa_rates:2266` | (A−B)/(A+B) | `fa_types` **set** | odor |
| `plot_fa_ratio_a_over_sessions:2829` | **A/(A+B)** | `== FA_time_in` | odor |
| `get_fa_ratio_a_stats:5948` | **A/(A+B)** | `startswith("FA_")` | odor |
| `plot_fa_ratio_by_hr_position:6203` | (A−B)/(A+B) | `fa_types` set | HR odor present, HR position |
| `plot_fa_ratio_by_hr_position:6222,6240` | (A−B)/(A+B) | `fa_types` set | two further HR slices |
| `plot_fa_ratio_by_abort_odor:6601` | (A−B)/(A+B) | `fa_types` set | abort odor, HR trials |
| `plot_fa_ratio_by_abort_odor:6638` | (A−B)/(A+B) | `fa_types` set | abort odor, non-HR trials |

One canonical `fa_port_counts(trials) -> (n_a, n_b)` plus a ratio helper collapses all of
them. The canonical needs (a) a set-valued FA filter and (b) to accept a pre-sliced frame
rather than only a whole `results` dict.

**2. "Reached position p" is defined three different ways.** All three are denominators of
per-position rates, and they do not agree on sessions with variable sequence length:

- **A — `metrics_utils.abortion_rate_positionX` / `fa_abortion_stats`:** aborted trials
  contribute to positions `1..last_odor_position` *inclusive*; completed trials contribute
  to `1..max(position_poke_times.keys())`, **per trial**.
- **B — `plot_position_completion_rate:4357`:** completed trials contribute to
  `1..max(num_odors)` — a **session-wide** maximum applied to every completed trial.
- **C — `plot_false_alarm_rate_by_position:4633`:** any trial contributes to exactly the
  positions listed in its `presentations` JSON.

A and B coincide only when every completed trial has the same sequence length.
**Resolved 2026-08-05 — see "Q5 resolution" below.** Measured on all 9 fixture sessions:
none of the three is correct as written, and the disagreements trace to two data-writing
bugs rather than to a genuine choice of definition.

**3. `fa_abortion_stats` returns formatted strings, so its consumer parses them back.**
`plot_abortion_and_fa_rates:2196-2325` recovers numbers with
`int(s.split()[0])` and `float(s.split("(")[-1].split(")")[0])`, with a `num/denom`
string-split fallback. The fix belongs in `metric_analysis`: return numeric columns (the
formatting is a `summary.py` concern in 4b), then the plotter just reads them.

**4. Sampling-time numbers in `visualization/` are not comparable to the canonical ones.**
Two independent divergences, both silent:
- canonical `avg_sampling_time_odor_x` reads **`position_poke_times`**;
  `plot_poke_duration_by_odor._extract_odor_poke_ms:5549` reads **`presentations`**.
- every viz extractor filters **`poke_ms > 0`**; no canonical metric does.

**5. Four near-identical poke-time extractors.** `extract_poke_times` (1756, nested),
`_extract_completed_position_poke_times` (4770), `extract_abort_poke_times` (1793, nested),
`_extract_aborted_position_poke_times` (4791) — the nested pair differ from the module-level
pair only by also returning the odor name. The canonical metrics parse the same JSON inline a
further three times. One tidy `position_poke_times`/`presentations` → long-format extractor
serves all of them (and pairs naturally with Phase 7b's `position_data` side-table).

**6. Decision accuracy is recomputed three times** while a fourth site reads it from the
metrics JSON: `plot_cumulative_rewards:3028`, `plot_decision_accuracy._decision_acc:5209`
and `plot_decision_accuracy_rolling_average` (windowed) all derive it from `trial_data`,
whereas `plot_behavior_metrics` and `plot_decision_accuracy_by_odor` correctly read
`metrics_*.json`. The two reading sites are the pattern to generalise.

---

## `visualization_utils.py` — 7,264 lines, 40 top-level functions

### No metric math — stays as is

| function | lines | verdict | note |
|---|---|---|---|
| `_clean_graph` | 48-84 | PLOT | strips labels for external editing |
| `load_tracking_with_behavior` | 86-158 | FETCH | tracking + behaviour join, `in_trial` labelling |
| `print_cache_keys` | 161-164 | PLUMBING | |
| `_extract_metric_value` | 168-195 | FETCH | dot-path lookup into the metrics dict — **the pattern 4a should generalise** |
| `_load_protocol_from_summary` | 197-208 | FETCH | |
| `_ensure_metrics_json` | 210-229 | FETCH | reads `metrics_*.json`, else `run_all_metrics` — **already the correct 4a shape** |
| `_series_line_widths` | 234-247 | PLOT | |
| `_build_odor_colors` | 762-779 | PLOT | colour scheme |
| `plot_decision_accuracy_by_odor` | 1225-1444 | FETCH + PLOT | reads `decision_accuracy_by_odor` + `global_choice_accuracy` from JSON. **Already correct.** |
| `_coerce_tz_naive` | 3411-3419 | PREP | |
| `_rolling_median_iqr` | 3549-3570 | PLOT (stat primitive) | rolling median + IQR; no metric knowledge → **Phase 5 primitive** |
| `_style_log_yaxis` | 3573-3594 | PLOT | |
| `plot_latency_over_time` | 3755-3792 | PLOT | thin wrapper |
| `plot_iti_over_time` | 3795-3824 | PLOT | thin wrapper |
| `_positions_in_presentations` | 4491-4509 | PREP | |
| `_extract_completed_position_poke_times` | 4770-4788 | PREP | dup — see finding 5 |
| `_extract_aborted_position_poke_times` | 4791-4812 | PREP | dup — see finding 5 |

### Derives non-metric facts

| function | lines | verdict | note |
|---|---|---|---|
| `_hr_odor_associations` | 705-759 | DERIVE | votes over HR-success trials to learn `{odor → 'A'|'B'}`. A protocol fact, not a metric. Used only for colours today. **Open question 1.** |
| `plot_choice_history` | 3827-4220 | DERIVE + PLOT | re-derives trial display categories (rewarded / unrewarded / timeout / aborted × HR) from `response_time_category` + `hidden_rule_success` + `fa_label`. No rate computed. Duplicates the *category rule* that Phase 6 consolidates — **leave to Phase 6, do not touch in 4a.** |
| `_plot_metric_over_sessions` | 3597-3752 | PLOT + DISPLAY-AGG | drives `_rolling_median_iqr`; `cumsum` builds the cumulative panel. Note it filters `value_col > 0`, silently dropping zero-valued trials. |
| `plot_cumulative_rewards_by_trial` | 3238-3408 | PLOT | `cumsum` of the rewarded flag along a trial axis — axis construction |

### Metric math — must move

| function | lines | verdict | what it computes | action |
|---|---|---|---|---|
| `plot_behavior_metrics` | 250-668 | FETCH + PLOT | reads any metric by dot-path; `groupby("session_num").mean()` for the group line | `DISPLAY-AGG` only — **no move** |
| `hidden_rule_and_false_alarm` | 782-1222 | MIXED | **per-odor FA rate** = FA-filtered aborts whose `last_odor_name` is *o*, ÷ (occurrences of *o* in completed trials' `odor_sequence` + those aborts). HR rate itself is read from JSON. | `NEW → fa_rate_by_odor(results, fa_types)`. Denominator matches **no** canonical: not `FA_odor_bias` (aborts@odor), not `odorx_abortion_rate` (presentations@odor). |
| `plot_decision_accuracy_rolling_average` `._build_plot_df` | 1537-1606 | MIXED | **rolling reward fraction** over trials. Numerator `response_time_category == "rewarded"` (optionally `& hidden_rule_success`); denominator = the whole window. Two windowing modes: `include_avg=True` back-fills the warm-up with the session mean — `(Σavailable + missing·overall_rate) / window`; `include_avg=False` emits full windows only. | `NEW → rolling_reward_fraction(...)`. **Not** `decision_accuracy`: with `completed_only=True` the denominator still contains `timeout_delayed`, so it is rewarded ÷ completed, matching no canonical metric. |
| `plot_sampling_times_analysis` | 1723-2111 | MIXED | poke duration **by position** and **by odor**, separately for completed and aborted trials, mean ± SD; plus per-session means of both | by-odor-completed = `VARIANT of avg_sampling_time_odor_x` (different source column + `>0` filter, finding 4); by-position and by-odor-aborted = `NEW`. Nested extractors dup finding 5. |
| `plot_abortion_and_fa_rates` | 2113-2596 | MIXED | (a) per-odor and per-position FA rate **re-parsed out of `fa_abortion_stats` strings**; (b) per-odor FA port ratio; (c) per-position abortion rate, string-parsed | (a) `DEDUP → fa_abortion_stats` once it returns numbers (finding 3); (b) `DEDUP → fa_port_ratio_by_odor` once its filter takes a set (finding 1); (c) `DEDUP → abortion_rate_positionX`. Cross-session mean±SEM at 2409-2530 is `DISPLAY-AGG`. |
| `plot_response_times_completed_vs_fa` | 2598-2756 | MIXED | mean `response_time_ms` over rewarded+unrewarded; mean `fa_latency_ms` over `FA_time_in` | `DEDUP → avg_response_time["Average Response Time (Rewarded + Unrewarded)"]` and `DEDUP → FA_avg_response_times["FA Time In"]`. Both are exact matches. |
| `plot_fa_ratio_a_over_sessions` | 2758-2893 | MIXED | per-odor **A/(A+B)** over `fa_port`, `FA_time_in` only | `VARIANT of fa_port_ratio_by_odor` — it returns (A−B)/(A+B); `A/(A+B) = (r+1)/2`. Rescale, do not recount. |
| `plot_cumulative_rewards` | 2899-3235 | MIXED | session `decision_accuracy` used as a threshold gate (`da > thresh`); cumulative reward count on a gap-collapsed time axis | `DEDUP → decision_accuracy` for the gate. The cumulative series and the gap collapsing are axis construction — stay. |
| `_load_subject_trial_timeline` | 3422-3546 | MIXED | **`iti_seconds`** = `sequence_start.shift(-1) − sequence_end`, within session, never across; `is_rewarded`; continuous `trial_index`; gap-collapsed `time_seconds` | ITI is `NEW → inter_trial_interval(results)` — nothing canonical is time-resolved and no canonical ITI exists at all. The axis construction stays. |
| `plot_position_completion_rate` | 4224-4488 | MIXED | per-position completion rate = reached-and-passed ÷ reached | `VARIANT of abortion_rate_positionX` (its complement) **but with "reached" definition B** — session-wide `max(num_odors)` for completed trials vs canonical's per-trial `position_poke_times`. See finding 2. Do not swap until the definition is settled. |
| `plot_false_alarm_rate_by_position` | 4512-4767 | MIXED | per-position FA rate = FA-aborts at *p* ÷ trials that reached *p* | `NEW → fa_rate_by_position(results, fa_types)`. Uses "reached" **definition C**. `FA_position_bias` is a normalised bias, not this; `fa_abortion_stats` by-position divides by total aborts, not by reached. |
| `plot_poke_duration_by_position` | 4815-5080 | MIXED | mean poke duration per position, split completed vs aborted | `NEW → poke_duration_by_position(results)`. Canonical has only the pooled scalars. |
| `plot_decision_accuracy` | 5083-5400 | MIXED | `_decision_acc(td, mask)` = rewarded/(rewarded+unrewarded) among a mask, evaluated for HR and non-HR trials separately; `_day_mean` across animals | `DEDUP → decision_accuracy` for the unmasked form (the docstring already says "as in the decision_accuracy metric") — canonical needs an optional trial mask. The HR / non-HR split is `NEW`: it is not `hidden_rule_performance`, which has a different numerator *and* denominator. `_day_mean` is `DISPLAY-AGG`. |
| `plot_poke_duration_by_odor` | 5403-5861 | MIXED | mean poke duration per odor per day, pooled into A/B / HR / OTHER series | `VARIANT of avg_sampling_time_odor_x` — reads `presentations`, filters `poke_ms > 0`, per-day. Both divergences of finding 4 apply. `_session_hr_odors` (5524) is FETCH. |
| `get_fa_ratio_a_stats` | 5867-5978 | **METRIC** (no plotting at all) | per-odor `A/(A+B)` across sessions, FA filter `startswith("FA_")` | `VARIANT of fa_port_ratio_by_odor`. **This one is a pure metric function sitting in `visualization/`** — it moves wholesale regardless of the dedup. |
| `plot_fa_ratio_by_hr_position` | 5981-6408 | MIXED | (A−B)/(A+B) over three HR slices (`count_ports` at 6145) | `VARIANT of fa_port_ratio_by_odor` ×3 — finding 1 |
| `plot_fa_ratio_by_abort_odor` | 6411-6848 | MIXED | (A−B)/(A+B) by abort odor, HR vs non-HR aborted sequences | `VARIANT of fa_port_ratio_by_odor` ×2 — finding 1 |
| `plot_hidden_rule_abort_poke_gap` | 6851-7126 | MIXED | **latency between the HR-position poke and the last poke of an aborted trial** (`last_end − hr_end`), plus a start→end variant | `NEW → hr_abort_poke_gap(results)`. No canonical latency of any kind between positions. Trailing mean±SEM is `DISPLAY-AGG`. |
| `plot_hr_reward_fraction_over_trials` | 7129-7264 | MIXED | rolling % of rewarded trials that are HR-rewarded, `rolling(window).mean() * 100` | `NEW → rolling_hr_reward_fraction(...)`. Related to `hidden_rule_performance` but different denominator (rewarded trials, not HR hits) and time-resolved. |

**Count for this file:** 17 stay, 4 derive/axis, **19 carry metric math** — of which 6 are
exact `DEDUP`, 8 are `VARIANT`, 8 are `NEW` (several functions carry more than one).

---

## `movement_analysis_utils.py` — 4,460 lines, 12 top-level functions

This file is structurally different from `visualization_utils.py`. Its metric math is not
scattered through the plotters — it is concentrated in **one 591-line pure-compute function**
that already writes a derivative artifact, and the plotters mostly read that artifact back.
The problem here is not "metrics inside plots" but "a metrics module filed under
`visualization/`".

### `compute_speed_analysis` is a metric module in the wrong package

`compute_speed_analysis` (1867-2457) contains **no plotting at all**. It loads SLEAP tracking,
aligns each trial to its last cue-port poke-out, and derives a whole family of movement
metrics, then writes `speed_analysis.parquet` per session:

| quantity | how |
|---|---|
| binned speed epoch | `np.gradient(x, t)`, `np.gradient(y, t)`, `hypot` → `pd.cut` into `bin_ms` bins, `mean` or `max` per bin |
| baseline μ, σ | pooled speed over the `[-0.15, -0.05] s` window across all trials in the session |
| speed threshold `vthresh` | `max(alpha·μ, μ + beta·σ)` |
| movement-onset latency | first bin after 0 crossing `vthresh`, then refined to the sample and **linearly interpolated** between the bracketing samples |
| `movement_onset_from_valve_s` | onset time re-referenced to the last valve start |
| `path_length_px` | `Σ hypot(diff(x), diff(y))` over the trial segment |
| `travel_time_s` | `t_end − t_zero` |
| `tortuosity` | path length ÷ straight-line distance, over the bin-aligned window |

**None of these exist in `metric_analysis`.** This is the single largest move in 4a:
`compute_speed_analysis` and its batch driver `run_speed_analysis_batch` (1784-1864) move
wholesale into `metric_analysis` (a new `movement.py`, or `metric_analysis/movement/`), along
with the nested primitives `_speed_by_bins`, `_speed_series`, `_compute_tortuosity`,
`_path_length` and the module-level `_binned_speed`.

It also means `metric_analysis` acquires a second saved artifact besides
`metrics_*.json`. Worth confirming that `speed_analysis.parquet` keeps its current path and
filename through the move — **open question 2**.

### Findings specific to this file

**7. The baseline/threshold block is computed three times.** `μ`, `σ` over `[-0.15, -0.05] s`
and `vthresh = max(αμ, μ+βσ)` appear identically in `compute_speed_analysis:2318-2330`,
`plot_epoch_speeds_by_condition:2556-2569` and `plot_traces_with_speed_threshold:3110-3114`.
The latter two already read `speed_analysis.parquet` — they should read the threshold from it
rather than re-deriving it, which also removes the risk of the plotted threshold disagreeing
with the one used to compute the saved latencies.

**8. Speed-from-tracking is implemented three times.** `_binned_speed` (57-93, module level),
`_speed_by_bins` (2047, nested — a near-verbatim copy that takes `edges` as an argument
instead of deriving them) and `_speed_series` (2079, the same derivation without binning).
One `speed_series(tracking, t0, t1)` primitive plus an optional binner replaces all three.

**9. `_kw_mwu_by_group` is generic statistics, not a metric.** It takes
`(df, value_col, group_col)` and runs Kruskal-Wallis, then pairwise Mann-Whitney U with Holm
correction. By the plan's own 0.2 test it knows only the *format* of its input, not what the
data *is* — which makes it `hypnose_helpers`-shaped rather than `metric_analysis`-shaped.
A `stats.py` is the right destination either way; **open question 3** is which repo.

**10. Trajectory helpers duplicate across files.** `_resample_trace` (arc-length resampling
onto a normalised grid) is written twice — `plot_trial_traces_by_mode:1083` and
`movement_analysis/sing_rew_movement.py:228`. `_smooth_tracking`, `_infer_port`,
`_last_poke_out` and `_extract_segment` likewise each appear 2-4 times across the four big
plotters in this file. All `PREP`; all belong in the shared `visualization/` prep module.

### Function table

| function | lines | verdict | what it computes | action |
|---|---|---|---|---|
| `_binned_speed` | 57-93 | METRIC | speed from tracking gradient, binned mean/max | `NEW →` move with `compute_speed_analysis`; dedup against `_speed_by_bins` (finding 8) |
| `_load_tracking_and_behavior` | 96-180 | FETCH | | → `visualization/io/` |
| `plot_movement_trace` | 183-296 | PLOT | rolling-mean smoothing only | stays |
| `plot_movement_by_trial_state` | 300-423 | PLOT | | stays |
| `plot_movement_with_behavior` | 426-854 | PLOT + DERIVE | `_last_odor_series` infers last-odor identity; `_plot_segments_by_mask` splits contiguous runs | stays (segment splitting is a drawing technique) |
| `plot_trial_traces_by_mode` | 857-1781 | PLOT + DISPLAY-AGG | mean trajectory ± SEM band across arc-length-resampled traces, with a normal-direction band | `DISPLAY-AGG` — stays. `_resample_trace` → shared prep (finding 10) |
| `run_speed_analysis_batch` | 1784-1864 | METRIC (driver) | batch loop over subjects/dates | moves with `compute_speed_analysis` → `metric_analysis/run.py` shape |
| **`compute_speed_analysis`** | **1867-2457** | **METRIC** | the whole movement-metric family above; writes `speed_analysis.parquet` | **`NEW →` `metric_analysis/movement.py`, wholesale. No plotting to leave behind.** |
| `plot_epoch_speeds_by_condition` | 2460-2662 | FETCH + PLOT | reads the parquet, then **re-derives** baseline μ/σ and `vthresh`; averages traces across trials | `DEDUP →` read the threshold from the parquet (finding 7). Trace averaging is `DISPLAY-AGG` |
| `plot_traces_with_speed_threshold` | 2664-3272 | FETCH + PLOT | reads `speed_threshold_time` from the parquet, but **re-derives** μ/σ/`vthresh` on the fallback path | `DEDUP →` same as above (finding 7) |
| `plot_tortuosity_lines_overlay` | 3275-3554 | FETCH + PLOT | reads tortuosity/timing from the parquet; draws data-derived and fixed reference lines | stays — **already the correct 4a shape** |
| `plot_movement_analysis_statistics` | 3557-4460 | FETCH + PLOT + STATS | reads all five per-trial movement metrics from the parquet; `_kw_mwu_by_group` (KW + MWU + Holm); `polyfit` trend lines; `groupby(condition, date).agg(mean, sem)`; min-max normalisation of session means | metrics already fetched — **correct shape**. `_kw_mwu_by_group` → `stats.py` (finding 9). Everything else is `DISPLAY-AGG` and stays |

**Count for this file:** 5 stay unchanged, 3 are already correct fetch-and-plot, **2 carry
metric math to move** (`compute_speed_analysis` + `_binned_speed`, with
`run_speed_analysis_batch` following them), 2 need a `DEDUP` of the threshold block, and 1
generic stats helper needs a home.

---

## `pred_seq_utils.py` — 1,886 lines, 9 public + ~25 private functions

Predicted-sequence analysis. Structurally the cleanest file in `visualization/`: about two
thirds of it is small `PREP` and `PLOT` helpers, and the public functions are boxplot/summary
plotters. But the per-trial *quantities* those boxplots show are computed here from raw
timestamps, and **almost none of them exist in `metric_analysis`** — this file is where the
trial-level latency and duration family lives.

### Finding 11 — three timing metrics silently disagree with their canonical namesakes

Each of these is computed from a **different pair of events** than the identically-named
canonical column. None is a `DEDUP`; all three are `NEW`.

| here | definition here | canonical | canonical definition |
|---|---|---|---|
| `response_time:1041` | `first_supply_time − poke_odor_end` | `trial_data.response_time_ms` | `first_reward_poke − last_poke_out_time` (`classification_utils:2441`) |
| `fa_analysis:1221` | `fa_time − poke_odor_end` | `trial_data.fa_latency_ms` | `fa_time − abortion_time` (`classification_utils:3039`) |
| `performance:1737` | rewarded/(rewarded+unrewarded) **per odor sequence** | `decision_accuracy` | same formula, whole session |

The first two measure **reward delivery** and **false alarm** relative to leaving the odor
port; the canonical columns measure them relative to the reward-port poke and to the abortion
timestamp. A function called `response_time` in `visualization/` therefore does not plot
`response_time_ms`. Whatever is decided, these must be *named apart* when they move.

### Finding 12 — two rolling accuracies with different denominators

`performance` (rolling form, `_rolling_pts:1628`) and
`visualization_utils.plot_decision_accuracy_rolling_average` both plot "rolling accuracy",
and they are **not the same metric**:

- `performance` drops `timeout_delayed` before windowing → denominator is
  rewarded+unrewarded → it genuinely *is* `over_windows(decision_accuracy, …)`.
- `plot_decision_accuracy_rolling_average` keeps every completed trial in the window →
  denominator includes timeouts, and it back-fills the warm-up with the session mean.

This is the concrete case for the resolver design above: the first collapses into
`over_windows(decision_accuracy, …)` for free; the second must stay a separately named
metric or its curve changes.

### Finding 13 — an outlier rule lives in the plotters

`response_time:1052` and `fa_analysis:1249` both drop values above `10 × group mean`
(`thresholds = {g: 10.0 * mean(vs)}`). That is a data-cleaning policy, it changes the plotted
numbers, it is written twice, and it exists nowhere in `metric_analysis`. It must travel with
the metrics rather than being left behind in the plot code — **open question 4**.

### Finding 14 — context-dependent odor labelling

`poke_time_all_pos:936` and `fa_analysis:1198` both relabel `OdorG` by the odor that preceded
it (`C → G(C)`, `F → G(F)`), because G means something different in each sequence. This is a
`DERIVE` on the odor sequence, duplicated in two places, and it silently changes the grouping
key of any per-odor metric computed here relative to one computed in `metric_analysis`.

### Function table

Stays as is — `PREP` and `PLOT` only:

| functions | lines | verdict |
|---|---|---|
| `_parse_json_value`, `_normalize_date`, `_sequence_label`, `_sequence_len_ok`, `_normalize_odor_name`, `_last_position_entry`, `_extract_position_entry`, `_ordered_position_entries` | 21-150 | PREP — JSON/label parsing. Shared with `sing_rew.py` and `sing_rew_movement.py`, so → the shared prep module |
| `_collect_sessions`, `_load_trial_data`, `_load_sorted_session` | 42-64, 359-372 | FETCH → `visualization/io/` |
| `_count_to_marker_size`, `_nice_round`, `_add_size_legend` | 153-236 | PLOT — marker sizing/legend |
| `_order_sequence_labels`, `_order_odor_labels`, `_darken`, `_resolve_color`, `_canonical_odor`, `_build_odor_filter`, `_ordered_groups` | 280-356 | PREP — ordering and colour |
| `_plot_violins_with_stats` | 239-278 | PLOT + DISPLAY-AGG (mean/std per violin) |
| `_plot_summary_daily`, `_plot_summary_rolling`, `_plot_summary`, `_is_multi_session`, `_apply_shared_ylim`, `_summary_save_suffix` | 375-589 | PLOT + DISPLAY-AGG — generic per-session mean and rolling mean of *whatever* values are passed in; no metric knowledge → **Phase 5 primitives** |
| `_plot_performance_daily`, `_plot_performance_rolling` | 1544-1708 | PLOT + DISPLAY-AGG — but `_rolling_pts:1628` carries the windowing rule (finding 12) |

Carries metric math:

| function | lines | verdict | what it computes | action |
|---|---|---|---|---|
| `last_odor_poke_time` | 592-689 | PREP + PLOT | selects `poke_time_ms` of the last position | **selection, not computation** — stays once the tidy poke table exists (finding 5) |
| `trial_poke_duration` | 692-793 | MIXED | `last.poke_odor_end − first.poke_odor_start` — the **wall-clock span** of the odor-sampling phase | `NEW → trial_poke_span(...)`. Distinct from `cummulative_poke_time` (span ≠ sum) |
| `first_odor_poke_duration` | 796-882 | PREP + PLOT | selects `poke_time_ms` at position 1 | selection — stays |
| `poke_time_all_pos` | 885-989 | PREP + PLOT + DERIVE | pools `poke_time_ms` across positions by odor; relabels `OdorG` by context (finding 14) | selection + the `DERIVE` of finding 14 |
| `response_time` | 992-1125 | MIXED | `first_supply_time − poke_odor_end`, per sequence; drops values > 10× group mean | `NEW → reward_delivery_latency(...)` — **not** `response_time_ms` (finding 11). Outlier rule travels with it (finding 13) |
| `fa_analysis` | 1128-1436 | MIXED | `fa_time − poke_odor_end` per last odor and FA port; A/B counts per odor; mean ± SEM | `NEW → fa_latency_from_pokeout(...)` (finding 11). The A/B counts at 1311 are the **9th** FA-port counting site (finding 1). Outlier rule as above |
| `valve_to_reward` | 1439-1541 | MIXED | `first_supply_time − valve_start` of the last position | `NEW → valve_to_reward_latency(...)`. No canonical valve-timing metric exists |
| `performance` | 1711-1779 | MIXED | rewarded/(rewarded+unrewarded) per odor sequence, session and rolling | `VARIANT of decision_accuracy` — collapses to `by_group(…, "sequence")` / `over_windows(…)` cleanly (finding 12) |
| `cummulative_poke_time` | 1782-1886 | MIXED | `Σ poke_time_ms` across positions, per trial | `NEW → trial_poke_total(...)`. Related to `avg_sampling_time_completed_sequence` but per-trial, not a session mean |

**Count for this file:** ~25 helpers stay (mostly to the shared prep/primitives modules),
**7 carry metric math** — 6 `NEW`, 1 `VARIANT`, 0 exact `DEDUP`.

---

## `sing_rew.py` — 1,291 lines

**The closest file to the 4a target already.** It imports `compute_sing_rew_metrics`,
`compute_sing_rew_rates` and `_classify_trial` from `metric_analysis.sing_rew_metrics` and
drives them per session (`_singrew_session_records:604`), so the eight headline plotters
(`dprime`, `hit_fa_rate`, `criterion`, `hit_earned_reward`, `early_rejection_anticipatory`,
`efficient_early_rejection`, `premature_omission_rates`, `correct_rejection_rate`) are
6-12 line wrappers that fetch and plot. **This is the shape every other file should end up
in.** Only three things carry metric math.

| function | lines | verdict | what it computes | action |
|---|---|---|---|---|
| `_metric_value` | 580-601 | MIXED | two rates the canonical `compute_sing_rew_rates` does **not** return, derived here from `counts`: `ambiguous_rate = n_amb / n_tot` and `correct_rejection_rate = correct_rejection / n_nogo`. The docstring says "(not stored)" | `NEW →` add both to `compute_sing_rew_rates`. Both are one line from counts already in the dict — **the cheapest, lowest-risk win in the whole of 4a** |
| `FR_ratio` | 167-304 | METRIC (no plotting; delegates) | false-response trials ÷ completed trials per session, filterable by `fr_label` via `_fr_mask` | `NEW → false_response_ratio(trials, fr_types)`. **Not** the canonical `fa_rate`, which is `false_alarm / n_nogo` off a different column (`fa_label`, not `fr_label`) |
| `_response_time_ms` | 72-87 | MIXED | `first_supply_time − poke_odor_end` — its own docstring says "computed as in `pred_seq_utils`" | duplicate of `pred_seq_utils.response_time:1041` — **the same `NEW` metric written twice** (finding 11) |

Everything else stays: `_normalize_fr_types`, `_port_label`, `_fr_mask`, `_trim_leading_empty`,
`_subject_color_map`, `_pretty_metric`, `_isnan`, `_size_legend_handles` are `PREP`;
`_session_fr_latencies:392` correctly reads the stored `fr_latency_ms` column;
`_plot_fr_ratio_daily`, `_plot_latency_box_AB`, `_plot_sing_rew_metrics`,
`_plot_outcome_composition` are `PLOT`; `_partition_total` and the `cumsum` in
`_plot_cumulative_hit_cr:1186` are `DISPLAY-AGG`; `_trim_timeline_to_singrew:1121` is `PREP`.

---

## `valve_poke_plots.py` — 646 lines, 1 public function

`plot_valve_and_poke_events` (27-646) is a single 620-line raw-data debugging plot: it reads
harp registers straight off disk and draws valve/poke event traces on a time axis. **No metric
math anywhere in the file** — the arithmetic the AST flagged is all timestamp offsetting,
window slicing and tick formatting.

**Finding 15 — it re-implements a loader, not a metric.** `_compute_real_time_offset`
(219-258) opens with *"Compute the same real_time_offset used by `load_all_streams`"* and then
does exactly that: parse the `YYYY-MM-DDTHH-MM-SS` folder timestamp, read the heartbeat
register, take the difference. Also `_load_register_files` (189-217) and `_safe_concat`,
`_apply_offset_and_localize`, `_slice` are loader plumbing.

This is out of 4a's literal remit (it is not metric math) but squarely inside its *goal*:
`visualization/` should fetch, not re-implement fetching. The offset computation belongs in
`io/loaders.py` next to `load_all_streams`, and drift between the two would silently shift
every timestamp on this plot relative to every other figure in the package. **Recommend
folding it in during 4a** while the file is open; it is a small, self-contained move.

| function | lines | verdict | action |
|---|---|---|---|
| `plot_valve_and_poke_events` | 27-646 | PLOT + FETCH | stays; extract the loader helpers |
| `_compute_real_time_offset` | 219-258 | FETCH (duplicated) | `DEDUP →` `io/loaders.load_all_streams`'s offset logic (finding 15) |
| `_load_register_files`, `_try_load`, `_safe_concat`, `_apply_offset_and_localize`, `_slice`, `parse_exp_ts_to_uk`, `_parse_hhmmss` | 73-307 | FETCH / PREP | → `visualization/io/` or `io/loaders.py` |
| `restrict`, `extend_to_window_end`, `__call__`, `update_ticks` | 415-621 | PLOT | stays |

---

## `modelling/switchpoint/plots.py` — 638 lines

One of the two files the plan's original count missed. **It needs nothing in 4a.**

Every function consumes an already-fitted model artifact produced by
`hypnose_behavior/modelling/switchpoint/` (`prep`, `fit`, `comparison`, `qlearning_fits`,
`qlearning_bands`, `sweep`, `acf`, `null_means`) and draws it. The model math correctly lives
in the `modelling/` package, and this file imports `logistic_p` and
`qlearning_generative_band` from there rather than re-deriving them — **the separation 4a is
trying to achieve elsewhere is already in place here.**

The only arithmetic is drawing math: FWHM bracket bounds for the posterior
(`plot_posterior:157`), a curve offset that keeps two overlapping lines legible
(`plot_model_comparison:253`), jitter, and the AIC/BIC table rows read from `comparison`.

| function | lines | verdict |
|---|---|---|
| `_rolling_mean` | 95-100 | PLOT (stat primitive) — centred moving average via `convolve`. **The third rolling-mean implementation** in `visualization/`, after `_rolling_median_iqr` (visualization_utils:3549) and `_rolling_pts` / `_plot_summary_rolling` (pred_seq_utils). → Phase 5 primitive |
| `_mark_sessions`, `plot_strategy`, `plot_posterior`, `_qlearn_label`, `_overlay_qlearning`, `plot_model_comparison`, `plot_qlearning_generative`, `plot_multistart`, `plot_qlearning_sweep`, `_plot_acf_panel`, `plot_residual_autocorr`, `_box_with_points`, `_null_distribution`, `plot_permutation` | 103-638 | PLOT — all of them |

---

## `movement_analysis/sing_rew_movement.py` — 433 lines

The other file the plan's original count missed. **No metric math**, but it is the worst
single case of `PREP` duplication.

It uses the canonical `_classify_trial` from `sing_rew_metrics` to sort trials into
Hit/Miss/FA/CR and draws one trace figure per (sub)category. The derivations it does —
where a trial's trace starts and ends — are `DERIVE`, not metrics.

| function | lines | verdict | note |
|---|---|---|---|
| `_port_letter`, `_odor_letter`, `_trial_port_group`, `_naive_dt` | 76-141 | PREP | `_odor_letter` is a fourth copy of the odor-token normaliser (also at `visualization_utils:874`, `:5511`, `pred_seq_utils._canonical_odor:333`) |
| `_last_poke_out` | 144-177 | DERIVE | final cue-port exit time — **third copy**, after `movement_analysis_utils:1059` and `:2829` |
| `_segment_end` | 180-202 | DERIVE | per-category trace endpoint (reward poke for Hit/FA, next initiation for Miss/CR) |
| `_smooth_tracking` | 205-212 | PREP | duplicate of `movement_analysis_utils:1035` / `:2928` |
| `_extract_segment` | 215-225 | PREP | duplicate of `movement_analysis_utils:1048` |
| `_resample_trace` | 228-241 | PREP | duplicate of `movement_analysis_utils:1083` (finding 10) |
| `_plot_category` | 244-305 | PLOT + DISPLAY-AGG | `nanmean` average trace across resampled traces |
| `plot_category_traces` | 308-433 | PLOT | |

Every `PREP`/`DERIVE` row here has a twin in `movement_analysis_utils.py`. They all go to the
shared `visualization/` prep module together — this file and `movement_analysis_utils.py`
should be de-duplicated in one pass, not two.

---

## Consolidated checklist — every metric that must exist afterwards

The hard constraint is **lose no metric**. These are the quantities computed in
`visualization/` today that have **no canonical version at all**. If a name below has no
home in `metric_analysis` when 4a finishes, a metric was dropped — and the QC regression
will not tell you, because it never sees a plot.

**Behavioural (from `visualization_utils.py`)**

| # | metric | current site | definition |
|---|---|---|---|
| 1 | FA rate by odor | `hidden_rule_and_false_alarm:1003` | FA-filtered aborts at odor ÷ (odor occurrences in completed `odor_sequence` + those aborts) |
| 2 | rolling reward fraction | `plot_decision_accuracy_rolling_average:1573` | rewarded ÷ window; warm-up back-filled with the session mean; optional HR-only numerator |
| 3 | poke duration by position | `plot_sampling_times_analysis`, `plot_poke_duration_by_position` | mean ± SD of `poke_time_ms` per position, split completed/aborted |
| 4 | poke duration by odor, aborted trials | `plot_sampling_times_analysis:1950` | canonical pools all aborted trials into one scalar; per-odor does not exist |
| 5 | FA rate by position | `plot_false_alarm_rate_by_position:4655` | FA aborts at *p* ÷ trials reaching *p* ("reached" definition C) |
| 6 | inter-trial interval | `_load_subject_trial_timeline:3459` | `sequence_start.shift(-1) − sequence_end`, within session only |
| 7 | decision accuracy, HR vs non-HR | `plot_decision_accuracy:5209` | `decision_accuracy` restricted to hidden-rule / non-hidden-rule trials |
| 8 | HR abort poke gap | `plot_hidden_rule_abort_poke_gap:7013` | last poke end − HR-position poke end, on aborts that hit the hidden rule |
| 9 | rolling HR reward fraction | `plot_hr_reward_fraction_over_trials:7229` | rolling % of rewarded trials that are HR-rewarded |

**Movement (from `movement_analysis_utils.compute_speed_analysis`)** — all seven, none canonical

| # | metric | definition |
|---|---|---|
| 10 | binned speed epoch | gradient speed, binned mean/max, aligned to last poke-out |
| 11 | baseline μ/σ and `vthresh` | `[-0.15,-0.05] s` pooled baseline; `max(αμ, μ+βσ)` |
| 12 | movement-onset latency | first supra-threshold crossing, linearly interpolated between samples |
| 13 | `movement_onset_from_valve_s` | onset re-referenced to the last valve start |
| 14 | `path_length_px` | `Σ hypot(diff(x), diff(y))` |
| 15 | `travel_time_s` | `t_end − t_zero` |
| 16 | `tortuosity` | path length ÷ straight-line distance |

**Trial timing (from `pred_seq_utils.py` / `sing_rew.py`)**

| # | metric | current site | definition |
|---|---|---|---|
| 17 | trial poke span | `trial_poke_duration:746` | last `poke_odor_end` − first `poke_odor_start` |
| 18 | reward-delivery latency | `pred_seq_utils:1041` **and** `sing_rew.py:87` | `first_supply_time − poke_odor_end` — **not** `response_time_ms` |
| 19 | FA latency from poke-out | `fa_analysis:1221` | `fa_time − poke_odor_end` — **not** `fa_latency_ms` |
| 20 | valve-to-reward latency | `valve_to_reward:1493` | `first_supply_time − valve_start` of the last position |
| 21 | trial poke total | `cummulative_poke_time:1835` | `Σ poke_time_ms` across positions, per trial |

**Single-reward (from `sing_rew.py`)**

| # | metric | current site | definition |
|---|---|---|---|
| 22 | false-response ratio | `FR_ratio:281` | FR trials ÷ completed trials, filterable by `fr_label` |
| 23 | `ambiguous_rate` | `_metric_value:591` | `n_amb / n_tot` — add to `compute_sing_rew_rates` |
| 24 | `correct_rejection_rate` | `_metric_value:595` | `correct_rejection / n_nogo` — add to `compute_sing_rew_rates` |

Plus the definitional questions that must be **settled, not silently picked**: which
"reached at position *p*" (finding 2), and whether the 10× outlier rule is part of the
metrics or part of the plots (finding 13).

---

## D0 resolution — the metric signature *(settled 2026-08-05)*

**Decision: every metric takes a frame and returns a value, with a thin `f(results)` wrapper
retained. Done as part of 4a, not deferred to 4b.**

```python
def decision_accuracy(trials) -> float:          # pure core: no I/O, no printing
    ...
def decision_accuracy_session(results):          # wrapper: run_all_metrics keeps working
    return decision_accuracy(results["trial_data"])
```

**Unchanged by this, deliberately:** `run_all_metrics`, the metrics `*.json`, every saved
value, and therefore the regression fingerprint. The change is additive — it exposes the
core, it does not alter what the session-level call returns. 4a stays GREEN.

**Why in 4a rather than 4b:** doing it after the moves means editing all 24 incoming metrics
twice, and the second pass has no plot-level guard — the regression fingerprints `trial_data`
and the metrics dict, never a figure. It is also the precondition for 4b's registry: a
registry over uniform `f(frame) -> value` functions works; one over functions that each take
a `results` dict and print to stdout does not.

### The signature is not uniform — four tiers

| tier | n | metrics | shape |
|---|---|---|---|
| **1 — trial-reducible** | 13 | `decision_accuracy`, `global_choice_accuracy`, `decision_accuracy_by_odor`, `premature_response_rate`, `response_contingent_FA_rate`, `global_FA_rate`, `sequence_completion_rate`, `hidden_rule_performance`, `hidden_rule_detection_rate`, `choice_timeout_rate`, `avg_response_time`, `FA_avg_response_times`, `response_rate` | `f(trials)`; every resolver works |
| **2 — grouping key inside a JSON blob** | 8 | `odorx_abortion_rate`, `hidden_rule_counts_by_odor`, `avg_sampling_time_odor_x`, `avg_sampling_time_completed_sequence`, `avg_sampling_time_aborted_sequence`, `abortion_rate_positionX`, `manual_vs_auto_stop_preference`, `fa_abortion_stats` | `f(trials)` + `f(position_data)` for per-position/odor grouping |
| **3 — normalised by a whole-frame quantity** | 3 | `FA_odor_bias`, `FA_position_bias`, `odor_initiation_bias` | `f(trials, *, reference=None)` |
| **4 — reads tables other than `trial_data`** | 4 | `avg_sampling_time_initiation_abortion`, `non_initiated_FA_rate`, `non_initiation_odor_bias`, `fa_port_ratio_by_odor` | being removed — see below |

### Tier 1 — store contributions, never a per-trial "value"

**The rule that is easy to get wrong.** A rate metric is not a per-trial quantity. Store the
**numerator and denominator contributions separately** and let any window or group reduce
them:

```python
num = (rtc == "rewarded")                       # contributes to the numerator
den = rtc.isin(["rewarded", "unrewarded"])      # contributes to the denominator
value_over(sl) = num[sl].sum() / den[sl].sum()
```

Storing one number per trial and taking a rolling mean gives `rewarded / window_size` — a
denominator silently containing timeouts and aborts. **That is finding 12**: it is exactly
why `pred_seq.performance` and `plot_decision_accuracy_rolling_average` disagree today. Mean-type
metrics store `(value, included)` and reduce to `sum/count`.

Two cumulative sums make any window O(1), so this is also the efficient form.

### Tier 2 — derive `position_data` at load time

The value is computable per trial today; what is impossible is `by_group(..., "position")`,
because one trial row holds many positions inside a dict and there is no column to group on.

**`position_data` is built by the loader, not written by the classifier alone.**
`load_session_results` emits the long table either by reading the flat one (new sessions) or
by expanding the blobs (legacy sessions), so metrics only ever see one shape and carry **no
backward-compatibility branch**. Skipping legacy sessions instead would disable every
per-position metric on every session already analysed — which is all of them.

This costs nothing extra: it is the same parsing finding 5 already flags as duplicated four
times in `visualization/` plus three more inline in `metrics_utils`. Writing it once in `io`
is work the audit already requires; making it the loader's job means the metrics never see a
blob. Pairs directly with Phase 7b's `position_data` side-table.

### Tier 3 — an optional `reference`, and the plotting option it buys

`FA_odor_bias` is a ratio to the animal's own overall rate:
`bias[odor] = (n_fa@odor / n_ab@odor) / (total_fa / total_ab)`. The divisor is computed over
whatever frame it is handed, so on a rolling call the baseline silently becomes *that
window's* rate. Both readings are legitimate, so make it explicit:

```python
def FA_odor_bias(trials, *, reference=None):
    ref = reference if reference is not None else _global_rate(trials)
```

| call | result |
|---|---|
| session, no `reference` | computed from the session — **identical to today** |
| rolling, no `reference` | each window normalises by itself (local baseline) |
| rolling, `reference=session_rate` | fixed session baseline |

**This gives the plotters a `baseline="session" | "window"` option for free.** The plotter
computes the session reference once (a fetch), or passes nothing — it never touches the
formula, so no metric math returns to `visualization/`.

### Tier 4 — removed, not ported *(output change — needs fixture regeneration)*

Decision 2026-08-05: **drop non-initiated trials from the metric set.** Integrating
non-initiated trials into `trial_data` properly is its own piece of work and is out of scope
for this restructure — noted, not scheduled.

Consequences, all deliberate:

- `metrics['non_initiated_FA_rate']` — removed
- `metrics['fa_port_ratio_by_odor']['with_non_initiated']` — removed; `run_all_metrics`
  currently stores both variants, and only `without_non_initiated` survives. **This makes
  `fa_port_ratio_by_odor` a clean tier-1 metric on `trial_data`** — which matters, because it
  is the canonical target for finding 1 (the FA port ratio is written 8 times).
- `avg_sampling_time_initiation_abortion`, `non_initiation_odor_bias` — removed
- `plot_abortion_and_fa_rates` loses its "Non-Initiated" position category;
  `plot_fa_ratio_a_over_sessions` and `get_fa_ratio_a_stats` lose `include_noninitiated`
- `load_session_results` can stop loading the three `non_initiated_*` tables

**This is the only part of 4a that changes saved metric values.** It takes a deliberate
`--generate` in the same commit, with the metric-key diff confirming only those keys left —
and confirmation before running it.

---

## Q5 resolution — "reached at position *p*" *(settled 2026-08-05, evidence-led)*

### The decision

**Two helpers, not one. "How far the sequence got" is never filtered; "was this position
sampled" is filterable. Default is unfiltered everywhere.**

| helper | returns | filterable | consumers |
|---|---|---|---|
| `sequence_depth(trial)` / `presented_positions(trial)` | contiguous `1..max(presented position)`, source = `presentations` | **no** | every per-position *denominator*: `abortion_rate_positionX`, `fa_abortion_stats`, position completion rate, FA-rate-by-position |
| `sampled_positions(trial, only_true_pokes=False)` | the positions actually recorded; may be gappy | **yes** | every *sampling* metric: poke duration by position/odor, `avg_sampling_time_*` |

`only_true_pokes=True` keeps only entries with `poke_source == "poke"`.

**Backward compatibility (required).** `poke_source` does not exist yet and will never exist
on already-saved sessions. When the field is **absent**, the `only_true_pokes` variants are
**not populated** — the metric key is omitted rather than silently returning the unfiltered
value. Treating "no marker" as "everything is a real poke" would make old and new sessions
look comparable when they are not.

### Why these two must be separate — structural, not a matter of volume

A single `reached_positions(trial, only_true_pokes=True)` returning a filtered set produces
**physically impossible sets**. Filtering a non-`poke` entry out of the middle of a trial
credits it with reaching position 5 but not position 3 — which cannot happen, and makes any
per-position denominator built from it non-monotonic. Illustration of the shape:

```
sub-048 gid=13   non-"poke" entry at position 3 of max 5   keys=[1,2,3,4,5]
                 -> filtering leaves [1, 2, 4, 5]
```

A gap is meaningless for *reached* and perfectly natural for *sampled* ("no sample at
position 3 on this trial"). This is the same distinction as `ses` vs `session_index` in
Phase 2b — selection versus positioning — and it fails the same silent way if merged.

The argument does not depend on how many such entries exist: it only requires that interior
ones occur at all, which they demonstrably do. Both grace entries and 0 ms positions can sit
mid-sequence.

**On prevalence — do not use the audit scan as an estimate.** A scan keyed on
`poke_first_in == poke_odor_start` and `poke_time_ms < PRE_ODOR_GRACE_MS` returns 112
last-position and 144 interior entries across the fixtures, but that is an **upper bound
that mostly counts genuine short pokes** — animals really do poke for under 20 ms. The real
figure comes from a direct measurement (re-running with the grace period set to 0):
**roughly 2-10 odors per session, varying by animal.** Not negligible, but nowhere near the
scan's numbers. The scan establishes that interior cases exist; it does not measure them.

### Why the default is unfiltered

A poke ending ~10 ms before the valve opens, or an animal dwelling at the cue port, has
plausibly still received the odor information. Discarding those by default is the stronger
assumption, not the safer one. Keeping them by default and offering `only_true_pokes` as an
opt-in leaves the judgement with the analyst.

At ~2-10 odors per session the filter refines the sampling numbers rather than transforming
them — worth having and worth being able to check, but not a headline correction.

### Why contiguous fill, and why it must not become plain membership

`presentations` is the source (the richest table), but the set is `1..max`, **not** bare
membership. Until the Phase 6/7 fix lands, the fill is doing real work:

> sub-057 gid=108 — `position_poke_times` keys `[2, 3]`, `num_odors=2`. Position 1 *was*
> presented; its 0 ms poke was never written, and `num_odors` dropped it too. `1..max` gives
> `{1,2,3}` — the truth. Plain membership gives `{2,3}` and loses a real position.

So today the fill recovers dropped positions, and plain membership would silently
under-count. Once 0 ms positions are written, `1..max` and membership coincide and the
distinction disappears — but the fill is the safe form either way.

### The three discrepancy patterns, with trial IDs

Measured per trial on all 9 fixture sessions (parquet, pyarrow 23.0.1 — the earlier CSV-fallback
run gave identical session totals, so the loader path does not affect any of this).
**15 trials disagree, in 3 of 9 sessions.** Clean: sub-061, sub-056, sub-046, sub-053, sub-040 ×2.

| pattern | session | `global_trial_id` | cause |
|---|---|---|---|
| 1 — grace entry on an aborted trial | sub-057 | 209, 214, 235, 327 | `presentations`/`position_poke_times` carry the grace position; `last_odor_position` does not |
| | sub-059 | 48, 203 | |
| | sub-048 | 122, 145, 162 | |
| 2 — 0 ms poke never written | sub-057 | 108, 124, 197, 226, 277 | position missing from `position_poke_times`, `presentations` **and** `num_odors` |
| 3 — null `last_odor_position` | sub-057 | 332 | aborted trial contributes nothing under the `last_odor_position` walk |

### The mechanism (confirmed in the code, not inferred)

`classification_utils.py`:

- `PRE_ODOR_GRACE_MS = 25.0` — `:47`
- `_grace_overlap_ms(last_poke_end, window_start, window_end)` — `:79`. Fires **only** when
  `last_poke_end <= window_start`, i.e. the poke ended *before* the odor window opened.
- `:1281-1293` — when no poke occurs during the odor window at all, a synthetic
  `position_poke_times[position]` entry is written with `poke_time_ms = grace_ms`,
  `poke_odor_start = poke_first_in = odor_start`. Those two timestamps are the *valve*
  opening, assigned by construction — that is the tell, and it is not decisive on its own.
- `:2986` — `last_odor_pos = presentations_valid[last_idx]["position"]`, taken from the
  abort-detection logic's chosen last event, which the grace path never updates.

Hence a position can appear in the poke/presentation record while `last_odor_position`
correctly reports an earlier one. Worked example: sub-048 gid 122, `odor_sequence` D,F —
D poked 57.6 ms, F "poked" 9.832 ms, `last_odor_position=1`. The F poke ended before the F
valve opened; the entry is a grace artifact.

### What 4a does, and what it defers

**4a — no value change, regression stays GREEN:**

Consolidate the three existing implementations into the two helpers above, keeping today's
behaviour exactly, and point all four consumers at them. Definition **B is deleted in the
process** — `plot_position_completion_rate` stops carrying its own `max(num_odors)` walk, so
that defect is fixed for free rather than needing a separate patch. Because the plotters'
denominators are not in the regression fingerprint, this changes figures but no metric values.

**Deferred to Phase 6/7** — see the TODO in `restructure_2_plan.md` §7b: write the 0 ms
positions, add `poke_source`. Only then is `only_true_pokes` computable. That change alters
`trial_data`, so it takes a deliberate fixture regeneration at that point — not in 4a.

---

## Tally

| file | lines | functions carrying metric math | of which DEDUP / VARIANT / NEW |
|---|---|---|---|
| `visualization_utils.py` | 7,264 | 19 | 6 / 8 / 8 |
| `movement_analysis_utils.py` | 4,460 | 3 (+2 threshold dedups) | 2 / 0 / 1 (but that one is 591 lines and 7 metrics) |
| `pred_seq_utils.py` | 1,886 | 7 | 0 / 1 / 6 |
| `sing_rew.py` | 1,291 | 3 | 0 / 0 / 3 |
| `valve_poke_plots.py` | 646 | 0 (1 loader dedup) | — |
| `modelling/switchpoint/plots.py` | 638 | 0 | — |
| `movement_analysis/sing_rew_movement.py` | 433 | 0 | — |
| **total** | **16,618** | **32 sites** | **24 metrics to add** |

The plan estimated "~27 metric/accuracy/rate-ish functions". The measured figure is **32
sites of metric math across 4 of the 7 files**, resolving to **24 metrics that do not exist
in `metric_analysis` today** plus 8 exact duplicates and 9 granularity variants.

Two of the three files that need no metric work are the two the plan's original count
missed — `modelling/switchpoint/plots.py` is already exemplary, and
`movement_analysis/sing_rew_movement.py` needs only de-duplication.

<!-- AUDIT-APPEND-HERE -->

---

## Open questions for Joschua

### Settled decisions

- **D0 — the metric signature.** Settled 2026-08-05: yes, as part of 4a. See "D0 resolution".
- **Q5 — "reached at position *p*".** Settled 2026-08-05, evidence-led. See "Q5 resolution".

### Local judgement calls — settled 2026-08-05

1. **`_hr_odor_associations` (visualization_utils:705)** → **moves to `metric_analysis` as
   session metadata.** It infers `{HR odor → reward identity}` by voting over HR-success
   trials; that is a derived property of the session, not a plotting concern, even though
   colour selection is its only consumer today.

2. **`speed_analysis.parquet`** → **no question to answer; withdrawn.** The output path is
   built from the session directory at runtime (`results_dir = ses / "saved_analysis_results"`,
   `:2154`; `analysis_path = results_dir / "speed_analysis.parquet"`, `:2426`), so moving
   `compute_speed_analysis` into `metric_analysis` leaves the artifact exactly where it is.
   Checked for the `__file__`-derived state that broke the Phase 2a `io/paths.py` move — there
   is none. Clean source-only move.

3. **`_kw_mwu_by_group`** → **`metric_analysis/stats/kw_mwu.py`**, one module per test family
   rather than a single `stats.py` that accretes. Stays in this repo for now; it is generic
   enough to graduate to `hypnose_helpers` by the 0.2 test, but that only earns its place once
   a second repo (eeg, ephys) actually wants it — narrower is cheaper to move later than
   broader is to unpick.

4. **The 10× outlier rule** (`pred_seq_utils:1052`, `:1249`) → **stays in `visualization/` as
   a display filter.** Establishes a general principle for the rest of 4a:

   > **Metrics are computed raw. Filtering is a display concern.** Filtering at plot time is
   > always possible; un-filtering a metric that was saved pre-filtered is not.

   Two consequences to implement rather than discover:
   - The plotted values will differ from the saved metric values **by design**. That has to be
     stated where the filter is applied, or someone comparing a figure to `metrics_*.json`
     will read it as a bug.
   - The rule is currently written twice. It becomes **one shared display helper** in the
     `visualization/` prep module, not two copies.

5. **"Reached at position *p*"** → see "Q5 resolution".

6. **`_compute_real_time_offset` (`valve_poke_plots:219`)** → **delete it; call the loaders'
   offset.** It duplicates `io/loaders.load_all_streams:145-167` step for step — heartbeat
   load, `'%Y-%m-%dT%H-%M-%S'` folder parse, UTC→UK conversion,
   `real_time_ref - start_time` — differing only in a parent-directory fallback and a
   returned heartbeat span. Today every timestamp on `plot_valve_and_poke_events` is shifted
   by its own copy while the rest of the package uses the loader's, so drift between them
   would be silent (the consumer is a figure, which the regression never sees).

   Expose the offset computation from `io/loaders.py` and have the plot call it. **No
   behaviour change intended** — the two implementations agree today, so this is a
   deduplication, not a fix. If the extracted version turns out *not* to reproduce the
   plot's current timestamps, that is a finding to report, not to paper over: it would mean
   the two have already drifted. The parent-directory fallback and the heartbeat span must
   be preserved (the plot uses the span to preselect overlapping sessions).

**All six settled. Nothing blocks the 4a moves.**
