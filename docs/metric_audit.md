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

A and B coincide only when every completed trial has the same sequence length. This must
be settled before any per-position rate is deduplicated.

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

<!-- AUDIT-APPEND-HERE -->

---

## Open questions for Joschua

1. **`_hr_odor_associations` (visualization_utils:705)** — it infers `{HR odor → reward
   identity}` by voting over HR-success trials, and today only picks plot colours. It is a
   protocol fact, not a metric. Move it to `metric_analysis` (or `io`) as session metadata,
   or leave it in `visualization/` since colour is its only consumer?

*(further questions appended as the remaining files are audited)*
