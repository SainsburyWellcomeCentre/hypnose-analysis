# scripts/modelling

Model fits to behavioural trial sequences. Like the other `scripts/`, these are entry
points over functions in `src/hypnose/`; the numeric model lives in
[`hypnose.models.switchpoint_helpers`](../../src/hypnose/models/switchpoint_helpers.py). Run from the repo root in the project conda environment; the scripts add `src/` to the path, so no install is required.


| Script | What it does |
| --- | --- |
| `switchpoint_analysis.py` | Detects the LONG → SHORT strategy switch per animal, and tests whether switches align with sleep |


### The sequence being modelled

Each kept trial `i` becomes one binary outcome:

```
s[i] = 1  if hidden_rule_success == True   (SHORT: left early, used the hidden rule)
s[i] = 0  otherwise                        (LONG: waited out the full sequence)
```

### Reward identity (A / B)

Every trial also carries the identity of the reward it is associated with:

```
identity = first_supply_odor_identity        if it is "A" or "B"
         = last_odor as a bare letter        if that is "A" or "B"
         = "" (unresolved)                   otherwise
```
Colours are fixed: **A = `#E53935`** (red), **B = `#00796B`** (teal), unresolved = grey.

### The models

`s[i] ~ Bernoulli(p_i)`, with five competing descriptions of `p_i`:

| Model | `p_i` | Params | Meaning |
| --- | --- | --- | --- |
| `constant` | `p` | 1 | No strategy change |
| `switch` | `p1` if `i < tau` else `p2` | 3 | One abrupt change at trial `tau` |
| `logistic` | `lo + (hi - lo) · sigmoid(slope · (i - midpoint))` | 4 | A graded change |
| `switch2` | `p1` \| `p2` \| `p3` split by `tau1 < tau2` | 5 | Two abrupt changes — an overshoot, or a change arriving in two stages |
| `qlearning` | — | 2 | **Not implemented** (stub); scored `-inf`, so it never wins |

`tau` is the index of the **first trial of the post-switch regime** (so `1 ≤ tau ≤ n-1`; a
switch at 0 is just the constant model). The primary hypothesis is one directional switch,
`p1` low → `p2` high.

`switch2` maximizes **exhaustively** over all ordered pairs `tau1 < tau2`


#### `qlearning` (planned)

The `logistic` model **nests** the `switch` model as `slope → ∞`, which is what makes
`slope` a direct read-out of **abruptness**: a large fitted slope is a step, a small one is
a slow drift.

### How the logistic is fitted (multi-start)

The logistic likelihood surface has **more than one basin**, and which one Nelder-Mead settles
in is decided mostly by the *initial slope*.

`fit_logistic` minimizes from **16 dispersed initial conditions** and keeps the best:

- the switch-point warm start (`midpoint = tau`, asymptotes at `p1`/`p2`), and
- a 5 × 3 grid — midpoints at the 10/30/50/70/90% quantiles of the trial axis, both asymptotes
  at the global SHORT rate, and initial slopes `0.05` / `0.5` / `5.0` (shallow-gradual through
  steep-near-step).

The start set lives in **one place**, `logistic_start_points()` — `fit_logistic`,
`fit_logistic_multistart` and the diagnostic all replay that same list, so what is plotted is
what is fitted. Add a start there and nowhere else.

Because the logistic nests the step, its optimum **must** be at least as good as the switch
model's. `fit_logistic` warns if it is not: that means every start got stuck, and it is an
optimization failure rather than a finding.

### How `tau` is estimated (no MCMC)

For a Bernoulli switch model the segment rates that maximize the likelihood are just the
segment means, so `p1` and `p2` can be profiled out analytically. That leaves a
one-dimensional search over `n` candidate switch trials, and each candidate's log-likelihood
is a function of the running success count — computable for **all** candidates at once from a
single prefix sum. The whole profile is therefore `O(n)` and exact, with no sampler, no
tuning, and no change-point package.

Probabilities are clipped away from 0 and 1 so a degenerate (all-0 or all-1) segment cannot
produce an infinite log-likelihood.

Under a uniform prior on `tau`, the normalized profile likelihood **is** the posterior over
the switch trial. It is exponentiated with its maximum subtracted, so nothing underflows.
Two widths summarize it:

- **95% HDI** (primary) — take trials in order of descending posterior mass until 95% is
  covered, then report the lowest and highest index in that set. Reported as a range, so it
  can be wide when the posterior is multimodal.
- **FWHM** (secondary) — first and last trial with at least half the peak mass. Sharper, but
  it says nothing about how much total mass the interval holds.

A narrow HDI means the animal's switch is localizable to a few trials — the signature of an
abrupt change.

### Model comparison

`AIC = 2k − 2·loglik` and `BIC = k·ln(n) − 2·loglik`; lower is better. BIC penalizes the
extra parameters harder, so it is the stricter test that a switch happened at all. Reading
the winner: `constant` → no switch; `switch` → one abrupt change; `logistic` → gradual;
`switch2` → two changes (an overshoot, or a change in two stages). `qlearning` is scored but,
being a stub, is never eligible to win. Read the caveat above before leaning on a narrow win.

### The sleep test (`run_permutation`)

For each animal with a switch, `f` = the number of trials from the start of the session
containing `tau` to `tau` itself. **Small `f` means the switch happened soon after sleep.**

- **Statistic**: the mean of `f` across included animals.
- **Null**: switches are unrelated to that animal's own sleep timing. It is realized by
  *donating* boundaries across animals — each recipient keeps its real `tau` and its own trial
  axis, but is scored against another animal's session starts. This preserves each animal's
  real `tau` and each donor's real session structure, and breaks only the pairing between them.
- **Direction**: one-sided, testing that switches sit *closer* to real sleep than chance.

```
p = (1 + #{null mean <= observed mean}) / (n_permutations + 1)
```

The `+1` correction keeps `p` strictly positive. A small `p` means real `f` is smaller than
donated `f` — switches track that animal's *own* sleep boundaries rather than merely occurring
at some typical depth into a session.

**The null is paired, not pooled.** One permutation assigns every recipient exactly **one**
donor (without replacement where possible) and takes the mean `f` over recipients;
`n_permutations` of these give the null distribution. Pooling all `N·(N-1)` recipient × donor
values instead would understate the null's spread, because it averages over far more values
than the observed statistic does. The exhaustive pooled array is still returned and plotted as
`shuffled_f`, but the p-value comes from the paired null.

#### The span guard

A donated boundary set is only meaningful where the donor actually has trials. If a recipient's
`tau` falls beyond the donor's trial axis — recipient switches at trial 5000, donor only ran
3000 trials — then `f` would be measured from the donor's *final* session start, an arbitrarily
inflated value that biases the null upward. **Such (recipient, donor) pairs are dropped**, from
the null and from the pooled array alike, and the count is printed.

The cutoff is the donor's **last trial**, not its last session start: a `tau` between the
donor's last session start and its last trial still lands inside a real donated session and is
kept. A recipient left with no valid donor at all is dropped from the test and reported in
`excluded_no_donor` — though it still donates its own boundaries to the others.

Only animals that actually switched are included. The rule is the `inclusion` parameter:

| `inclusion` | Keeps an animal when |
| --- | --- |
| `bic_switch_wins` *(default)* | The single-switch model has the lowest BIC of all — the change is real **and** abrupt. Note `switch2` now competes, so an animal best described by two changes is *excluded* by this rule |
| `bic_beats_constant` | The switch model beats the constant model on BIC; a gradual fit may still be better |
| `aic_switch_wins` | As the default, under the milder AIC penalty |
| `all` | No filtering |

---

## Entry points

Both are importable from a notebook and runnable from the terminal, and neither depends on
the other having run. They select their own subjects and recompute their own fits.

```python
from switchpoint_analysis import run_analysis, run_permutation

results = run_analysis(
    subjids=[40, 45],
    date_ranges={40: (20251201, 20251231), 45: None},
    rewarded_only=True,
    likelihood_window=100,
    split_ab=False,      # True -> fit and plot the A- and B-reward trials separately
    show=False,          # keep the figures rather than displaying them
)
results[40]["tau"], results[40]["hdi_width"], results[40]["figures"]["posterior"]

# with split_ab=True the per-subject value is nested by reward identity:
#   results[40]["A"]["tau"], results[40]["B"]["figures"]["strategy"]

perm = run_permutation(
    subjids=[40, 45, 48, 50],       # may be a different set of animals
    date_ranges={s: None for s in (40, 45, 48, 50)},
    rewarded_only=True,
    inclusion="bic_switch_wins",
    n_permutations=10000,
    seed=0,                          # reproducible null
    show=False,
)
perm["observed_mean"], perm["p_value"], perm["null_means"], perm["n_pairs_dropped"]
```

`date_ranges` maps each subject to an inclusive `(start, end)` `YYYYMMDD` tuple, an explicit
date list, or `None` for all sessions. A `{subjid: date_range}` dict may be passed as
`subjids` on its own, matching the convention of the plotters in `hypnose.visualization`.

### `run_analysis` — three figures per animal

Shown in this order, one animal at a time, so an animal's figures stay together rather than
interleaving with another's. **SHORT is the lower row, LONG the upper row** in the first two.

1. **Strategy** — SHORT/LONG per trial on the continuous trial axis, each trial coloured by
   its reward identity (A red, B teal, unresolved grey), with a blue dotted vertical line at
   each session end (sleep).
2. **Model comparison** — the data with every fitted model overlaid (constant line, switch
   step, switch2 two-step, logistic curve), plus an empirical 21-trial rolling mean, and the
   five-row AIC/BIC table in-panel with the BIC winner marked, so *no switch / abrupt /
   gradual / two-stage* can be read off directly. The printed table adds each model's loglik,
   the nesting check, and the winner's fitted parameters.
3. **Posterior** — the switch-point posterior over *all* trials, plotted windowed to
   ±`likelihood_window` trials around its peak. `tau`, its session, and the HDI width are
   printed and annotated (HDI primary, FWHM secondary).

#### `split_ab`

With `split_ab=True` the trials are split by reward identity and **each subset is modelled and
plotted independently** — six figures per animal, the three above for A then the three for B.
Each subset gets its own contiguous `0 .. m-1` trial axis, because the switch-point index must
index the sequence being fitted; `tau` is therefore a position within that subset, and
`global_tau` reports the same trial's position on the full, unsplit axis. Session boundaries
are recomputed over the subset, and sessions holding no trial of that identity drop out.
Unresolved trials belong to neither subset and are dropped, with a count printed.

This matters when an animal adopts the SHORT strategy for one reward before the other — subject
40 switches for B at global trial 1174 (session 20251208) but not for A until global trial 1623
(20251212), a four-day gap the pooled fit averages away.

Returns a dict keyed by subjid holding `tau`, `tau_session`, `hdi`, `hdi_width`, `fwhm`,
`p1`, `p2`, `comparison`, `session_ends`, `session_starts`, and the `figures`.

### `run_logistic_diagnostic` — is the logistic fit trustworthy?

A **standalone** diagnostic; `run_analysis` does not call it. Per animal it replays the shipped
start set and shows where each initial condition ends up, so you can see whether the starts
funnel into one optimum or split into basins:

- every converged sigmoid overlaid on the data, one colour per start — **winner bold**, the
  switch-point warm start **dashed**;
- in the margin above the data, each start's **initial** midpoint (▽) joined by a faint
  connector to where it **converged** (○), in that start's colour;
- a printed per-start table: initial midpoint → converged midpoint, converged slope, converged
  loglik, with `[best]` / `[warm]` tags, the number of distinct basins, and a note when a
  dispersed start beat the warm start.

```python
from switchpoint_analysis import run_logistic_diagnostic

diag = run_logistic_diagnostic([40], {40: (20251125, 20251231)}, rewarded_only=True,
                               split_ab=True, show=False)
diag[40]["A"]["best_label"], diag[40]["A"]["n_basins"], diag[40]["A"]["fig"]
```

Honours `split_ab` with the same A/B partition `run_analysis` uses, producing a table and a
figure per reward identity, with the reward type in every title and header.

### `run_permutation` — one two-panel figure

**Left**: two boxplots with the points overlaid — real `f` (one point per included animal) and
the span-guarded pool of shuffled `f` (one point per valid recipient × donor pair), annotated
with the observed mean and the p-value. **Right**: the paired-permutation null distribution of
the mean `f`, with the observed mean marked.

Returns `real_f`, `shuffled_f`, `null_means`, `observed_mean`, `p_value`, `n_permutations`,
`n_pairs_dropped`, `included_subjids`, `excluded_subjids` (no switch), `excluded_no_donor` (no
donor spans their `tau`), `per_subject`, and `fig`.

---

## Terminal usage

```bash
# per-animal switch-point fit and figures
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 --likelihood-window 100

# restrict to a date range, rewarded trials only
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 45 \
    --date-range 20251201 20251231 --rewarded-only

# fit the A- and B-reward trials separately
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 \
    --date-range 20251125 20251231 --rewarded-only --split-ab

# where does each logistic multi-start initial condition converge?
python scripts/modelling/switchpoint_analysis.py diagnostic --subjids 40 --rewarded-only --split-ab

# do switches align with sleep?
python scripts/modelling/switchpoint_analysis.py permutation --subjids 40 45 48 50 --rewarded-only

# a looser inclusion rule, more permutations, a different seed
python scripts/modelling/switchpoint_analysis.py permutation --subjids 40 45 48 50 51 \
    --inclusion bic_beats_constant --n-permutations 50000 --seed 1
```

| Argument | Subcommand | Meaning |
| --- | --- | --- |
| `--subjids ID [ID ...]` | both | Subject id(s). Required. |
| `--dates D [D ...]` | both | Specific dates `YYYYMMDD`. |
| `--date-range START END` | both | Inclusive `YYYYMMDD` range (alternative to `--dates`). |
| `--rewarded-only` | both | Keep only rewarded trials; aborts are always dropped. |
| `--likelihood-window N` | `analysis` | Half-width of the posterior plot window (default 100). |
| `--split-ab` | `analysis`, `diagnostic` | Fit and plot the A- and B-reward trials separately. |
| `--inclusion RULE` | `permutation` | Which animals count as having switched (default `bic_switch_wins`). |
| `--n-permutations N` | `permutation` | Permutations drawn for the null (default 10000). |
| `--seed N` | `permutation` | RNG seed for a reproducible null (default 0). |

`--dates` and `--date-range` are mutually exclusive; omit both for all dates. The CLI applies
one date range to every subject — for per-subject ranges, call the functions directly.
Subjects with no data are skipped via `hypnose.qc.validate.validate_subject`, as in the other
scripts.

## Caveats

- **The p-value's resolution is set by the number of animals, not by `n_permutations`.** With
  `N` recipients the paired null has at most `!N`-ish distinct donor assignments (9 for `N`=4,
  before the span guard removes some), so raising `n_permutations` only resamples the same
  handful of values — the null histogram is visibly discrete. `run_permutation` needs at least
  **two** included animals with a valid donor and raises otherwise, but a two- or three-animal
  test cannot produce a small `p` no matter how many permutations are drawn.
- Animals excluded by the span guard (`excluded_no_donor`) still act as donors for the others.
  They contribute no `f` of their own, so the observed statistic averages over fewer animals
  than were fitted.
- `f` is NaN when a `tau` precedes every session start. Session starts always begin at trial
  0 by construction, so this cannot occur for real or donated boundaries — it is a guard, not
  an expected case.



Quick explanations: 
tau = switch trial, printed with session 
p1, p2 = two regime rates. Probability of short-sequence solve before and after switch. 
95% HDI: uncertainty where switch happens. (how many trials cover 95% confidence)
FWHM: All trials whose posterior probability are within half of the width of the peak
AIC/BIC: both score model performance
