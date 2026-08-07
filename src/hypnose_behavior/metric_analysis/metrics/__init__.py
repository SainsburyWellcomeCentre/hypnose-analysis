"""Metric definitions, grouped by behavioural construct.

restructure_2 Phase 4b splits `metric_analysis/metrics_utils.py` -- 2,639 lines
mixing I/O, orchestration, merging, saving and ~40 definitions -- so that the
file layout says what 4a made true: `metric_analysis` is the single definition
site for every metric in the package.

    accuracy      correct choices, response rate, the rolling reward fraction
    false_alarm   every FA-labelled quantity, incl. FA port bias and latency
    sequence      completion, abortion by odor and by position
    hidden_rule   hidden-rule performance, detection, and the HR split
    sampling      how long the animal spent at each odor port
    timing        response and reward latencies
    common        the predicates, rate reduction and frame slicing they share

plus `../movement.py` and `../sing_rew_metrics.py`, which already existed.

**Grouping is by construct, not by frame.** Which frame a metric consumes
(`trials` or `position_data`) is declared to the registry by a decorator
argument, so `fa_latency_from_pokeout` sits with the false alarms it measures
rather than with the other latencies.

Every metric is a pure `f(frame) -> value` **core** plus a thin
`*_session(results)` **wrapper** that prints and returns the same value (Phase
4a, decision D0). `run.py` calls the wrappers; anything wanting another
granularity calls the core through `resolvers.by_group` / `over_windows`.

Several of these had no canonical version before 4a -- they were computed inside
a plotter in `visualization/` and existed nowhere else (the audit's `NEW` class,
i.e. the "lose no metric" checklist). Their arithmetic reproduces the plotters';
what stayed behind is axis construction, cross-subject aggregation and styling,
which are properties of a figure rather than of the data. None of them is
reached by `run_all_metrics`, so none enters `metrics_*.json` or the regression
fingerprint.
"""
