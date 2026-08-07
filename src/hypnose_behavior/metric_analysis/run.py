# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Running the metric set: one session, or a batch with merges.

Mirrors ``trial_classification/run.py`` -- the orchestration, not the
definitions. Moved out of ``metrics_utils.py`` in restructure_2 Phase 4b.

``run_all_metrics`` is what writes ``metrics_<subj>_<date>.json`` and
``.txt``. The **json** is fingerprinted by ``qc/regression.py``; the **txt** is
the wrappers' captured stdout and is not, so a print-only change here is
invisible to that gate -- check both when touching this file.
"""

import contextlib
import io
import json
from pathlib import Path

from hypnose_behavior.io.layout import derivatives
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.io.results import load_session_results
from hypnose_behavior.metric_analysis.merge import pool_results_dicts
from hypnose_behavior.metric_analysis.sing_rew_metrics import (
    compute_sing_rew_metrics,
    compute_sing_rew_rates,
    is_singrew_session,
)
from hypnose_behavior.metric_analysis.registry import REGISTRY
from hypnose_behavior.metric_analysis.summary import save_merged_metrics_txt
from hypnose_behavior.utils.helpers import _filter_session_dirs
# Imported for their **registrations**: a metric declares itself where it is
# defined, so every definition module must be imported before REGISTRY is read.
from hypnose_behavior.metric_analysis.metrics import (  # noqa: F401
    accuracy,
    false_alarm,
    hidden_rule,
    sampling,
    sequence,
    timing,
)
from hypnose_behavior.metric_analysis.metrics.false_alarm import fa_abortion_stats_session

__all__ = ["run_all_metrics", "batch_run_all_metrics_with_merge", "REPORT"]

# The order metrics are reported in, and therefore the order of every
# `metrics_<subj>_<date>.txt` on disk. Deliberately **not** derived from
# registration order, which would make it a function of import order -- i.e. of
# a file layout that has just changed twice. Being in REGISTRY makes a metric
# discoverable; being here is the separate decision to report and save it, which
# is why the metrics 4a recovered from `visualization/` are registered and
# absent below.
REPORT = [
    "decision_accuracy",
    "decision_accuracy_by_odor",
    "global_choice_accuracy",
    "premature_response_rate",
    "response_contingent_FA_rate",
    "global_FA_rate",
    "FA_odor_bias",
    "FA_position_bias",
    "sequence_completion_rate",
    "odorx_abortion_rate",
    "hidden_rule_performance",
    "hidden_rule_detection_rate",
    "hidden_rule_counts_by_odor",
    "choice_timeout_rate",
    "avg_sampling_time_odor_x",
    "avg_sampling_time_completed_sequence",
    "avg_sampling_time_aborted_sequence",
    "abortion_rate_positionX",
    "avg_response_time",
    "FA_avg_response_times",
    "response_rate",
    "manual_vs_auto_stop_preference",
    "odor_initiation_bias",
    "fa_abortion_stats",
    "fa_port_ratio_by_odor",
]


def _report_fa_abortion_stats(results):
    """`fa_abortion_stats`' saved shape, printing its three tables on the way.

    The one reported metric whose report is tables rather than a value, so it
    does not fit the loop's `wrapper -> adapter` shape.
    """
    fa_ab_stats = fa_abortion_stats_session(results, return_df=True)
    if fa_ab_stats is None:
        return None
    payload = {
        'by_odor': fa_ab_stats[0].to_dict(orient='records') if hasattr(fa_ab_stats[0], 'to_dict') else None,
        'by_position': fa_ab_stats[1].to_dict(orient='records') if hasattr(fa_ab_stats[1], 'to_dict') else None,
        'by_odor_position': fa_ab_stats[2].to_dict(orient='records') if hasattr(fa_ab_stats[2], 'to_dict') else None,
    }
    print("\nFA Abortion Stats by Odor:")
    print(fa_ab_stats[0].to_string(index=False) if hasattr(fa_ab_stats[0], 'to_string') else fa_ab_stats[0])
    print("\nFA Abortion Stats by Position:")
    print(fa_ab_stats[1].to_string(index=False) if hasattr(fa_ab_stats[1], 'to_string') else fa_ab_stats[1])
    print("\nFA Abortion Stats by Odor and Position:")
    print(fa_ab_stats[2].to_string(index=False) if hasattr(fa_ab_stats[2], 'to_string') else fa_ab_stats[2])
    return payload

__all__ = ["run_all_metrics", "batch_run_all_metrics_with_merge"]


def run_all_metrics(results, save_txt=True, save_json=True):
    """
    Run all metrics, print results, and save to txt and json in the session's results directory.
    Returns a dict of all metric values.
    """
    derivatives_dir = get_derivatives_root()
    manifest = results.get("manifest", {}) or {}
    summary = results.get("summary", {}) or {}

    def _safe_session_value(container, *keys):
        cur = container
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    subjid = (
        _safe_session_value(manifest, "session", "subject_id")
        or _safe_session_value(summary, "session", "subject_id")
        or _safe_session_value(manifest, "session", "subjid")
        or _safe_session_value(summary, "session", "subjid")
    )
    date = (
        _safe_session_value(manifest, "session", "date")
        or _safe_session_value(summary, "session", "date")
        or _safe_session_value(manifest, "session", "session_date")
        or _safe_session_value(summary, "session", "session_date")
    )

    paths = manifest.get("paths", {}) if isinstance(manifest, dict) else {}
    sub_folder = paths.get("sub_folder")
    ses_folder = paths.get("ses_folder")
    manifest_results_dir = manifest.get("results_dir")
    results_dir_hint = (
        results.get("results_dir")
        or results.get("_results_dir")
    )

    def _is_relative_to(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def _normalize_subjid(value):
        if value is None:
            return None
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return f"sub-{digits.zfill(3)}" if digits else None

    def _normalize_date(value):
        if value is None:
            return None
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return digits if digits else None

    def _clean_folder_component(component: str) -> Path | None:
        if not component:
            return None
        sanitized = component.strip().replace("..", "")
        return Path(sanitized).name if sanitized else None

    def _session_dir_from_manifest_parts() -> Path | None:
        sub_comp = _clean_folder_component(sub_folder)
        ses_comp = _clean_folder_component(ses_folder)
        if not sub_comp or not ses_comp:
            return None
        return derivatives_dir / sub_comp / ses_comp / "saved_analysis_results"

    def _session_dir_from_ids() -> Path | None:
        # One link in a fallback chain, so every failure is None rather than an
        # exception -- including the ambiguous-tree errors the shared resolver
        # raises, which the next candidate may well sidestep.
        sub_norm = _normalize_subjid(subjid)
        date_norm = _normalize_date(date)
        if not sub_norm or not date_norm:
            return None
        try:
            found = derivatives.find_sessions(sub_norm, date=date_norm, missing_ok=True)
        except (ValueError, OSError):
            return None
        return found[0].path / "saved_analysis_results" if found else None

    def _determine_output_dir() -> Path:
        if results_dir_hint:
            return Path(results_dir_hint).expanduser().resolve(strict=False)
        if manifest_results_dir:
            candidate = Path(manifest_results_dir).expanduser().resolve(strict=False)
            if _is_relative_to(candidate, derivatives_dir.resolve(strict=False)):
                return candidate
        manifest_candidate = _session_dir_from_manifest_parts()
        if manifest_candidate is not None:
            return manifest_candidate
        id_candidate = _session_dir_from_ids()
        if id_candidate is not None:
            return id_candidate
        raise RuntimeError(
            "Could not determine output directory for metrics. "
            "Ensure manifest contains valid paths or run load_session_results() before run_all_metrics()."
        )

    need_output = bool(save_txt or save_json)
    out_dir: Path | None = None
    if need_output:
        out_dir = _determine_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Run metrics and capture output ---
    metrics = {}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for name in REPORT:
            spec = REGISTRY[name]
            print(f"\n--- {spec.title} ---")
            if name == "fa_abortion_stats":
                metrics[spec.key] = _report_fa_abortion_stats(results)
                continue
            value = spec.session(results)
            metrics[spec.key] = spec.adapter(value) if spec.adapter else value

        # Single-reward protocol only: outcome-category metrics (Hit / Miss / FA / CR)
        # built from the singrew trial_data columns. Only computed when the session's
        # stage_name contains "singrew", so default-protocol output is unchanged.
        if is_singrew_session(results):
            print("\n--- Single-Reward Outcome Categories ---")
            sing_rew = compute_sing_rew_metrics(results)
            metrics['sing_rew_categories'] = sing_rew
            sing_rew_rates = compute_sing_rew_rates(sing_rew)
            metrics['sing_rew_metrics'] = sing_rew_rates
            print(f"Total trials: {sing_rew.get('total_trials', 0)}")
            for cat in ("hit", "miss", "false_alarm", "correct_rejection",
                        "premature_port_entry", "premature_abort", "uncategorized"):
                cat_info = sing_rew.get(cat, {})
                print(f"{cat}: n={cat_info.get('n', 0)}")
                for sub, sub_info in cat_info.get("subcategories", {}).items():
                    print(f"    {sub}: n={sub_info.get('n', 0)}")
            val = sing_rew.get("validation", {})
            print(f"Validation: classified {val.get('n_classified', 0)}/"
                  f"{val.get('n_total_trials', 0)}")
            not_any = val.get("not_in_any_category", {})
            in_multi = val.get("in_multiple_categories", {})
            if not_any.get("n", 0):
                print(f"  [FLAG] {not_any['n']} trial(s) in NO category: "
                      f"{not_any.get('global_trial_ids', [])}")
            if in_multi.get("n", 0):
                print(f"  [FLAG] {in_multi['n']} global_trial_id(s) in MULTIPLE categories: "
                      f"{in_multi.get('global_trial_ids', {})}")
            if val.get("n_trials_missing_global_trial_id", 0):
                print(f"  [FLAG] {val['n_trials_missing_global_trial_id']} trial(s) "
                      f"missing global_trial_id")
            print("\n--- Single-Reward Metrics ---")
            counts = sing_rew_rates.get("counts", {})
            print(f"n_go={counts.get('n_go', 0)} n_nogo={counts.get('n_nogo', 0)} "
                  f"n_amb={counts.get('n_amb', 0)} n_det={counts.get('n_det', 0)} "
                  f"n_tot={counts.get('n_tot', 0)}")
            for key in ("hit_rate", "fa_rate", "H_prime", "F_prime", "headline_sensitivity",
                        "criterion", "balanced_accuracy", "earned_reward_rate", "port_accuracy",
                        "efficient_rejection_rate", "early_rejection_index", "anticipatory_rate",
                        "forfeit_rate", "omission_rate", "impulsivity_rate", "impatience_rate"):
                print(f"    {key}: {sing_rew_rates.get(key)}")

    # Print to screen
    print(buffer.getvalue())

    # --- Save TXT and JSON ---
    if save_txt:
        txt_path = out_dir / f"metrics_{subjid}_{date}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"Saved metrics summary to {txt_path}")
    if save_json:
        json_path = out_dir / f"metrics_{subjid}_{date}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Saved metrics values to {json_path}")

    return metrics


def batch_run_all_metrics_with_merge(
    subjids=None,
    dates=None,
    protocol=None,
    save_txt=True,
    save_json=True,
    verbose=True
):
    """
    Batch run metrics for combinations of subjids and dates, with optional protocol filter.
    Also computes and saves merged metrics across all sessions, per subject, and across all subjects.
    """
    derivatives_dir = get_derivatives_root()
    results = []
    results_dicts = []

    # Track session processing stats per subject
    session_stats = {}  # Format: {subjid: {'analyzed': [dates], 'skipped': [(date, reason)], 'failed': [(date, error)]}}

    # Find all subject directories. Sorted by subject number, where the previous glob
    # returned them in filesystem order -- so a cohort now merges in the same order on
    # every machine.
    subj_dirs = [d for _, d in derivatives.iter_subjects(subjids)]
    if verbose:
        print(f"Found {len(subj_dirs)} subject directories.")

    def _print_session_banner(subjid_str: str, date_str: str):
        banner = f"\n ======================= Subject {subjid_str} Date {date_str} ======================="
        print(banner)

    for subj_dir in subj_dirs:
        subj_results = []  # Store results for this subject
        subj_dates = []  # Track processed dates for this subject
        subjid = subj_dir.name.split("_")[0].replace("sub-", "")
        session_stats[subjid] = {'analyzed': [], 'skipped': [], 'failed': []}

        # Find all session directories for this subject
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        if not ses_dirs:
            continue
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            summary_path = results_dir / "summary.json"
            date = ses_dir.name.split("_date-")[-1]
            
            if not summary_path.exists():
                if verbose:
                    print(f"Skipping {subjid} date {date}: summary.json not found at {summary_path}")
                session_stats[subjid]['skipped'].append((date, "summary.json not found"))
                continue
            
            # Protocol filter
            skip_protocol = False
            if protocol is not None:
                try:
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    runs = summary.get("session", {}).get("runs", [])
                    if not runs or "stage" not in runs[0]:
                        skip_protocol = True
                    else:
                        stage_name = runs[0]["stage"].get("stage_name", "")
                        if protocol not in stage_name:
                            skip_protocol = True
                except Exception as e:
                    if verbose:
                        print(f"Skipping {subjid} date {date}: Protocol filter error - {e}")
                    session_stats[subjid]['skipped'].append((date, f"Protocol filter error: {e}"))
                    continue
                
                if skip_protocol:
                    if verbose:
                        print(f"Skipping {subjid} date {date}: Does not match protocol '{protocol}'")
                    session_stats[subjid]['skipped'].append((date, f"Protocol '{protocol}' not in stage"))
                    continue
            
            # Run metrics
            try:
                if verbose:
                    _print_session_banner(subjid, date)
                session_results = load_session_results(subjid, date)
                metrics = run_all_metrics(
                    session_results,
                    save_txt=save_txt,
                    save_json=save_json
                )
                subj_results.append(session_results)  # Collect results for this subject
                subj_dates.append(date)  # Track processed dates for this subject
                results_dicts.append(session_results)  # Add to global results
                session_stats[subjid]['analyzed'].append(date)
                if verbose:
                    print(f"Processed subjid={subjid}, date={date}")
            except Exception as e:
                if verbose:
                    print(f"Failed for subjid={subjid}, date={date}: {e}")
                session_stats[subjid]['failed'].append((date, str(e)))

        # --- Merge results for this subject ---
        if subj_results:
            def _range_str(dates_list):
                unique_sorted = sorted(set(dates_list))
                if not unique_sorted:
                    return "None"
                return unique_sorted[0] if len(unique_sorted) == 1 else f"{unique_sorted[0]}-{unique_sorted[-1]}"

            pooled_results = pool_results_dicts(subj_results)
            # --- Capture pretty print output ---
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                merged_metrics = run_all_metrics(pooled_results, save_txt=False, save_json=False)
            pretty_print_str = buffer.getvalue()
            if len(subj_results) > 1:
                banner_range = _range_str(subj_dates)
                print(f"\n======================= Subject {subjid} Summary {banner_range} =======================")
                print(pretty_print_str)
            elif verbose:
                print(f"Merged metrics not echoed to console for subjid={subjid} (single session). Files still saved.")
            # Prepare header
            header = (
                "Merged Results for:\n"
                f"Subjid: {subjid}\n"
                f"Date(s): {', '.join(subj_dates)}\n"
                f"Protocol: {protocol if protocol else 'all'}"
            )
            subj_dates_sorted = sorted(subj_dates)
            first_date = subj_dates_sorted[0][4:]
            last_date = subj_dates_sorted[-1][4:]
            # Output directory and filenames
            merged_dir = subj_dir / "merged_results"
            merged_dir.mkdir(parents=True, exist_ok=True)
            fname = f"merged_{subjid}_{protocol if protocol else 'all'}_{first_date}_to_{last_date}"
            txt_path = merged_dir / f"{fname}.txt"
            json_path = merged_dir / f"{fname}.json"
            # Save txt using the pretty print string
            save_merged_metrics_txt(merged_metrics, header, txt_path, pretty_print_str=pretty_print_str)
            if verbose:
                print(f"Saved merged metrics summary for subjid={subjid} to {txt_path}")
            # Save json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(merged_metrics, f, indent=2, default=str)
            if verbose:
                print(f"Saved merged metrics values for subjid={subjid} to {json_path}")

    # --- Total merged metrics across all subjects ---
    if results_dicts:
        pooled_results = pool_results_dicts(results_dicts)
        # --- Capture pretty print output ---
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            merged_metrics = run_all_metrics(pooled_results, save_txt=False, save_json=False)
        pretty_print_str = buffer.getvalue()
        # Prepare header
        subjids_merged = pooled_results["manifest"]["merged_subjects"]
        dates_merged = pooled_results["manifest"]["merged_dates"]
        protocol_merged = pooled_results["manifest"]["protocol"]
        header = (
            "Merged Results for:\n"
            f"Subjid(s): {', '.join(subjids_merged)}\n"
            f"Date(s): {', '.join(dates_merged)}\n"
            f"Protocol: {protocol_merged if protocol_merged else 'all'}"
        )
        # Extract first and last dates
        dates_sorted = sorted(dates_merged)
        first_date = dates_sorted[0][4:]  # Extract MMDD from YYYYMMDD
        last_date = dates_sorted[-1][4:]  # Extract MMDD from YYYYMMDD
        # Output directory and filenames
        merged_dir = derivatives_dir / "merged"
        if protocol is not None:
            merged_dir = merged_dir / "protocol_merged"
        else:
            merged_dir = merged_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        subjids_str = "_".join(subjids_merged)
        fname = f"merged_subjids_{subjids_str}_{protocol_merged if protocol_merged else 'all'}_{first_date}_to_{last_date}"
        txt_path = merged_dir / f"{fname}.txt"
        json_path = merged_dir / f"{fname}.json"
        # Save txt using the pretty print string
        save_merged_metrics_txt(merged_metrics, header, txt_path, pretty_print_str=pretty_print_str)
        if verbose:
            print(f"Saved total merged metrics summary to {txt_path}")
        # Save json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_metrics, f, indent=2, default=str)
        if verbose:
            print(f"Saved total merged metrics values to {json_path}")
        # Only echo the combined summary when spanning multiple subjects or dates
        if len(subjids_merged) > 1 or len(set(dates_merged)) > 1:
            banner_dates = sorted(set(dates_merged))
            date_range = banner_dates[0] if len(banner_dates) == 1 else f"{banner_dates[0]}-{banner_dates[-1]}"
            print(f"\n======================= Subjects {subjids_str} {date_range} =======================")
            print(pretty_print_str)
        elif verbose:
            print("Merged metrics not echoed to console for single subject/date (already shown above).")

    # ===== FINAL SESSION SUMMARY =====
    print("\n" + "="*80)
    print("SESSION PROCESSING SUMMARY")
    print("="*80)
    
    for subjid in sorted(session_stats.keys()):
        stats = session_stats[subjid]
        analyzed = stats['analyzed']
        skipped = stats['skipped']
        failed = stats['failed']
        
        print(f"\nSubject ID: {subjid}")
        print(f"  ✓ Analyzed ({len(analyzed)}): {', '.join(analyzed) if analyzed else 'None'}")
        
        if skipped:
            print(f"  ⊘ Skipped ({len(skipped)}):")
            for date, reason in skipped:
                print(f"      - {date}: {reason}")
        else:
            print(f"  ⊘ Skipped: None")
        
        if failed:
            print(f"  ✗ Failed ({len(failed)}):")
            for date, error in failed:
                print(f"      - {date}: {error}")
        else:
            print(f"  ✗ Failed: None")
    
    print("\n" + "="*80)
    total_analyzed = sum(len(s['analyzed']) for s in session_stats.values())
    total_skipped = sum(len(s['skipped']) for s in session_stats.values())
    total_failed = sum(len(s['failed']) for s in session_stats.values())
    print(f"TOTALS: Analyzed={total_analyzed} | Skipped={total_skipped} | Failed={total_failed}")
    print("="*80 + "\n")

    return results
