# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

import sys
import os
from pathlib import Path
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
import math
from glob import glob
import ast
from IPython.display import display
import io
import contextlib
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Iterable, Optional, Union
from hypnose_behavior.utils.helpers import _filter_session_dirs
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.io.layout import derivatives, normalize_subjid
from hypnose_behavior.metric_analysis.sing_rew_metrics import (
    compute_sing_rew_metrics,
    compute_sing_rew_rates,
    is_singrew_session,
)
# `parse_json_column` is defined in frames.py (it is frame construction, not a
# metric) and re-exported here so existing importers keep working.
from hypnose_behavior.metric_analysis.frames import (  # noqa: F401
    build_position_data,
    odor_letter,
    odor_sequence_tokens,
    parse_json_column,
    presented_positions,
    reached_counts as _reached_counts,
    sampled_positions,
    sequence_depth,
)
# ================== Loading, Wrapper, and Helper Functions ==================

def load_session_results(subjid, date):
    """
    Load saved analysis results for a given subject and date.
    Returns a dict with trial_data, non-initiated tables, and metadata.
    """
    # One resolver for the whole family (restructure_2 Phase 2b); it reports the
    # available sessions on a miss and raises rather than warning on an ambiguous
    # subject or date.
    session = derivatives.find_session(subjid, date=date)
    subject_dir = session.subject_dir
    session_dir = session.path

    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load manifest and summary
    manifest = json.load(open(results_dir / "manifest.json"))
    summary = json.load(open(results_dir / "summary.json"))

    results: dict = {}

    # Prefer the unified trial_data parquet; fall back to CSV if needed
    trial_parquet = results_dir / "trial_data.parquet"
    trial_csv = results_dir / "trial_data.csv"
    trial_df = pd.DataFrame()
    if trial_parquet.exists():
        try:
            trial_df = pd.read_parquet(trial_parquet)
        except Exception as e:
            print(f"Warning: failed to read {trial_parquet}: {e}")
    if trial_df.empty and trial_csv.exists():
        trial_df = pd.read_csv(trial_csv)
    results["trial_data"] = trial_df

    # Long per-position frame, derived here rather than written by the classifier,
    # so metrics never parse a JSON blob and legacy sessions need no
    # compatibility branch (D0, tier 2). Phase 7b's position_data side-table
    # turns this from a derivation into a read.
    results["position_data"] = build_position_data(trial_df)

    # Tables still saved separately
    for t in ["non_initiated_sequences", "non_initiated_odor1_attempts", "non_initiated_FA"]:
        f = results_dir / f"{t}.csv"
        results[t] = pd.read_csv(f) if f.exists() else pd.DataFrame()

    # Attach manifest and summary
    results["manifest"] = manifest
    results["summary"] = summary
    results["results_dir"] = str(results_dir)

    return results


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
        print("\n--- Decision Accuracy ---")
        metrics['decision_accuracy'] = decision_accuracy_session(results)
        print("\n--- Decision Accuracy by Odor ---")
        accuracy_by_odor = decision_accuracy_by_odor_session(results)
        metrics['decision_accuracy_by_odor'] = accuracy_by_odor.to_dict() if len(accuracy_by_odor) > 0 else {}
        print("\n--- Global Choice Accuracy ---")
        metrics['global_choice_accuracy'] = global_choice_accuracy_session(results)
        print("\n--- Premature Response Rate ---")
        metrics['premature_response_rate'] = premature_response_rate_session(results)
        print("\n--- Response-Contingent False Alarm Rate ---")
        metrics['response_contingent_FA_rate'] = response_contingent_FA_rate_session(results)
        print("\n--- Global False Alarm Rate ---")
        metrics['global_FA_rate'] = global_FA_rate_session(results)
        print("\n--- FA Odor Bias ---")
        fa_odor = FA_odor_bias_session(results)
        metrics['FA_odor_bias'] = fa_odor.to_dict() if hasattr(fa_odor, 'to_dict') else fa_odor
        print("\n--- FA Position Bias ---")
        fa_pos = FA_position_bias_session(results)
        metrics['FA_position_bias'] = fa_pos.to_dict() if hasattr(fa_pos, 'to_dict') else fa_pos
        print("\n--- Sequence Completion Rate ---")
        metrics['sequence_completion_rate'] = sequence_completion_rate_session(results)
        print("\n--- Odor Abortion Rate ---")
        odor_ab = odorx_abortion_rate_session(results)
        metrics['odorx_abortion_rate'] = odor_ab.to_dict() if hasattr(odor_ab, 'to_dict') else odor_ab
        print("\n--- Hidden Rule Performance ---")
        metrics['hidden_rule_performance'] = hidden_rule_performance_session(results)
        print("\n--- Hidden Rule Detection Rate ---")
        metrics['hidden_rule_detection_rate'] = hidden_rule_detection_rate_session(results)
        print("\n--- Hidden Rule Performance/Detection by Odor ---")
        metrics['hidden_rule_by_odor'] = hidden_rule_counts_by_odor_session(results)
        print("\n--- Choice Timeout Rate ---")
        metrics['choice_timeout_rate'] = choice_timeout_rate_session(results)
        print("\n--- Average Sampling Time per Odor (Completed) ---")
        avg_samp_odor = avg_sampling_time_odor_x_session(results)
        metrics['avg_sampling_time_odor_x'] = avg_samp_odor.to_dict() if hasattr(avg_samp_odor, 'to_dict') else avg_samp_odor
        print("\n--- Average Sampling Time (Completed Sequences) ---")
        metrics['avg_sampling_time_completed_sequence'] = avg_sampling_time_completed_sequence_session(results)
        print("\n--- Average Sampling Time (Aborted Sequences) ---")
        metrics['avg_sampling_time_aborted_sequence'] = avg_sampling_time_aborted_sequence_session(results)
        print("\n--- Average Sampling Time (Initiation Abortions) ---")
        metrics['avg_sampling_time_initiation_abortion'] = avg_sampling_time_initiation_abortion(results)
        print("\n--- Abortion Rate by Position ---")
        abrt_pos = abortion_rate_positionX_session(results)
        metrics['abortion_rate_positionX'] = abrt_pos.to_dict() if hasattr(abrt_pos, 'to_dict') else abrt_pos
        print("\n--- Average Response Time ---")
        metrics['avg_response_time'] = avg_response_time_session(results)
        print("\n--- FA Average Response Times ---")
        metrics['FA_avg_response_times'] = FA_avg_response_times_session(results)
        print("\n--- Response Rate ---")
        metrics['response_rate'] = response_rate_session(results)
        print("\n--- Manual vs Auto Stop Preference ---")
        metrics['manual_vs_auto_stop_preference'] = manual_vs_auto_stop_preference_session(results)
        print("\n--- Non-Initiated FA Rate ---")
        metrics['non_initiated_FA_rate'] = non_initiated_FA_rate(results)
        print("\n--- Non-Initiation Odor Bias ---")
        noninit_odor = non_initiation_odor_bias(results)
        metrics['non_initiation_odor_bias'] = noninit_odor.to_dict() if hasattr(noninit_odor, 'to_dict') else noninit_odor
        print("\n--- Odor Initiation Bias ---")
        odor_init = odor_initiation_bias_session(results)
        metrics['odor_initiation_bias'] = odor_init.to_dict() if hasattr(odor_init, 'to_dict') else odor_init
        print("\n--- FA Abortion Stats ---")
        fa_ab_stats = fa_abortion_stats_session(results, return_df=True)
        if fa_ab_stats is not None:
            metrics['fa_abortion_stats'] = {
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
        else:
            metrics['fa_abortion_stats'] = None
        print("\n--- FA Port Ratio by Odor ---")
        # Calculate with non-initiated FAs included
        fa_port_ratio_with = fa_port_ratio_by_odor(results, include_non_initiated=True)
        # Calculate without non-initiated FAs
        fa_port_ratio_without = fa_port_ratio_by_odor(results, include_non_initiated=False)
        
        metrics['fa_port_ratio_by_odor'] = {
            'with_non_initiated': {
                'by_odor': fa_port_ratio_with['by_odor'].to_dict() if hasattr(fa_port_ratio_with['by_odor'], 'to_dict') else fa_port_ratio_with['by_odor'],
                'counts': fa_port_ratio_with['counts'],
                'total_fa_by_odor': fa_port_ratio_with['total_fa_by_odor'],
            },
            'without_non_initiated': {
                'by_odor': fa_port_ratio_without['by_odor'].to_dict() if hasattr(fa_port_ratio_without['by_odor'], 'to_dict') else fa_port_ratio_without['by_odor'],
                'counts': fa_port_ratio_without['counts'],
                'total_fa_by_odor': fa_port_ratio_without['total_fa_by_odor'],
            }
        }

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

def pool_results_dicts(results_dicts):
    """
    Given a list of results dicts (from load_session_results), pool all DataFrames by key.
    Returns a single results dict with concatenated DataFrames and merged manifest/summary.
    """
    pooled = {}
    # Pool DataFrames
    all_keys = set()
    for r in results_dicts:
        all_keys.update(r.keys())
    for key in all_keys:
        dfs = [r[key] for r in results_dicts if key in r and isinstance(r[key], pd.DataFrame)]
        if dfs:
            pooled[key] = pd.concat(dfs, ignore_index=True)
        else:
            pooled[key] = results_dicts[0].get(key, None)
    # Merge manifest/summary for merged info

    def get_subjid(r):
        sess = r.get("manifest", {}).get("session", {})
        return str(sess.get("subject_id") or sess.get("subjid") or "")

    def get_date(r):
        sess = r.get("manifest", {}).get("session", {})
        return str(sess.get("date") or sess.get("session_date") or "")

    subjids = sorted({get_subjid(r) for r in results_dicts if get_subjid(r)})
    dates = sorted({get_date(r) for r in results_dicts if get_date(r)})

    protocol = None
    for r in results_dicts:
        runs = r.get("summary", {}).get("session", {}).get("runs", [])
        if runs and "stage" in runs[0]:
            protocol = runs[0]["stage"].get("stage_name", None)
            if protocol:
                break
    pooled["manifest"] = {
        "merged_subjects": subjids,
        "merged_dates": dates,
        "protocol": protocol
    }
    pooled["summary"] = {
        "merged_subjects": subjids,
        "merged_dates": dates,
        "protocol": protocol
    }
    return pooled

def save_merged_metrics_txt(metrics, header, txt_path, pretty_print_str=None):
    """
    Save merged metrics to a txt file with a header and formatted output.
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        if pretty_print_str is not None:
            f.write(pretty_print_str)
        else: 
            for k, v in metrics.items():
                if isinstance(v, (tuple, list)) and len(v) == 3 and isinstance(v[0], (int, float)):
                    # Standard metric: numerator, denominator, value
                    num, denom, val = v
                    f.write(f"{k.replace('_',' ').title()}: {num}/{denom} = {val:.3f}\n")
                elif isinstance(v, dict) and "bias" in v and "n_fa" in v and "n_ab" in v:
                    # FA Odor Bias
                    f.write(f"{k.replace('_',' ').title()}:\n")
                    for od in v["bias"]:
                        bias = v["bias"][od]
                        n_fa = v["n_fa"].get(od, 0)
                        n_ab = v["n_ab"].get(od, 0)
                        f.write(f"  {od}: {n_fa}/{n_ab} FA, Bias: {bias:.3f}\n")
                elif isinstance(v, dict):
                    f.write(f"{k.replace('_',' ').title()}:\n")
                    for subk, subv in v.items():
                        if isinstance(subv, float):
                            f.write(f"  {subk}: {subv:.3f}\n")
                        else:
                            f.write(f"  {subk}: {subv}\n")
                elif isinstance(v, pd.Series):
                    f.write(f"{k.replace('_',' ').title()}:\n")
                    for idx, val in v.items():
                        f.write(f"  {idx}: {val:.3f}\n")
                elif isinstance(v, float):
                    f.write(f"{k.replace('_',' ').title()}: {v:.3f}\n")
                else:
                    f.write(f"{k.replace('_',' ').title()}: {v}\n")

def merged_results_output_dir(subjids, dates, protocol):
    """
    Determine the output directory for merged results based on subjids, dates, and protocol.
    """
    derivatives_dir = get_derivatives_root()
    subjids = sorted(set(str(s) for s in subjids))
    dates = sorted(set(str(d) for d in dates))
    if len(subjids) == 1:
        subj_dir = derivatives.subject_dir(subjids[0])
        merged_dir = subj_dir / "merged_results"
    else:
        merged_dir = derivatives_dir / "merged"
        merged_dir = merged_dir / ("protocol_merged" if protocol else "merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    return merged_dir

def merged_metrics_filename(subjids, dates, protocol):
    """
    Construct merged metrics filename based on subjids, dates, and protocol.
    """
    subjids = sorted(set(str(s) for s in subjids))
    dates = sorted(set(str(d) for d in dates))
    n_dates = len(dates)
    if len(subjids) == 1:
        proto = protocol if protocol else "all"
        fname = f"merged_{proto}_{n_dates}_dates"
    else:
        subj_str = "_".join(subjids)
        fname = f"merged_subjids_{subj_str}_{n_dates}_dates"
    return fname

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

# ================== Behavioral Metrics Functions =================================================================================================================================

# ================== Metric cores: f(frame) -> value ==============================
#
# restructure_2 Phase 4a, decision D0. Every metric gets a pure core taking a
# *trial frame* plus a thin `_session(results)` wrapper that prints and keeps
# `run_all_metrics` (and therefore the saved JSON) unchanged.
#
# Rate metrics additionally expose their numerator and denominator
# *contributions* as per-trial Series, because **a rate is not a per-trial
# quantity**. Storing one value per trial and taking a rolling mean gives
# `rewarded / window_size` -- a denominator silently containing timeouts and
# aborts. That is finding 12 of the audit: it is exactly why
# `pred_seq.performance` and `plot_decision_accuracy_rolling_average` disagree
# today. Reducing `num.sum() / den.sum()` over any slice is correct at every
# granularity, and two cumulative sums make a rolling window O(1).


def _aborted_mask(trials):
    """The `df["is_aborted"] == True` mask every metric builds, once."""
    if "is_aborted" in trials.columns:
        return trials["is_aborted"] == True  # noqa: E712
    return pd.Series(False, index=trials.index)


def _flag(trials, column, value):
    """`trials[column] == value` as a boolean Series; all-False when absent."""
    if column in trials.columns:
        return trials[column] == value
    return pd.Series(False, index=trials.index)


def _truthy(trials, column):
    if column in trials.columns:
        return trials[column].apply(_is_truthy).astype(bool)
    return pd.Series(False, index=trials.index)


def _reduce_rate(num, den):
    """(numerator, denominator) contributions -> (n, denom, rate)."""
    n = int(np.asarray(num, dtype=float).sum())
    d = int(np.asarray(den, dtype=float).sum())
    return n, d, (n / d if d > 0 else np.nan)


def _initiated(trials):
    """Denominator "an initiated trial": a non-null global_trial_id, else all rows."""
    if "global_trial_id" in trials.columns:
        return trials["global_trial_id"].notna().astype(int)
    return pd.Series(1, index=trials.index)


def decision_accuracy_contributions(trials):
    rtc = trials["response_time_category"]
    return ((rtc == "rewarded").astype(int),
            rtc.isin(["rewarded", "unrewarded"]).astype(int))


def decision_accuracy(trials):
    """rewarded / (rewarded + unrewarded)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*decision_accuracy_contributions(trials))


def decision_accuracy_session(results):
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty or "response_time_category" not in trials.columns:
        print("Decision Accuracy: no trial_data with response_time_category")
        return 0, 0, np.nan
    n_rew, denom, acc = decision_accuracy(trials)
    print(f"Decision Accuracy: {n_rew}/{denom} = {acc:.3f}")
    return n_rew, denom, acc


def global_choice_accuracy_contributions(trials):
    rtc = trials["response_time_category"]
    # Counts are summed, not or-ed: a trial flagged both ways contributes twice,
    # as it does today.
    return ((rtc == "rewarded").astype(int),
            rtc.isin(["rewarded", "unrewarded"]).astype(int)
            + _flag(trials, "fa_label", "FA_time_in").astype(int))


def global_choice_accuracy(trials):
    """rewarded / (rewarded + unrewarded + FA_time_in)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*global_choice_accuracy_contributions(trials))


def global_choice_accuracy_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Global Choice Accuracy: no trial_data with response_time_category")
        return 0, 0, np.nan
    n_correct, n_total, accuracy = global_choice_accuracy(df)
    n_incorrect = int((df["response_time_category"] == "unrewarded").sum())
    n_fa_time_in = int(_flag(df, "fa_label", "FA_time_in").sum())
    print(f"Global Choice Accuracy: {n_correct}/{n_total} = {accuracy:.3f}")
    print(f"  - Correct choices: {n_correct}")
    print(f"  - Incorrect choices: {n_incorrect}")
    print(f"  - False alarms (FA Time In): {n_fa_time_in}")
    return n_correct, n_total, accuracy

def decision_accuracy_by_odor(trials):
    """Per-odor `decision_accuracy`, plus a `_total` variant including timeouts."""
    if trials.empty or "response_time_category" not in trials.columns or "last_odor" not in trials.columns:
        return pd.DataFrame()

    def extract_odor_letter(odor_str):
        if pd.isna(odor_str):
            return np.nan
        if isinstance(odor_str, str) and odor_str.startswith("Odor"):
            return odor_str.replace("Odor", "")
        return odor_str

    df_local = trials.copy()
    df_local["odor_letter"] = df_local["last_odor"].apply(extract_odor_letter)

    rows = []
    for odor in sorted(df_local["odor_letter"].dropna().unique()):
        odor_trials = df_local[df_local["odor_letter"] == odor]
        n_rew = int((odor_trials["response_time_category"] == "rewarded").sum())
        n_unr = int((odor_trials["response_time_category"] == "unrewarded").sum())
        n_tmo = int((odor_trials["response_time_category"] == "timeout_delayed").sum())
        denom_ab = n_rew + n_unr
        denom_total = denom_ab + n_tmo
        rows.append({
            'odor': odor,
            'rewarded': n_rew,
            'unrewarded': n_unr,
            'timeout': n_tmo,
            'decision_accuracy_ab': n_rew / denom_ab if denom_ab > 0 else np.nan,
            'decision_accuracy_total': n_rew / denom_total if denom_total > 0 else np.nan,
            'denominator_ab': denom_ab,
            'denominator_total': denom_total,
        })

    return pd.DataFrame(rows).set_index('odor').sort_index()


def decision_accuracy_by_odor_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "last_odor" not in df.columns:
        print("Decision Accuracy by Odor: no trial_data with response_time_category/last_odor")
        return pd.DataFrame()
    out = decision_accuracy_by_odor(df)

    def _fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "nan"

    print("Decision Accuracy by Odor:")
    for odor, r in out.iterrows():
        # int(): a row Series takes one common dtype, so these counts arrive as
        # floats and would render as "65.0 rewarded" in metrics_*.txt.
        n_rew, n_unr, n_tmo = int(r['rewarded']), int(r['unrewarded']), int(r['timeout'])
        d_ab, d_total = int(r['denominator_ab']), int(r['denominator_total'])
        print(f"  Odor {odor}: {n_rew} rewarded, {n_unr} unrewarded, {n_tmo} timeout")
        print(f"       Decision Accuracy AB: {n_rew}/{d_ab} = {_fmt(r['decision_accuracy_ab'])}, "
              f"Total: {n_rew}/{d_total} = {_fmt(r['decision_accuracy_total'])}")
    return out

def premature_response_rate_contributions(trials):
    ab = _aborted_mask(trials)
    return ((ab & _flag(trials, "fa_label", "FA_time_in")).astype(int),
            ab.astype(int))


def premature_response_rate(trials):
    """FA_time_in among aborted / n aborted."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*premature_response_rate_contributions(trials))


def premature_response_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Premature Response Rate: no trial_data")
        return 0, 0, np.nan
    n_fa, n_total, rate = premature_response_rate(df)
    if n_total == 0:
        print("Premature Response Rate: no aborted trials")
        return 0, 0, np.nan
    print(f"Premature Response Rate: {n_fa}/{n_total} = {rate:.3f}")
    return n_fa, n_total, rate

def response_contingent_FA_rate_contributions(trials):
    num = (_aborted_mask(trials) & _flag(trials, "fa_label", "FA_time_in")).astype(int)
    rtc = trials["response_time_category"]
    return num, num + rtc.isin(["rewarded", "unrewarded"]).astype(int)


def response_contingent_FA_rate(trials):
    """FA_time_in / (FA_time_in + rewarded + unrewarded)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*response_contingent_FA_rate_contributions(trials))


def response_contingent_FA_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Response-Contingent False Alarm Rate: missing trial_data/response_time_category")
        return 0, 0, np.nan
    n_fa, denom, rate = response_contingent_FA_rate(df)
    print(f"Response-Contingent False Alarm Rate: {n_fa}/{denom} = {rate:.3f}")
    return n_fa, denom, rate

def global_FA_rate_contributions(trials):
    return (_flag(trials, "fa_label", "FA_time_in").astype(int), _initiated(trials))


def global_FA_rate(trials):
    """FA_time_in / n initiated."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*global_FA_rate_contributions(trials))


def global_FA_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Global False Alarm Rate: no trial_data")
        return 0, 0, np.nan
    n_fa, n_ini, rate = global_FA_rate(df)
    print(f"Global False Alarm Rate: {n_fa}/{n_ini} = {rate:.3f}")
    return n_fa, n_ini, rate

def FA_odor_bias(trials, *, reference=None):
    """Per-odor FA rate normalised by a baseline FA rate.

    `bias[odor] = (n_fa@odor / n_ab@odor) / reference`, with `reference`
    defaulting to this frame's own `total_fa / total_ab`. Passing it explicitly
    is what lets a rolling call keep a fixed session baseline instead of
    normalising each window by itself -- the plotters' `baseline="session"` vs
    `"window"` option, without any metric math moving back into `visualization/`.
    """
    empty = {'bias': {}, 'n_fa': {}, 'n_ab': {}, 'total_fa': 0, 'total_ab': 0}
    if trials.empty or "fa_label" not in trials.columns:
        return empty
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return empty
    aborted = trials[_aborted_mask(trials)]
    if aborted.empty:
        return empty

    fa_mask = aborted["fa_label"] == "FA_time_in"
    total_fa = int(fa_mask.sum())
    total_ab = len(aborted)
    ref = reference if reference is not None else (
        (total_fa / total_ab) if total_ab > 0 and total_fa > 0 else None)

    bias, n_fa, n_ab = {}, {}, {}
    for od in sorted(aborted[odor_col].dropna().unique()):
        at_od = aborted[odor_col] == od
        n_fa_od = int((fa_mask & at_od).sum())
        n_ab_od = int(at_od.sum())
        n_fa[od], n_ab[od] = n_fa_od, n_ab_od
        bias[od] = (n_fa_od / n_ab_od) / ref if n_ab_od > 0 and ref else np.nan
    return {'bias': bias, 'n_fa': n_fa, 'n_ab': n_ab,
            'total_fa': total_fa, 'total_ab': total_ab}


def FA_odor_bias_session(results):
    print("FA Odor Bias for FA Time In:")
    out = FA_odor_bias(results.get("trial_data", pd.DataFrame()))
    for od, bias in out['bias'].items():
        print(f"{od}: {out['n_fa'][od]}/{out['n_ab'][od]} FA, Bias: {bias:.3f}")
    return out

def FA_position_bias(trials, *, reference=None, with_counts=False):
    """`FA_odor_bias` by `last_odor_position`. See it for the `reference` rule."""
    if trials.empty or "fa_label" not in trials.columns:
        return ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    position_col = "last_odor_position" if "last_odor_position" in trials.columns else "last_event_index"
    if position_col not in trials.columns:
        return ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    aborted = trials[_aborted_mask(trials)]
    if aborted.empty:
        return ({}, {}, {}) if with_counts else pd.Series(dtype=float)

    fa_mask = aborted["fa_label"] == "FA_time_in"
    total_fa = int(fa_mask.sum())
    total_ab = len(aborted)
    ref = reference if reference is not None else (
        (total_fa / total_ab) if total_ab > 0 and total_fa > 0 else None)

    bias, n_fa, n_ab = {}, {}, {}
    for pos in sorted(aborted[position_col].dropna().unique()):
        at_pos = aborted[position_col] == pos
        n_fa_pos = int((fa_mask & at_pos).sum())
        n_ab_pos = int(at_pos.sum())
        key = int(pos) + 1 if position_col == "last_event_index" else int(pos)
        n_fa[key], n_ab[key] = n_fa_pos, n_ab_pos
        bias[key] = (n_fa_pos / n_ab_pos) / ref if n_ab_pos > 0 and ref else np.nan
    if with_counts:
        return bias, n_fa, n_ab
    return pd.Series(bias).sort_index()


def FA_position_bias_session(results):
    print("FA Position Bias for FA Time In:")
    trials = results.get("trial_data", pd.DataFrame())
    parts = FA_position_bias(trials, with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    bias, n_fa, n_ab = parts
    for pos in sorted(bias):
        print(f"Position {pos}: {n_fa[pos]}/{n_ab[pos]} FA, Bias: {bias[pos]:.3f}")
    return pd.Series(bias).sort_index()

def sequence_completion_rate_contributions(trials):
    return ((~_aborted_mask(trials)).astype(int), _initiated(trials))


def sequence_completion_rate(trials):
    """completed / initiated."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*sequence_completion_rate_contributions(trials))


def sequence_completion_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Sequence Completion Rate: no trial_data")
        return 0, 0, np.nan
    n_completed, denom, rate = sequence_completion_rate(df)
    print(f"Sequence Completion Rate: {n_completed}/{denom} = {rate:.3f}")
    return n_completed, denom, rate

# ================== Metric cores: tier 2, grouped inside a position blob =========
#
# restructure_2 Phase 4a, decision D0 tier 2. These eight metrics group by odor
# or by position -- keys that used to live inside a per-trial JSON blob, so each
# one parsed a blob inline. They now read `position_data`, the long
# `trial x position` frame `load_session_results` derives (see
# `metric_analysis/frames.build_position_data`).
#
# **Each filters on the provenance flag for the blob it read before.** The three
# blobs do NOT carry the same positions: `position_valve_times` records valve
# activations whose poke registered as ~0 ms, which `position_poke_times` and
# `presentations` both drop. On the fixture set that is 34 rows on sub-053, 19 on
# sub-057, 17 on sub-059, 7 on sub-048, 4 on sub-040 20251124 and 1 each on
# sub-046 and sub-056. Reading `position_data` unfiltered would silently widen
# every sampling metric.
#
# Summation style is reproduced deliberately, not incidentally: the two pooled
# `avg_sampling_time_*` metrics accumulate `total += x` left to right, while
# `avg_sampling_time_odor_x` calls `np.mean` on a per-odor list, which is
# pairwise. The two disagree in the last ULP over a few hundred values -- enough
# to move the metrics md5, and invisible in any printed output.


def _position_rows(position_data, blob, *, aborted=None):
    """Rows of `position_data` that came from `blob`, optionally by outcome.

    Returns None when the frame is absent or unusable, which every caller treats
    as "no positions" -- the same answer today's inline blob walk gives.
    """
    if position_data is None or len(position_data) == 0:
        return None
    if blob not in position_data.columns:
        return None
    rows = position_data[position_data[blob].astype(bool)]
    if aborted is not None:
        rows = rows[rows["is_aborted"].astype(bool) == aborted]
    return rows


def _sequential_mean(values):
    """Mean by left-to-right accumulation.

    Matches the `total = 0.0; total += x` loops this replaces. `np.mean` and
    `Series.mean` sum pairwise and differ in the last ULP over a few hundred
    values, which moves the metrics fingerprint.
    """
    total = 0.0
    n = 0
    for v in values:
        total += v
        n += 1
    return total / n if n > 0 else np.nan


def presentation_counts_by_odor(position_data):
    """`{odor_name: n presentations}` -- the denominator of `odorx_abortion_rate`.

    Counts `presentations` rows only, so the valve-only positions never enter.
    """
    rows = _position_rows(position_data, "in_presentations")
    if rows is None or rows.empty:
        return {}
    rows = rows[rows["odor_name"].notna()]
    return {od: int(n) for od, n in rows.groupby("odor_name").size().items()}


def odorx_abortion_rate(trials, position_data, *, with_counts=False):
    """aborts@odor / presentations@odor."""
    empty = ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    if trials.empty or "presentations" not in trials.columns:
        return empty
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return empty

    aborted = trials[_aborted_mask(trials)]
    abortions = aborted[odor_col].dropna().value_counts().to_dict()
    presentations = presentation_counts_by_odor(position_data)

    all_odors = set(presentations.keys()).union(abortions.keys())
    rates = {}
    for od in sorted(all_odors):
        n_pres = presentations.get(od, 0)
        rates[od] = abortions.get(od, 0) / n_pres if n_pres > 0 else np.nan
    if with_counts:
        return rates, abortions, presentations
    return pd.Series(rates, dtype=float).sort_index()


def odorx_abortion_rate_session(results):
    parts = odorx_abortion_rate(results.get("trial_data", pd.DataFrame()),
                                results.get("position_data"), with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    rates, abortions, presentations = parts
    for od in sorted(rates):
        print(f"{od}: {abortions.get(od, 0)}/{presentations.get(od, 0)} abortions, "
              f"Rate: {rates[od]:.3f}")
    return pd.Series(rates, dtype=float).sort_index()

def hidden_rule_performance_contributions(trials):
    return (((_truthy(trials, "hidden_rule_success")
              & _flag(trials, "response_time_category", "rewarded")).astype(int)),
            _truthy(trials, "hit_hidden_rule").astype(int))


def hidden_rule_performance(trials):
    """(HR success & rewarded) / hit_hidden_rule."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*hidden_rule_performance_contributions(trials))


def hidden_rule_performance_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Performance: no trial_data")
        return 0, 0, np.nan
    n_hr_rewarded, denom, rate = hidden_rule_performance(df)
    print(f"Hidden Rule Performance: {n_hr_rewarded}/{denom} = {rate:.3f}")
    return n_hr_rewarded, denom, rate

def hidden_rule_detection_rate_contributions(trials):
    return ((((~_aborted_mask(trials)) & _truthy(trials, "hidden_rule_success")).astype(int)),
            _truthy(trials, "hit_hidden_rule").astype(int))


def hidden_rule_detection_rate(trials):
    """(not aborted & HR success) / hit_hidden_rule."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*hidden_rule_detection_rate_contributions(trials))


def hidden_rule_detection_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Detection Rate: no trial_data")
        return 0, 0, np.nan
    n_hr_completed, denom, rate = hidden_rule_detection_rate(df)
    print(f"Hidden Rule Detection Rate: {n_hr_completed}/{denom} = {rate:.3f}")
    return n_hr_completed, denom, rate


def _extract_hr_config(results):
    """Return (hr_odors, hr_positions) from session metadata or results dict if available."""
    # Prefer values already attached to results by classification
    hr_odors = results.get("hidden_rule_odors") or []
    if isinstance(hr_odors, str):
        hr_odors = [hr_odors]

    hr_positions = results.get("hidden_rule_positions") or []
    if isinstance(hr_positions, (int, float)):
        hr_positions = [hr_positions]

    manifest = results.get("manifest", {}) or {}
    manifest_params = manifest.get("params", {}) if isinstance(manifest, dict) else {}
    manifest_session = manifest.get("session", {}) if isinstance(manifest, dict) else {}

    # Fallback to summary params
    summary = results.get("summary", {}) or {}
    params = summary.get("params", {}) if isinstance(summary, dict) else {}
    if not hr_odors:
        hr_odors = (
            params.get("hidden_rule_odors")
            or params.get("hiddenrule_odors")
            or manifest_params.get("hidden_rule_odors")
            or manifest_params.get("hiddenrule_odors")
            or manifest_session.get("hidden_rule_odors")
            or manifest.get("hidden_rule_odors")
            or []
        )
        if isinstance(hr_odors, str):
            hr_odors = [hr_odors]
    hr_odors = [str(o) for o in hr_odors if o]

    if not hr_positions:
        hr_positions = (
            params.get("hidden_rule_positions")
            or params.get("hiddenrule_positions")
            or manifest_params.get("hidden_rule_positions")
            or manifest_params.get("hiddenrule_positions")
            or manifest_session.get("hidden_rule_positions")
            or manifest.get("hidden_rule_positions")
            or []
        )
        if isinstance(hr_positions, (int, float)):
            hr_positions = [hr_positions]

    hr_pos_clean = []
    hr_iter = hr_positions if isinstance(hr_positions, (list, tuple)) else []
    for pos in hr_iter:
        try:
            hr_pos_clean.append(int(pos))
        except Exception:
            continue
    return hr_odors, hr_pos_clean


def _is_truthy(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        try:
            return not math.isnan(val) and val != 0
        except Exception:
            return val != 0
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _infer_hr_odors_from_row(row, hr_odors, hr_positions):
    """Best-effort identification of HR odor(s) for a trial row. Returns list of candidates."""

    def _parse_seq(val):
        seq = parse_json_column(val)
        if isinstance(seq, (list, tuple)):
            return list(seq)
        if isinstance(seq, str):
            try:
                return list(ast.literal_eval(seq)) if seq.strip() else []
            except Exception:
                return [seq]
        return []

    seq_fields = ["odor_sequence", "odor_sequence_full", "odor_sequence_list"]
    seq = []
    for key in seq_fields:
        if key in row:
            seq = _parse_seq(row.get(key))
            if seq:
                break

    # Per-row hidden rule positions, if present
    hr_pos_row = _parse_seq(row.get("hidden_rule_positions")) if "hidden_rule_positions" in row else []
    hr_pos_row_int = []
    for p in hr_pos_row if isinstance(hr_pos_row, (list, tuple)) else []:
        try:
            hr_pos_row_int.append(int(p))
        except Exception:
            continue

    positions_to_use = hr_pos_row_int or hr_positions

    found = []

    # Try using positions to pick odor from sequence
    if seq and positions_to_use:
        for pos in positions_to_use:
            idx = pos - 1
            if 0 <= idx < len(seq):
                candidate = seq[idx]
                if candidate is not None:
                    found.append(candidate)

    # If we have HR odor list, look for unique match in sequence
    if not found and seq and hr_odors:
        matches = [o for o in seq if o in hr_odors]
        if matches:
            found.extend(matches)

    # Hidden-rule-specific columns
    for key in ["hidden_rule_odor", "hidden_rule_odors"]:
        if key in row:
            vals = _parse_seq(row.get(key))
            if vals:
                found.extend(vals)

    # Fallback: last odor name
    for key in ["last_odor_name", "last_odor"]:
        if key in row:
            val = row.get(key)
            if val:
                found.append(val)

    # Normalize and deduplicate while preserving order
    out = []
    seen = set()
    for od in found:
        if od is None:
            continue
        s = str(od)
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out or ["Unknown"]


def _fmt_rate(val):
    return f"{val:.3f}" if isinstance(val, (int, float, np.floating)) and not np.isnan(val) else "nan"


def hidden_rule_counts_by_odor(trials, position_data, hr_odors, hr_positions):
    """
    Aggregate HR trials by odor across outcome categories to support per-odor performance/detection.
    Returns a dict with hr_odors, hr_positions, and per-odor counts plus rates.

    `hr_odors` / `hr_positions` are session *metadata*, not trial data, so the
    core takes them as arguments and `_extract_hr_config` stays in the wrapper.
    """
    df = trials
    if df.empty:
        return {"hr_odors": [], "hr_positions": [], "by_odor": {}}

    hr_set = set(hr_odors)
    counts = defaultdict(lambda: defaultdict(int))

    # Pre-seed known HR odors to ensure they appear even if zero counts
    for od in hr_odors:
        _ = counts[od]

    seen_odors = set(hr_odors)

    def _add_counts(mask: pd.Series, label: str):
        subset = df[mask] if isinstance(mask, pd.Series) else pd.DataFrame()
        if subset.empty:
            return
        for _, row in subset.iterrows():
            odors = _infer_hr_odors_from_row(row, hr_odors, hr_positions)
            for od in odors:
                if od not in hr_set:
                    continue
                seen_odors.add(od)
                counts[od][label] += 1

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    success_mask = df["hidden_rule_success"].apply(_is_truthy) if "hidden_rule_success" in df.columns else pd.Series(False, index=df.index)
    hit_mask = df["hit_hidden_rule"].apply(_is_truthy) if "hit_hidden_rule" in df.columns else pd.Series(False, index=df.index)

    # Completed HR trials by outcome (only count HR successes)
    if "response_time_category" in df.columns:
        _add_counts((df["response_time_category"] == "rewarded") & success_mask, "rewarded")
        _add_counts((df["response_time_category"] == "unrewarded") & success_mask, "unrewarded")
        _add_counts((df["response_time_category"] == "timeout_delayed") & success_mask, "timeout")

    # Aborted HR trials (any aborted hit)
    _add_counts(aborted_mask & hit_mask, "aborted")

    # Missed HR trials: not aborted and not successful
    _add_counts((~aborted_mask) & (~success_mask), "missed")

    # Total presentations per odor -- the same count `odorx_abortion_rate` uses,
    # restricted to the hidden-rule odors.
    presentations = {od: n for od, n in presentation_counts_by_odor(position_data).items()
                     if od in hr_set}

    by_odor = {}
    for odor in sorted(seen_odors):
        c = counts.get(odor, {})
        rewarded = c.get("rewarded", 0)
        unrewarded = c.get("unrewarded", 0)
        timeout = c.get("timeout", 0)
        missed = c.get("missed", 0)
        aborted = c.get("aborted", 0)

        total_presentations = presentations.get(odor, 0)
        completed_no_timeout = rewarded + unrewarded
        completed_with_timeout = completed_no_timeout + timeout

        performance = rewarded / completed_no_timeout if completed_no_timeout > 0 else np.nan
        detection_rate = completed_no_timeout / total_presentations if total_presentations > 0 else np.nan

        by_odor[odor] = {
            "rewarded": int(rewarded),
            "unrewarded": int(unrewarded),
            "timeout": int(timeout),
            "missed": int(missed),
            "aborted": int(aborted),
            "total_presentations": int(total_presentations),
            "completed_total": int(completed_with_timeout),
            "completed_no_timeout": int(completed_no_timeout),
            "performance": performance,
            "performance_fraction": [int(rewarded), int(completed_no_timeout)],
            "detection_rate": detection_rate,
            "detection_fraction": [int(completed_no_timeout), int(total_presentations)],
        }

    return {
        "hr_odors": sorted(seen_odors),
        "hr_positions": hr_positions,
        "by_odor": by_odor,
    }


def hidden_rule_counts_by_odor_session(results):
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty:
        print("Hidden Rule Counts by Odor: no trial_data")
        return {"hr_odors": [], "hr_positions": [], "by_odor": {}}
    hr_odors, hr_positions = _extract_hr_config(results)
    out = hidden_rule_counts_by_odor(trials, results.get("position_data"),
                                     hr_odors, hr_positions)
    for odor in out["hr_odors"]:
        c = out["by_odor"][odor]
        print(
            f"Hidden Rule Odor {odor}: {c['rewarded']} Rewarded, {c['unrewarded']} Unrewarded, "
            f"{c['timeout']} Timeout, {c['total_presentations']} Total Presentations."
        )
        print(
            f"  HR Odor {odor} Performance: {c['rewarded']}/{c['completed_no_timeout']} = "
            f"{_fmt_rate(c['performance'])}, "
            f"HR Odor {odor} Detection Rate: {c['completed_no_timeout']}/{c['total_presentations']} = "
            f"{_fmt_rate(c['detection_rate'])}"
        )
    return out

def choice_timeout_rate_contributions(trials):
    completed = ~_aborted_mask(trials)
    return ((completed & _flag(trials, "response_time_category", "timeout_delayed")).astype(int),
            completed.astype(int))


def choice_timeout_rate(trials):
    """timeout_delayed / completed."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*choice_timeout_rate_contributions(trials))


def choice_timeout_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Choice Timeout Rate: no trial_data/response_time_category")
        return 0, 0, np.nan
    n_tmo, denom, rate = choice_timeout_rate(df)
    print(f"Choice Timeout Rate: {n_tmo}/{denom} = {rate:.3f}")
    return n_tmo, denom, rate

def avg_sampling_time_odor_x(position_data):
    """Mean `poke_time_ms` per odor over completed trials, from `position_poke_times`."""
    rows = _position_rows(position_data, "in_poke_times", aborted=False)
    if rows is None or rows.empty:
        return pd.Series(dtype=float)
    rows = rows[rows["odor_name"].notna() & rows["poke_time_ms"].notna()]
    if rows.empty:
        return pd.Series(dtype=float)
    # `np.mean` on each group's values rather than `Series.mean()`: this
    # replaces a per-odor Python list passed to `np.mean`, and pandas' reduction
    # kernel can land a ULP away. `rename`s keep the unnamed Series shape the
    # single-dict construction produced.
    avg_times = (rows.groupby("odor_name")["poke_time_ms"]
                 .apply(lambda s: np.mean(s.to_numpy()))
                 .rename(None).rename_axis(None).sort_index())
    return avg_times


def avg_sampling_time_odor_x_session(results):
    avg_times = avg_sampling_time_odor_x(results.get("position_data"))
    for odor, avg_time in avg_times.items():
        print(f"{odor} Average Sampling Time: {avg_time:.2f} ms")
    return avg_times


def avg_sampling_time_completed_sequence(position_data):
    """Pooled mean `poke_time_ms` over completed trials' `position_poke_times`."""
    rows = _position_rows(position_data, "in_poke_times", aborted=False)
    if rows is None or rows.empty:
        return np.nan
    return _sequential_mean(rows.loc[rows["poke_time_ms"].notna(), "poke_time_ms"])


def avg_sampling_time_completed_sequence_session(results):
    if results.get("trial_data", pd.DataFrame()).empty:
        return np.nan
    avg = avg_sampling_time_completed_sequence(results.get("position_data"))
    print(f"Average Sampling Time (Completed Sequences): {avg:.2f} ms")
    return avg


def avg_sampling_time_aborted_sequence(position_data):
    """Pooled mean `poke_time_ms` over aborted trials' `presentations`.

    Excludes the abort event itself -- the entry whose `index_in_trial` equals
    the trial's `last_event_index`. A null `last_event_index` matches nothing, so
    that trial contributes every entry, as it does today.
    """
    rows = _position_rows(position_data, "in_presentations", aborted=True)
    if rows is None or rows.empty:
        return np.nan
    idx = rows["index_in_trial"]
    keep = (idx.notna() & (idx != rows["last_event_index"])
            & rows["poke_time_ms"].notna())
    return _sequential_mean(rows.loc[keep, "poke_time_ms"])


def avg_sampling_time_aborted_sequence_session(results):
    # Silent on an empty trial table and on a session with no aborted trials --
    # both bail before the print today, where a session that aborted but
    # recorded no usable presentation still prints "nan ms".
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty or not _aborted_mask(trials).any():
        return np.nan
    avg = avg_sampling_time_aborted_sequence(results.get("position_data"))
    print(f"Average Sampling Time (Aborted Sequences): {avg:.2f} ms")
    return avg

def avg_sampling_time_initiation_abortion(results):
    def _choose_poke_series(df, columns):
        for col in columns:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    return s
        return pd.Series(dtype=float)
    non_ini = results.get("non_initiated_sequences", pd.DataFrame())
    pos1 = results.get("non_initiated_odor1_attempts", pd.DataFrame())
    base_vals = _choose_poke_series(non_ini, ["continuous_poke_time_ms", "poke_time_ms", "poke_time", "poke_ms"])
    pos1_vals = _choose_poke_series(pos1, ["pos1_poke_time_ms", "attempt_poke_time_ms", "poke_time_ms", "poke_time", "poke_ms"])
    all_vals = pd.concat([base_vals, pos1_vals], ignore_index=True)
    print(f"Average Sampling Time (Initiation Abortions): {all_vals.mean() if not all_vals.empty else np.nan:.2f} ms")
    return all_vals.mean() if not all_vals.empty else np.nan

def abortion_rate_positionX(trials, *, with_counts=False):
    """aborts@position / trials that reached it.

    The denominator is `frames.reached_counts` -- the single definition of
    "reached" for the package (audit Q5), which is why this core needs only the
    trial frame even though it is a per-position metric.
    """
    empty = ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    if trials.empty:
        return empty
    position_col = "last_odor_position" if "last_odor_position" in trials.columns else "last_event_index"
    if position_col not in trials.columns:
        return empty

    aborted = trials[_aborted_mask(trials)]
    abortions = aborted[position_col].dropna().value_counts().to_dict()
    reached = _reached_counts(trials)

    rates = {}
    for pos in sorted(set(list(abortions.keys()) + list(reached.keys()))):
        n_reached = reached.get(pos, 0)
        rates[pos] = abortions.get(pos, 0) / n_reached if n_reached > 0 else np.nan
    if with_counts:
        return rates, abortions, reached
    return pd.Series(rates, dtype=float).sort_index()


def abortion_rate_positionX_session(results):
    parts = abortion_rate_positionX(results.get("trial_data", pd.DataFrame()),
                                    with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    rates, abortions, reached = parts
    for pos in sorted(rates):
        print(f"Position {pos}: {abortions.get(pos, 0)}/{reached.get(pos, 0)} abortions, "
              f"Rate: {rates[pos]:.3f}")
    return pd.Series(rates, dtype=float).sort_index()

def avg_response_time(trials):
    """Mean `response_time_ms` by category, plus the pooled rewarded+unrewarded."""
    if (trials.empty or "response_time_category" not in trials.columns
            or "response_time_ms" not in trials.columns):
        return {}
    vals = pd.to_numeric(trials["response_time_ms"], errors="coerce")
    out = {}
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"),
                       ("Reward Timeout", "timeout_delayed")]:
        s = vals[trials["response_time_category"] == key].dropna()
        out[label] = float(s.mean()) if not s.empty else np.nan
    both = vals[trials["response_time_category"].isin(["rewarded", "unrewarded"])].dropna()
    out["Average Response Time (Rewarded + Unrewarded)"] = float(both.mean()) if not both.empty else np.nan
    return out


def avg_response_time_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "response_time_ms" not in df.columns:
        print("No response time data available.")
        return {}
    out = avg_response_time(df)
    vals = pd.to_numeric(df["response_time_ms"], errors="coerce")
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"),
                       ("Reward Timeout", "timeout_delayed")]:
        avg, n = out[label], len(vals[df["response_time_category"] == key].dropna())
        print(f"{label}: {avg:.1f} ms (n={n})" if not np.isnan(avg) else f"{label}: nan (n={n})")
    key = "Average Response Time (Rewarded + Unrewarded)"
    avg_both = out[key]
    n_both = len(vals[df["response_time_category"].isin(["rewarded", "unrewarded"])].dropna())
    print(f"{key}: {avg_both:.1f} ms (n={n_both})" if not np.isnan(avg_both)
          else f"{key}: nan (n={n_both})")
    return out

def FA_avg_response_times(trials):
    """Mean `fa_latency_ms` per FA subtype."""
    out = {}
    if trials.empty or "fa_label" not in trials.columns or "fa_latency_ms" not in trials.columns:
        return out
    fa_df = trials[trials["fa_label"].notna()]
    for label, pretty in [("FA_time_in", "FA Time In"), ("FA_time_out", "FA Time Out"),
                          ("FA_late", "FA Late")]:
        s = pd.to_numeric(fa_df.loc[fa_df["fa_label"] == label, "fa_latency_ms"], errors="coerce").dropna()
        avg = s.mean() if not s.empty else np.nan
        out[pretty] = float(avg) if not np.isnan(avg) else np.nan
    return out


def FA_avg_response_times_session(results):
    df = results.get("trial_data", pd.DataFrame())
    out = FA_avg_response_times(df)
    if not out:
        return out
    fa_df = df[df["fa_label"].notna()]
    for label, pretty in [("FA_time_in", "FA Time In"), ("FA_time_out", "FA Time Out"),
                          ("FA_late", "FA Late")]:
        n = len(pd.to_numeric(fa_df.loc[fa_df["fa_label"] == label, "fa_latency_ms"],
                              errors="coerce").dropna())
        avg = out[pretty]
        print(f"{pretty}: avg={avg:.1f} ms (n={n})" if not np.isnan(avg) else f"{pretty}: nan (n={n})")
    return out

def response_rate_contributions(trials):
    rtc = trials["response_time_category"]
    num = rtc.isin(["rewarded", "unrewarded"]).astype(int)
    return num, num + (rtc == "timeout_delayed").astype(int)


def response_rate(trials):
    """(rewarded + unrewarded) / (rewarded + unrewarded + timeout)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*response_rate_contributions(trials))


def response_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Response Rate: no trial_data/response_time_category")
        return 0, 0, np.nan
    num, denom, rate = response_rate(df)
    print(f"Response Rate: {num}/{denom} = {rate:.3f}")
    return num, denom, rate

def manual_vs_auto_stop_preference(position_data):
    """Valve durations on completed trials, split at 1000 ms.

    Reads `position_valve_times` only. That blob is a *superset* of the other
    two -- it records positions whose poke registered as ~0 ms -- so this is the
    one metric that would gain rows if the provenance filter were dropped.
    """
    rows = _position_rows(position_data, "in_valve_times", aborted=False)
    if rows is None or rows.empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}
    dur = rows.loc[rows["valve_duration_ms"].notna(), "valve_duration_ms"]
    # `if dur <= 1000 ... elif dur >= 1000`: exactly 1000 ms counts short only.
    short = int((dur <= 1000).sum())
    long = int((dur > 1000).sum())
    return {"short_valve": short, "long_valve": long,
            "ratio": short / long if long > 0 else float('nan')}


def manual_vs_auto_stop_preference_session(results):
    if results.get("trial_data", pd.DataFrame()).empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}
    out = manual_vs_auto_stop_preference(results.get("position_data"))
    print(f"Manual Stops: {out['short_valve']}")
    print(f"Auto Stops: {out['long_valve']}")
    print(f"Manual vs Auto Stop: {out['ratio']:.2f}")
    return out

def non_initiated_FA_rate(results):
    fa_noninit_df = results.get("non_initiated_FA", pd.DataFrame())
    if fa_noninit_df.empty or "fa_label" not in fa_noninit_df.columns:
        return np.nan
    n_fa = (fa_noninit_df["fa_label"] == "FA_time_in").sum()
    print(f"Non-Initiated FA Rate: {n_fa}/{len(fa_noninit_df)} = {n_fa/len(fa_noninit_df) if len(fa_noninit_df)>0 else np.nan:.3f}")
    return n_fa, len(fa_noninit_df), n_fa / len(fa_noninit_df) if len(fa_noninit_df) > 0 else np.nan

def non_initiation_odor_bias(results):
    non_ini = results.get("non_initiated_sequences", pd.DataFrame())
    pos1 = results.get("non_initiated_odor1_attempts", pd.DataFrame())
    trial_df = results.get("trial_data", pd.DataFrame())

    # Only consider first odor attempts in non-initiated tables
    non_ini = non_ini[non_ini["odor_position"] == 1] if "odor_position" in non_ini.columns else non_ini
    pos1 = pos1[pos1["odor_position"] == 1] if "odor_position" in pos1.columns else pos1

    all_non_init = pd.concat([non_ini, pos1], ignore_index=True)

    # Numerator: non-initiated trials with this odor as first odor
    if all_non_init.empty or "odor_name" not in all_non_init.columns:
        count_odors = pd.Series(dtype=int)
    else:
        count_odors = all_non_init["odor_name"].value_counts()

    # Denominator: all trials (initiated + non-initiated) with first odor = odor
    first_odors = []

    # Initiated trials from trial_data presentations
    if not trial_df.empty and "presentations" in trial_df.columns:
        for _, row in trial_df.iterrows():
            pres_list = parse_json_column(row.get("presentations", []))
            if isinstance(pres_list, list):
                for pres in pres_list:
                    if not isinstance(pres, dict):
                        continue
                    pos = pres.get("position")
                    if pos is None and pres.get("index_in_trial") is not None:
                        try:
                            pos = int(pres.get("index_in_trial")) + 1
                        except Exception:
                            pos = None
                    if pos == 1:
                        first_odors.append(pres.get("odor_name"))
                        break

    # Non-initiated (baseline and pos1)
    for df in [non_ini, pos1]:
        if not df.empty and "odor_name" in df.columns:
            first_odors.extend(df["odor_name"].dropna().tolist())

    total_first_odors = pd.Series(first_odors).value_counts()

    # Global rates for normalization
    total_noninit = len(all_non_init)
    total_trials = int(total_first_odors.sum()) if not total_first_odors.empty else 0
    global_rate = total_noninit / total_trials if total_trials > 0 else np.nan

    bias = {}
    for od in sorted(total_first_odors.index):
        n_noninit = count_odors.get(od, 0)
        n_total = total_first_odors.get(od, 0)
        if n_total > 0 and global_rate > 0:
            bias[od] = (n_noninit / n_total) / global_rate
        else:
            bias[od] = np.nan
        print(f"{od}: {n_noninit}/{n_total} non-initiated, Bias: {bias[od]:.3f}")

    return pd.Series(bias).sort_index()

def odor_initiation_bias(trials, *, reference=None, with_counts=False):
    """Per-odor initiation-abortion share / the overall share. See `FA_odor_bias`."""
    empty = ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    if trials.empty or "abortion_type" not in trials.columns:
        return empty
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return empty
    aborted = trials[_aborted_mask(trials)]
    if aborted.empty:
        return empty

    init_mask = aborted["abortion_type"] == "initiation_abortion"
    total_init = int(init_mask.sum())
    total_ab = len(aborted)
    ref = reference if reference is not None else (
        (total_init / total_ab) if total_ab > 0 and total_init > 0 else None)

    bias, n_init, n_ab = {}, {}, {}
    for od in sorted(aborted[odor_col].dropna().unique()):
        at_od = aborted[odor_col] == od
        n_init_od = int((at_od & init_mask).sum())
        n_ab_od = int(at_od.sum())
        n_init[od], n_ab[od] = n_init_od, n_ab_od
        bias[od] = (n_init_od / n_ab_od) / ref if n_ab_od > 0 and ref else np.nan
    if with_counts:
        return bias, n_init, n_ab
    return pd.Series(bias).sort_index()


def odor_initiation_bias_session(results):
    parts = odor_initiation_bias(results.get("trial_data", pd.DataFrame()), with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    bias, n_init, n_ab = parts
    for od in sorted(bias):
        print(f"{od}: {n_init[od]}/{n_ab[od]} initiation abortions, Bias: {bias[od]:.3f}")
    return pd.Series(bias).sort_index()

def _fa_abortion_frames_missing(trials):
    """The guard `fa_abortion_stats` fails on, or None. Message is the caller's."""
    if trials.empty or "fa_label" not in trials.columns:
        return "No FA abortion data available."
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return "No FA abortion data available (missing odor column)."
    if "last_odor_position" not in trials.columns:
        return "No FA abortion data available (missing last_odor_position)."
    if not _aborted_mask(trials).any():
        return "No aborted trials found."
    return None


def fa_abortion_stats(trials):
    """FA abortion breakdown by odor / position / odor x position.

    Returns three DataFrames, empty when the frame lacks what they need. Values
    are pre-formatted strings (`"3/10 (0.30)"`) -- the audit's finding 3 wants
    them numeric, which is a 4b/`summary.py` change, not a 4a one.
    """
    df = trials
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if _fa_abortion_frames_missing(df) is not None:
        return empty

    odor_col = "last_odor_name" if "last_odor_name" in df.columns else "last_odor"
    pos_col = "last_odor_position"

    aborted_all = df[_aborted_mask(df)]
    allowed_fa = {"FA_time_in", "FA_time_out", "FA_late"}

    subtype_labels = [
        ("FA_time_in", "FA Time In"),
        ("FA_time_out", "FA Time Out"),
        ("FA_late", "FA Late"),
    ]

    # Odor+Position table
    rows = []
    odors = sorted(aborted_all[odor_col].dropna().unique())
    positions = sorted(aborted_all[pos_col].dropna().unique())
    for odor in odors:
        for pos in positions:
            sub_all = aborted_all[(aborted_all[odor_col] == odor) & (aborted_all[pos_col] == pos)]
            if sub_all.empty:
                continue
            sub_fa = sub_all[sub_all["fa_label"].isin(allowed_fa)]
            n_total = len(sub_all)
            fa_labels = sub_fa["fa_label"].astype(str)
            row = {
                "Odor": odor,
                "Position": pos,
                "Total Abortions": n_total,
            }
            n_fa = len(sub_fa)
            row["FA Abortion Rate"] = f"{n_fa}/{n_total} ({n_fa/n_total:.2f})"
            for subtype, pretty in subtype_labels:
                count = (fa_labels == subtype).sum()
                row[pretty] = f"{count} ({count/n_total:.2f})"
            rows.append(row)
    df_out = pd.DataFrame(rows)

    # Per-odor table
    odor_rows = []
    for odor in odors:
        sub_all = aborted_all[aborted_all[odor_col] == odor]
        if sub_all.empty:
            continue
        sub_fa = sub_all[sub_all["fa_label"].isin(allowed_fa)]
        n_total = len(sub_all)
        fa_labels = sub_fa["fa_label"].astype(str)
        row = {
            "Odor": odor,
            "Total Abortions": n_total,
        }
        n_fa = len(sub_fa)
        row["FA Abortion Rate"] = f"{n_fa}/{n_total} ({n_fa/n_total:.2f})"
        for subtype, pretty in subtype_labels:
            count = (fa_labels == subtype).sum()
            row[pretty] = f"{count} ({count/n_total:.2f})"
        odor_rows.append(row)
    df_odor = pd.DataFrame(odor_rows)

    # Compute reached counts per position (denominator for overall abortion rate)
    reached = _reached_counts(df)

    # Per-position table (add overall abortion rate using reached counts)
    pos_rows = []
    for pos in positions:
        sub_all = aborted_all[aborted_all[pos_col] == pos]
        if sub_all.empty:
            continue
        sub_fa = sub_all[sub_all["fa_label"].isin(allowed_fa)]
        n_total = len(sub_all)
        fa_labels = sub_fa["fa_label"].astype(str)
        reached_pos = reached.get(int(pos), 0)
        rate_val = (n_total / reached_pos) if reached_pos > 0 else np.nan
        rate_str = f"{n_total}/{reached_pos} ({rate_val:.2f})" if reached_pos > 0 else "N/A"

        row = {
            "Position": pos,
            "Total Abortions": n_total,
            "Reached Trials": reached_pos,
            "Abortion Rate": rate_str,
            "Abortion Rate Value": rate_val,
        }
        n_fa = len(sub_fa)
        row["FA Abortion Rate"] = f"{n_fa}/{n_total} ({n_fa/n_total:.2f})"
        for subtype, pretty in subtype_labels:
            count = (fa_labels == subtype).sum()
            row[pretty] = f"{count} ({count/n_total:.2f})"
        pos_rows.append(row)
    df_pos = pd.DataFrame(pos_rows)

    return df_odor, df_pos, df_out


def fa_abortion_stats_session(results, return_df=False):
    trials = results.get("trial_data", pd.DataFrame())
    missing = _fa_abortion_frames_missing(trials)
    if missing is not None:
        print(missing)
        return None if not return_df else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    df_odor, df_pos, df_out = fa_abortion_stats(trials)

    if not return_df:
        if not df_odor.empty:
            print("=== By Odor ===")
            display(df_odor)
        if not df_pos.empty:
            print("=== By Position ===")
            display(df_pos)
        if not df_out.empty:
            print("=== By Odor+Position ===")
            display(df_out)
        if df_odor.empty and df_pos.empty and df_out.empty:
            print("No FA abortions found.")
    return (df_odor, df_pos, df_out) if return_df else None

def fa_port_ratio_by_odor(results, include_non_initiated=True, fa_type="FA_time_in"):
    """
    Calculate FA port bias ratio per odor: (Port A - Port B) / (Port A + Port B).
    
    This metric shows the signed bias in which port (A or B) is selected during
    false alarm responses for each odor. A ratio of 0 indicates no preference,
    positive values indicate bias towards port A, and negative values indicate bias towards port B.
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing 'aborted_sequences_detailed' and optionally 'non_initiated_FA'
    include_non_initiated : bool
        If True, include non-initiated FAs in calculation. Default: True
    fa_type : str
        Which FA type to filter for. Default: 'FA_time_in'
        Can be 'FA_time_in', 'FA_time_out', 'FA_late', or 'all' for all FA types.
    
    Returns:
    --------
    dict : Dictionary with structure:
        {
            'by_odor': pd.Series indexed by odor letter with FA port ratios,
            'counts': dict with counts of FA events per port per odor,
            'total_fa_by_odor': dict with total FA counts per odor
        }
    """
    print(f"FA Port Ratio by Odor ({fa_type}):")
    
    df = results.get("trial_data", pd.DataFrame())
    fa_noninit = results.get("non_initiated_FA", pd.DataFrame()) if include_non_initiated else pd.DataFrame()

    if df.empty and fa_noninit.empty:
        print("  No FA data with port and odor information found.")
        return {'by_odor': pd.Series(dtype=float), 'counts': {}, 'total_fa_by_odor': {}}

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    fa_ab = df[aborted_mask] if not df.empty else pd.DataFrame()

    # Define filter function based on fa_type
    if fa_type.lower() == 'all':
        fa_filter = lambda x: x.astype(str).str.startswith('FA_', na=False)
    else:
        fa_filter = lambda x: x.astype(str) == fa_type

    fa_ab = fa_ab[fa_filter(fa_ab.get("fa_label", pd.Series(dtype=str)))] if not fa_ab.empty else pd.DataFrame()
    fa_ni = fa_noninit[fa_filter(fa_noninit.get("fa_label", pd.Series(dtype=str)))] if not fa_noninit.empty else pd.DataFrame()

    fa_all = pd.concat([fa_ab, fa_ni], ignore_index=True) if include_non_initiated else fa_ab

    if fa_all.empty or "fa_port" not in fa_all.columns or "last_odor_name" not in fa_all.columns:
        print("  No FA data with port and odor information found.")
        return {'by_odor': pd.Series(dtype=float), 'counts': {}, 'total_fa_by_odor': {}}
    
    # Calculate ratio per odor
    ratios = {}
    counts = {}
    total_fa_by_odor = {}
    
    for odor in sorted(fa_all["last_odor_name"].dropna().unique()):
        fa_odor = fa_all[fa_all["last_odor_name"] == odor]
        n_port_a = (fa_odor["fa_port"] == 1).sum()
        n_port_b = (fa_odor["fa_port"] == 2).sum()
        n_total = n_port_a + n_port_b
        
        if n_total > 0:
            ratio = (n_port_a - n_port_b) / n_total
            ratios[odor] = ratio
            counts[odor] = {'port_a': n_port_a, 'port_b': n_port_b}
            total_fa_by_odor[odor] = n_total
            print(f"  {odor}: A={n_port_a}, B={n_port_b}, Bias ratio: {ratio:.3f}")
        else:
            ratios[odor] = np.nan
            counts[odor] = {'port_a': 0, 'port_b': 0}
            total_fa_by_odor[odor] = 0
    
    return {
        'by_odor': pd.Series(ratios).sort_index(),
        'counts': counts,
        'total_fa_by_odor': total_fa_by_odor
    }


# ================== NEW metrics -- no canonical version before Phase 4a =========
#
# restructure_2 Phase 4a. Each of these was computed inside a plotter in
# `visualization/` and existed nowhere in `metric_analysis` -- the audit's `NEW`
# class, i.e. the "lose no metric" checklist. The arithmetic reproduces the
# plotters'; what stays behind in `visualization/` is axis construction,
# cross-subject aggregation and styling, which are properties of a figure rather
# than of the data.
#
# None of these is reached by `run_all_metrics`, so none enters `metrics_*.json`
# or the regression fingerprint. Whether to save them is a 4b question (the
# registry decides), not a 4a one.


def _tz_naive(series):
    """Datetime Series with any timezone dropped, so subtraction is safe."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return s


def _fa_filter_mask(frame, fa_types=None):
    """The `fa_label` mask the FA-rate metrics share.

    `fa_types=None` means "any labelled false alarm", spelled as the plotters
    spell it: not the literal `nFA`, and not null. A set selects subtypes and is
    matched case-insensitively.
    """
    if "fa_label" not in frame.columns:
        return pd.Series(False, index=frame.index)
    labels = frame["fa_label"]
    lower = labels.astype(str).str.lower()
    if fa_types is None:
        return lower.ne("nfa") & labels.notna()
    return lower.isin({str(s).strip().lower() for s in fa_types})


def hidden_rule_mask(trials):
    """Boolean mask of hidden-rule trials -- the grouping key for the HR split.

    `by_group(decision_accuracy, trials, hidden_rule_mask(trials))` is the
    audit's checklist 7 (decision accuracy, HR vs non-HR): a granularity of
    `decision_accuracy`, not a metric of its own. It is deliberately **not**
    `hidden_rule_performance`, which has a different numerator *and* denominator.
    """
    return _truthy(trials, "hidden_rule_success")


def fa_rate_by_odor(trials, *, fa_types=None, odors=None):
    """FA aborts at an odor / (its passes in completed sequences + those aborts).

    Checklist 1. The denominator matches no canonical metric: not `FA_odor_bias`
    (aborts@odor) and not `odorx_abortion_rate` (presentations@odor). It counts
    how often the odor was sampled and passed, plus the times it was false-alarmed
    on -- so the rate answers "when this odor came up, how often did it draw a
    false alarm".

    `odors` fixes the index (and its order); by default every odor seen is
    reported. An odor with a zero denominator is omitted, not reported as 0.
    """
    if trials.empty:
        return pd.Series(dtype=float)
    aborted = _aborted_mask(trials)
    completed = trials[~aborted]
    ab = trials[aborted]
    ab_fa = ab[_fa_filter_mask(ab, fa_types)]

    completed_counts: dict = {}
    if "odor_sequence" in completed.columns:
        for seq in completed["odor_sequence"]:
            for tok in odor_sequence_tokens(seq):
                if tok is None or (isinstance(tok, float) and np.isnan(tok)):
                    continue
                letter = odor_letter(tok)
                completed_counts[letter] = completed_counts.get(letter, 0) + 1

    fa_counts: dict = {}
    if "last_odor_name" in ab_fa.columns:
        for last in ab_fa["last_odor_name"]:
            if last is None or (isinstance(last, float) and np.isnan(last)):
                continue
            letter = odor_letter(last)
            fa_counts[letter] = fa_counts.get(letter, 0) + 1

    keys = ([odor_letter(o) for o in odors] if odors is not None
            else sorted(set(completed_counts) | set(fa_counts)))
    rates = {}
    for od in keys:
        denom = completed_counts.get(od, 0) + fa_counts.get(od, 0)
        if denom > 0:
            rates[od] = fa_counts.get(od, 0) / denom
    return pd.Series(rates, dtype=float)


def fa_rate_by_position(trials, *, fa_types=None):
    """FA aborts at position *p* / trials that reached *p*.

    Checklist 5. The denominator is `frames.reached_counts`, the package's single
    definition of "reached" (audit Q5). The plotter used to count the positions
    listed in each trial's `presentations` blob -- Q5's "definition C", now
    deleted -- so the drawn denominators change here even though no saved metric
    value does.
    """
    if trials.empty:
        return pd.Series(dtype=float)
    reached = _reached_counts(trials)
    aborted = trials[_aborted_mask(trials)]
    fa = aborted[_fa_filter_mask(aborted, fa_types)]
    fa_counts: dict = {}
    if "last_odor_position" in fa.columns:
        pos = pd.to_numeric(fa["last_odor_position"], errors="coerce").dropna().astype(int)
        fa_counts = {int(p): int(n) for p, n in pos.value_counts().items()}
    rates = {p: fa_counts.get(p, 0) / n for p, n in reached.items() if n > 0}
    return pd.Series(rates, dtype=float).sort_index()


def rolling_reward_fraction(trials, window, *, step=1, include_avg=False, hr_only=False):
    """Rolling fraction of trials rewarded, divided by the **window**.

    Checklist 2, and deliberately not `over_windows(decision_accuracy, ...)`.
    The denominator is the window size, so timeouts -- and, unless the caller has
    already dropped them, aborts -- sit inside it. That is the audit's finding 12:
    the curve differs visibly from a rolling `decision_accuracy`, which is why
    this is a separately named metric rather than a granularity of an existing
    one.

    `include_avg` back-fills the warm-up, completing a not-yet-full window with
    the frame's overall rate so the series starts at the first trial instead of
    at trial `window`. `hr_only` narrows the numerator to hidden-rule rewards.

    Returns one value per row of `trials`, NaN where no window ends there.
    """
    n = len(trials)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    numerator = _flag(trials, "response_time_category", "rewarded")
    if hr_only:
        hr = trials.get("hidden_rule_success")
        hr = (hr.fillna(False).astype(bool) if isinstance(hr, pd.Series)
              else pd.Series(False, index=trials.index))
        numerator = numerator & hr
    rewards = numerator.astype(int).to_numpy(dtype=float)
    overall = float(np.mean(rewards))

    if include_avg:
        for i in range(0, n, step):
            if i < window:
                avail = rewards[: i + 1]
                out[i] = (float(np.sum(avail)) + (window - len(avail)) * overall) / float(window)
            else:
                out[i] = float(np.mean(rewards[i - window + 1: i + 1]))
    else:
        for end in range(window, n + 1, step):
            out[end - 1] = float(np.mean(rewards[end - window: end]))
    return out


def rolling_hr_reward_fraction(trials, window):
    """Rolling percentage of rewarded trials that were hidden-rule rewarded.

    Checklist 9. Related to `hidden_rule_performance` but not a granularity of
    it: the denominator is rewarded trials, not hidden-rule hits. Indexed by the
    rows of `trials` it kept, in `sequence_start` order.
    """
    rewarded = trials[(~_aborted_mask(trials))
                      & _flag(trials, "response_time_category", "rewarded")]
    if rewarded.empty:
        return pd.Series(dtype=float)
    for col in ("hidden_rule_success", "hit_hidden_rule"):
        if col in rewarded.columns:
            hr = rewarded[col].fillna(False).astype(bool)
            break
    else:
        hr = pd.Series(False, index=rewarded.index)
    if "sequence_start" in rewarded.columns:
        hr = hr.loc[rewarded["sequence_start"].sort_values().index]
    return hr.astype(int).rolling(window, min_periods=1).mean() * 100.0


def poke_durations(position_data, *, aborted=False):
    """Per-position poke durations for one outcome class, as a tidy frame.

    Completed trials come from `position_poke_times`; aborted trials from
    `presentations` with the abort event excluded -- the same sources, and the
    same exclusion, the canonical `avg_sampling_time_*` metrics use.

    **No `poke_time_ms > 0` filter.** The four extractors in `visualization/`
    each carried one; measured across all 9 fixture sessions it drops nothing,
    because a ~0 ms position is currently omitted by the writer entirely. Once
    Phase 7b writes those positions the filter would start excluding exactly the
    rows that fix adds, so it is removed rather than relocated --
    `sampled_positions(only_true_pokes=True)` is its proper successor.
    """
    empty = pd.DataFrame(columns=["position", "odor_name", "poke_time_ms"])
    if aborted:
        rows = _position_rows(position_data, "in_presentations", aborted=True)
        if rows is None or rows.empty:
            return empty
        idx = rows["index_in_trial"]
        rows = rows[idx.notna() & (idx != rows["last_event_index"])]
    else:
        rows = _position_rows(position_data, "in_poke_times", aborted=False)
        if rows is None or rows.empty:
            return empty
    rows = rows[rows["poke_time_ms"].notna()]
    if rows.empty:
        return empty
    return rows.loc[:, ["position", "odor_name", "poke_time_ms"]].reset_index(drop=True)


def _mean_sd_by(frame, key):
    if frame.empty:
        return pd.DataFrame(columns=["mean", "sd", "n"])
    grouped = frame.dropna(subset=[key]).groupby(key)["poke_time_ms"]
    # ddof=0: the plotters draw `np.std(values)`, i.e. the population SD.
    return pd.DataFrame({"mean": grouped.mean(), "sd": grouped.std(ddof=0),
                         "n": grouped.size()})


def poke_duration_by_position(position_data, *, aborted=False):
    """Mean and population SD of `poke_time_ms` per position. Checklist 3."""
    return _mean_sd_by(poke_durations(position_data, aborted=aborted), "position")


def poke_duration_by_odor(position_data, *, aborted=False):
    """Mean and population SD of `poke_time_ms` per odor.

    Checklist 4 in its `aborted=True` form: the canonical
    `avg_sampling_time_aborted_sequence` pools every aborted trial into one
    scalar, and no per-odor version existed. With `aborted=False` it is the
    per-odor completed-trial mean, i.e. `avg_sampling_time_odor_x` with an SD
    and a count alongside.
    """
    return _mean_sd_by(poke_durations(position_data, aborted=aborted), "odor_name")


def inter_trial_interval(trials):
    """Seconds from one trial ending to the next starting. Checklist 6.

    `sequence_start.shift(-1) - sequence_end`, so the last row is NaN. Pass a
    single session's trials: shifting across a session boundary would measure the
    gap between recordings, which is not an inter-trial interval.
    """
    if (trials.empty or "sequence_start" not in trials.columns
            or "sequence_end" not in trials.columns):
        return pd.Series(np.nan, index=trials.index, dtype=float)
    start = _tz_naive(trials["sequence_start"])
    end = _tz_naive(trials["sequence_end"])
    return (start.shift(-1) - end).dt.total_seconds()


def _first_hr_position(val):
    """First entry of `hidden_rule_hit_positions`, however it is stored."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        try:
            return int(val[0])
        except Exception:
            return None
    if isinstance(val, str):
        parsed = parse_json_column(val)
        try:
            if isinstance(parsed, (list, tuple)) and parsed:
                return int(parsed[0])
            if isinstance(parsed, (int, float)):
                return int(parsed)
        except Exception:
            return None
    return None


def hr_abort_poke_gap(trials, position_data):
    """Latency from the hidden-rule poke to the last poke of an aborted trial.

    Checklist 8: `last poke_odor_end - hidden-rule poke_odor_end`, on trials that
    aborted having hit the hidden rule, plus the start-to-end variant. No
    canonical metric measures any latency *between positions*.

    One row per qualifying trial; trials without a hidden-rule position or
    without usable poke timestamps are dropped rather than reported as NaN.
    """
    cols = ["global_trial_id", "hidden_rule_position",
            "delta_seconds", "delta_start_end_seconds"]
    if trials.empty or position_data is None or len(position_data) == 0:
        return pd.DataFrame(columns=cols)
    if "global_trial_id" not in position_data.columns:
        return pd.DataFrame(columns=cols)

    hr_trials = trials[_aborted_mask(trials) & _truthy(trials, "hit_hidden_rule")]
    if hr_trials.empty:
        return pd.DataFrame(columns=cols)

    poke = _position_rows(position_data, "in_poke_times")
    if poke is None or poke.empty:
        return pd.DataFrame(columns=cols)
    poke = poke.assign(_end=_tz_naive(poke["poke_odor_end"]),
                       _start=_tz_naive(poke["poke_odor_start"]))
    by_trial = {gid: sub for gid, sub in poke.groupby("global_trial_id")}

    rows = []
    for _, trial in hr_trials.iterrows():
        hr_pos = _first_hr_position(trial.get("hidden_rule_hit_positions"))
        if hr_pos is None:
            continue
        sub = by_trial.get(trial.get("global_trial_id"))
        if sub is None or sub.empty:
            continue
        ends = sub["_end"].dropna()
        if ends.empty:
            continue
        at_hr = sub[sub["position"] == hr_pos]
        hr_end = at_hr["_end"].dropna()
        if hr_end.empty:
            continue
        hr_start = at_hr["_start"].dropna()
        last_end = ends.max()
        rows.append({
            "global_trial_id": trial.get("global_trial_id"),
            "hidden_rule_position": hr_pos,
            "delta_seconds": (last_end - hr_end.iloc[-1]).total_seconds(),
            "delta_start_end_seconds": ((last_end - hr_start.iloc[-1]).total_seconds()
                                        if not hr_start.empty else np.nan),
        })
    return pd.DataFrame(rows, columns=cols)


# ---- trial-timing family (checklist 17-22) -------------------------------------
#
# All indexed by `global_trial_id`, so pass one session's frames: pooled frames
# repeat ids and the index alignment below would mis-pair trials.
#
# The 10x-group-mean outlier rule that `pred_seq_utils.response_time` and
# `fa_analysis` apply is deliberately **not** here. Judgement call 4 of the audit
# settles it: metrics raw, filtering is display -- so the rule stays in
# `visualization/`, where it can be seen and changed.


def _trial_position_frame(position_data, blob):
    """One blob's rows sorted by position within trial, or None if unusable."""
    rows = _position_rows(position_data, blob)
    if rows is None or rows.empty or "global_trial_id" not in rows.columns:
        return None
    return rows.sort_values(["global_trial_id", "position"], kind="stable")


def _deepest_position_timestamp(position_data, blob, field):
    """`field` at each trial's deepest position, tz-naive, indexed by trial id."""
    rows = _trial_position_frame(position_data, blob)
    if rows is None:
        return None
    frame = pd.DataFrame({"gid": rows["global_trial_id"].to_numpy(),
                          "ts": _tz_naive(rows[field]).to_numpy()})
    return frame.groupby("gid", sort=True)["ts"].agg(lambda s: s.iloc[-1])


def _trial_timestamp(trials, field):
    """A trial-level timestamp column, tz-naive, indexed by trial id."""
    if field not in trials.columns or "global_trial_id" not in trials.columns:
        return None
    return pd.Series(_tz_naive(trials[field]).to_numpy(),
                     index=trials["global_trial_id"].to_numpy())


def _latency_ms(later, earlier):
    """`later - earlier` in ms, dropping pairs where either side is missing.

    Vectorised `.dt.total_seconds()` on purpose. The plotters walk trials one at
    a time and go through *scalar* `Timedelta.total_seconds()`, which truncates
    to microseconds, so they silently discard the nanoseconds the blob
    timestamps carry (`...T13:49:07.507839999`). The timedeltas themselves are
    bit-identical either way; only the conversion differs. Measured on all 9
    fixture sessions the two forms agree to **0.999 ns**, on latencies of
    hundreds to thousands of ms -- so this keeps the exact value rather than
    reproducing the truncation.
    """
    if later is None or earlier is None:
        return pd.Series(dtype=float)
    return (later - earlier).dropna().dt.total_seconds() * 1000.0


def trial_poke_span(position_data):
    """Wall-clock span of a trial's odor-sampling phase, in ms. Checklist 17.

    `poke_odor_end` at the deepest position minus `poke_odor_start` at position 1.
    Distinct from `trial_poke_total`: the span contains the travel between ports,
    the sum does not. Trials missing either timestamp are dropped.
    """
    rows = _trial_position_frame(position_data, "in_poke_times")
    if rows is None:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({
        "gid": rows["global_trial_id"].to_numpy(),
        # Only position 1 contributes a start, so `max` picks it out.
        "start": _tz_naive(rows["poke_odor_start"]).where(rows["position"] == 1).to_numpy(),
        "end": _tz_naive(rows["poke_odor_end"]).to_numpy(),
    })
    grouped = frame.groupby("gid", sort=True)
    span = grouped["end"].agg(lambda s: s.iloc[-1]) - grouped["start"].max()
    return span.dropna().dt.total_seconds() * 1000.0


def trial_poke_total(position_data):
    """Sum of `poke_time_ms` across a trial's positions, in ms. Checklist 21.

    Related to `avg_sampling_time_completed_sequence` but per trial rather than a
    session mean.
    """
    rows = _position_rows(position_data, "in_poke_times")
    if rows is None or rows.empty or "global_trial_id" not in rows.columns:
        return pd.Series(dtype=float)
    usable = rows[rows["poke_time_ms"].notna()]
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.groupby("global_trial_id")["poke_time_ms"].sum()


def reward_delivery_latency(trials, position_data):
    """`first_supply_time` minus the last odor poke-out, in ms. Checklist 18.

    **Not** `trial_data.response_time_ms`, which is measured from the reward-port
    poke rather than from leaving the odor port -- the audit's finding 11, two
    quantities sharing an everyday name and not a definition. Written twice
    today, as `pred_seq_utils.response_time` and `sing_rew._response_time_ms`.
    """
    return _latency_ms(_trial_timestamp(trials, "first_supply_time"),
                       _deepest_position_timestamp(position_data, "in_poke_times",
                                                   "poke_odor_end"))


def valve_to_reward_latency(trials, position_data):
    """`first_supply_time` minus the last position's `valve_start`, in ms.

    Checklist 20. Nothing canonical measures anything from a valve opening.
    """
    return _latency_ms(_trial_timestamp(trials, "first_supply_time"),
                       _deepest_position_timestamp(position_data, "in_valve_times",
                                                   "valve_start"))


def fa_latency_from_pokeout(trials, position_data, *, fa_types=None):
    """`fa_time` minus the poke-out of the trial's last odor, in ms. Checklist 19.

    **Not** `trial_data.fa_latency_ms`, which is measured from the abortion
    timestamp (finding 11). The reference is the *first* position whose
    `odor_name` matches the trial's `last_odor`, not the deepest one -- they
    differ when an odor repeats within a sequence, and the plotter uses the first.
    """
    rows = _trial_position_frame(position_data, "in_poke_times")
    if rows is None or "last_odor" not in trials.columns:
        return pd.Series(dtype=float)
    selected = trials[_fa_filter_mask(trials, fa_types)] if fa_types is not None else trials
    if selected.empty or "global_trial_id" not in selected.columns:
        return pd.Series(dtype=float)

    last_odor = pd.Series(selected["last_odor"].map(odor_letter).to_numpy(),
                          index=selected["global_trial_id"].to_numpy())
    wanted = rows["global_trial_id"].map(last_odor)
    at_last_odor = rows[rows["odor_name"].map(odor_letter) == wanted]
    if at_last_odor.empty:
        return pd.Series(dtype=float)
    first = at_last_odor.groupby("global_trial_id", sort=True).head(1)
    poke_end = pd.Series(_tz_naive(first["poke_odor_end"]).to_numpy(),
                         index=first["global_trial_id"].to_numpy())
    return _latency_ms(_trial_timestamp(selected, "fa_time"), poke_end)


def false_response_ratio_contributions(trials, *, fr_types=None):
    completed = ~_aborted_mask(trials)
    if "false_response" not in trials.columns:
        return pd.Series(0, index=trials.index), completed.astype(int)
    fr = trials["false_response"] == True  # noqa: E712 (element-wise, NaN-safe)
    if fr_types is not None and "fr_label" in trials.columns:
        wanted = {fr_types} if isinstance(fr_types, str) else set(fr_types)
        fr = fr & trials["fr_label"].isin(wanted)
    return (completed & fr).astype(int), completed.astype(int)


def false_response_ratio(trials, *, fr_types=None):
    """False-response trials / completed trials. Checklist 22.

    **Not** the single-reward `fa_rate`, which is `false_alarm / n_nogo` off a
    different column (`fa_label`, not `fr_label`). `fr_types=None` counts every
    `false_response == True` trial whatever its label.
    """
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*false_response_ratio_contributions(trials, fr_types=fr_types))

