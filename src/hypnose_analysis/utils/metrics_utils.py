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
from hypnose_analysis.paths import get_derivatives_root
# ================== Loading, Wrapper, and Helper Functions ==================

def load_session_results(subjid, date):
    """
    Load saved analysis results for a given subject and date.
    Returns a dict with trial_data, non-initiated tables, and metadata.
    """
    derivatives_dir = get_derivatives_root()
    sub_str = f"sub-{str(subjid).zfill(3)}"
    date_str = str(date)

    # Find subject directory (may have multiple _id-*)
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    if len(subject_dirs) > 1:
        print(f"Warning: Multiple subject directories found for {sub_str}, using first one")
    subject_dir = subject_dirs[0]

    # Find session directory for the date
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        # Show available sessions for better error reporting
        all_sessions = list(subject_dir.glob("ses-*"))
        session_names = [d.name for d in all_sessions]
        raise FileNotFoundError(f"No session found for date {date_str} in {subject_dir}.\n"
                                f"Available sessions: {session_names}")
    if len(session_dirs) > 1:
        print(f"Warning: Multiple sessions found for date {date_str}, using first one")
    session_dir = session_dirs[0]

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

    # Tables still saved separately
    for t in ["non_initiated_sequences", "non_initiated_odor1_attempts", "non_initiated_FA"]:
        f = results_dir / f"{t}.csv"
        results[t] = pd.read_csv(f) if f.exists() else pd.DataFrame()

    # Attach manifest and summary
    results["manifest"] = manifest
    results["summary"] = summary
    results["results_dir"] = str(results_dir)

    return results


def _filter_session_dirs(subj_dir: Path, dates: Optional[Union[Iterable[Union[int,str]], tuple]]):
    ses_dirs = sorted(subj_dir.glob("ses-*_date-*"))
    if dates is None:
        return ses_dirs
    # dates can be list/iterable or (start, end) tuple
    def norm_date(d):
        s = str(d)
        return int(s) if s.isdigit() else None
    if isinstance(dates, tuple) and len(dates) == 2:
        start = norm_date(dates[0]); end = norm_date(dates[1])
        out = []
        for d in ses_dirs:
            try:
                ds = int(d.name.split("_date-")[-1])
                if (start is None or ds >= start) and (end is None or ds <= end):
                    out.append(d)
            except Exception:
                pass
        return out
    # iterable of dates
    wanted = {norm_date(d) for d in dates}
    out = []
    for d in ses_dirs:
        try:
            ds = int(d.name.split("_date-")[-1])
            if ds in wanted:
                out.append(d)
        except Exception:
            pass
    return out

def parse_json_column(val):
    if isinstance(val, str):
        try:
            val_fixed = val.replace('""', '"')
            return json.loads(val_fixed)
        except Exception:
            return {} if val.strip().startswith("{") else []
    return val



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
        sub_norm = _normalize_subjid(subjid)
        date_norm = _normalize_date(date)
        if not sub_norm or not date_norm:
            return None
        subject_dirs = sorted(derivatives_dir.glob(f"{sub_norm}_id-*"))
        if not subject_dirs:
            return None
        # prefer deterministic ordering
        for subj_dir in subject_dirs:
            session_dirs = sorted(subj_dir.glob(f"ses-*_date-{date_norm}"))
            if session_dirs:
                return session_dirs[0] / "saved_analysis_results"
        return None

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
        metrics['decision_accuracy'] = decision_accuracy(results)
        print("\n--- Decision Accuracy by Odor ---")
        accuracy_by_odor = decision_accuracy_by_odor(results)
        metrics['decision_accuracy_by_odor'] = accuracy_by_odor.to_dict() if len(accuracy_by_odor) > 0 else {}
        print("\n--- Global Choice Accuracy ---")
        metrics['global_choice_accuracy'] = global_choice_accuracy(results)
        print("\n--- Premature Response Rate ---")
        metrics['premature_response_rate'] = premature_response_rate(results)
        print("\n--- Response-Contingent False Alarm Rate ---")
        metrics['response_contingent_FA_rate'] = response_contingent_FA_rate(results)
        print("\n--- Global False Alarm Rate ---")
        metrics['global_FA_rate'] = global_FA_rate(results)
        print("\n--- FA Odor Bias ---")
        fa_odor = FA_odor_bias(results)
        metrics['FA_odor_bias'] = fa_odor.to_dict() if hasattr(fa_odor, 'to_dict') else fa_odor
        print("\n--- FA Position Bias ---")
        fa_pos = FA_position_bias(results)
        metrics['FA_position_bias'] = fa_pos.to_dict() if hasattr(fa_pos, 'to_dict') else fa_pos
        print("\n--- Sequence Completion Rate ---")
        metrics['sequence_completion_rate'] = sequence_completion_rate(results)
        print("\n--- Odor Abortion Rate ---")
        odor_ab = odorx_abortion_rate(results)
        metrics['odorx_abortion_rate'] = odor_ab.to_dict() if hasattr(odor_ab, 'to_dict') else odor_ab
        print("\n--- Hidden Rule Performance ---")
        metrics['hidden_rule_performance'] = hidden_rule_performance(results)
        print("\n--- Hidden Rule Detection Rate ---")
        metrics['hidden_rule_detection_rate'] = hidden_rule_detection_rate(results)
        print("\n--- Hidden Rule Performance/Detection by Odor ---")
        metrics['hidden_rule_by_odor'] = hidden_rule_counts_by_odor(results)
        print("\n--- Choice Timeout Rate ---")
        metrics['choice_timeout_rate'] = choice_timeout_rate(results)
        print("\n--- Average Sampling Time per Odor (Completed) ---")
        avg_samp_odor = avg_sampling_time_odor_x(results)
        metrics['avg_sampling_time_odor_x'] = avg_samp_odor.to_dict() if hasattr(avg_samp_odor, 'to_dict') else avg_samp_odor
        print("\n--- Average Sampling Time (Completed Sequences) ---")
        metrics['avg_sampling_time_completed_sequence'] = avg_sampling_time_completed_sequence(results)
        print("\n--- Average Sampling Time (Aborted Sequences) ---")
        metrics['avg_sampling_time_aborted_sequence'] = avg_sampling_time_aborted_sequence(results)
        print("\n--- Average Sampling Time (Initiation Abortions) ---")
        metrics['avg_sampling_time_initiation_abortion'] = avg_sampling_time_initiation_abortion(results)
        print("\n--- Abortion Rate by Position ---")
        abrt_pos = abortion_rate_positionX(results)
        metrics['abortion_rate_positionX'] = abrt_pos.to_dict() if hasattr(abrt_pos, 'to_dict') else abrt_pos
        print("\n--- Average Response Time ---")
        metrics['avg_response_time'] = avg_response_time(results)
        print("\n--- FA Average Response Times ---")
        metrics['FA_avg_response_times'] = FA_avg_response_times(results)
        print("\n--- Response Rate ---")
        metrics['response_rate'] = response_rate(results)
        print("\n--- Manual vs Auto Stop Preference ---")
        metrics['manual_vs_auto_stop_preference'] = manual_vs_auto_stop_preference(results)
        print("\n--- Non-Initiated FA Rate ---")
        metrics['non_initiated_FA_rate'] = non_initiated_FA_rate(results)
        print("\n--- Non-Initiation Odor Bias ---")
        noninit_odor = non_initiation_odor_bias(results)
        metrics['non_initiation_odor_bias'] = noninit_odor.to_dict() if hasattr(noninit_odor, 'to_dict') else noninit_odor
        print("\n--- Odor Initiation Bias ---")
        odor_init = odor_initiation_bias(results)
        metrics['odor_initiation_bias'] = odor_init.to_dict() if hasattr(odor_init, 'to_dict') else odor_init
        print("\n--- FA Abortion Stats ---")
        fa_ab_stats = fa_abortion_stats(results, return_df=True)
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
        sub_str = f"sub-{str(subjids[0]).zfill(3)}"
        subj_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
        if not subj_dirs:
            raise FileNotFoundError(f"No subject directory found for {sub_str}")
        subj_dir = subj_dirs[0]
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

    # Find all subject directories
    subj_dirs = list(derivatives_dir.glob("sub-*_id-*")) if subjids is None else [
        d for subjid in subjids
        for d in derivatives_dir.glob(f"sub-{str(subjid).zfill(3)}_id-*")
    ]
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

def decision_accuracy(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Decision Accuracy: no trial_data with response_time_category")
        return 0, 0, np.nan

    rew_mask = df["response_time_category"] == "rewarded"
    unr_mask = df["response_time_category"] == "unrewarded"
    n_rew = int(rew_mask.sum())
    n_unr = int(unr_mask.sum())
    denom = n_rew + n_unr
    acc = n_rew / denom if denom > 0 else np.nan
    print(f"Decision Accuracy: {n_rew}/{denom} = {acc:.3f}")
    return n_rew, denom, acc


def global_choice_accuracy(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Global Choice Accuracy: no trial_data with response_time_category")
        return 0, 0, np.nan

    n_correct = int((df["response_time_category"] == "rewarded").sum())
    n_incorrect = int((df["response_time_category"] == "unrewarded").sum())
    n_fa_time_in = int((df.get("fa_label") == "FA_time_in").sum()) if "fa_label" in df.columns else 0

    n_total = n_correct + n_incorrect + n_fa_time_in
    accuracy = n_correct / n_total if n_total > 0 else np.nan

    print(f"Global Choice Accuracy: {n_correct}/{n_total} = {accuracy:.3f}")
    print(f"  - Correct choices: {n_correct}")
    print(f"  - Incorrect choices: {n_incorrect}")
    print(f"  - False alarms (FA Time In): {n_fa_time_in}")

    return n_correct, n_total, accuracy

def decision_accuracy_by_odor(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "last_odor" not in df.columns:
        print("Decision Accuracy by Odor: no trial_data with response_time_category/last_odor")
        return pd.DataFrame()

    def extract_odor_letter(odor_str):
        if pd.isna(odor_str):
            return np.nan
        if isinstance(odor_str, str) and odor_str.startswith("Odor"):
            return odor_str.replace("Odor", "")
        return odor_str

    df_local = df.copy()
    df_local["odor_letter"] = df_local["last_odor"].apply(extract_odor_letter)

    rows = []
    odors = sorted(df_local["odor_letter"].dropna().unique())

    print("Decision Accuracy by Odor:")
    for odor in odors:
        odor_trials = df_local[df_local["odor_letter"] == odor]
        n_rew = int((odor_trials["response_time_category"] == "rewarded").sum())
        n_unr = int((odor_trials["response_time_category"] == "unrewarded").sum())
        n_tmo = int((odor_trials["response_time_category"] == "timeout_delayed").sum())
        denom_ab = n_rew + n_unr
        denom_total = denom_ab + n_tmo
        acc_ab = n_rew / denom_ab if denom_ab > 0 else np.nan
        acc_total = n_rew / denom_total if denom_total > 0 else np.nan

        def _fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "nan"

        print(f"  Odor {odor}: {n_rew} rewarded, {n_unr} unrewarded, {n_tmo} timeout")
        print(f"       Decision Accuracy AB: {n_rew}/{denom_ab} = {_fmt(acc_ab)}, Total: {n_rew}/{denom_total} = {_fmt(acc_total)}")

        rows.append({
            'odor': odor,
            'rewarded': n_rew,
            'unrewarded': n_unr,
            'timeout': n_tmo,
            'decision_accuracy_ab': acc_ab,
            'decision_accuracy_total': acc_total,
            'denominator_ab': denom_ab,
            'denominator_total': denom_total,
        })

    return pd.DataFrame(rows).set_index('odor').sort_index()

def premature_response_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Premature Response Rate: no trial_data")
        return 0, 0, np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    if aborted.empty:
        print("Premature Response Rate: no aborted trials")
        return 0, 0, np.nan

    n_fa = int((aborted.get("fa_label") == "FA_time_in").sum()) if "fa_label" in aborted.columns else 0
    n_total = len(aborted)
    rate = n_fa / n_total if n_total > 0 else np.nan
    print(f"Premature Response Rate: {n_fa}/{n_total} = {rate:.3f}")
    return n_fa, n_total, rate

def response_contingent_FA_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Response-Contingent False Alarm Rate: missing trial_data/response_time_category")
        return 0, 0, np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    n_fa = int((aborted.get("fa_label") == "FA_time_in").sum()) if "fa_label" in aborted.columns else 0

    n_rew = int((df["response_time_category"] == "rewarded").sum())
    n_unr = int((df["response_time_category"] == "unrewarded").sum())

    denom = n_fa + n_rew + n_unr
    rate = n_fa / denom if denom > 0 else np.nan
    print(f"Response-Contingent False Alarm Rate: {n_fa}/{denom} = {rate:.3f}")
    return n_fa, denom, rate

def global_FA_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Global False Alarm Rate: no trial_data")
        return 0, 0, np.nan

    n_fa = int((df.get("fa_label") == "FA_time_in").sum()) if "fa_label" in df.columns else 0

    if "global_trial_id" in df.columns:
        n_ini = int(df["global_trial_id"].notna().sum())
    else:
        n_ini = len(df)

    rate = n_fa / n_ini if n_ini > 0 else np.nan
    print(f"Global False Alarm Rate: {n_fa}/{n_ini} = {rate:.3f}")
    return n_fa, n_ini, rate

def FA_odor_bias(results):
    print("FA Odor Bias for FA Time In:")
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "fa_label" not in df.columns:
        return {'bias': {}, 'n_fa': {}, 'n_ab': {}, 'total_fa': 0, 'total_ab': 0}

    odor_col = "last_odor_name" if "last_odor_name" in df.columns else "last_odor"
    if odor_col not in df.columns:
        return {'bias': {}, 'n_fa': {}, 'n_ab': {}, 'total_fa': 0, 'total_ab': 0}

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    if aborted.empty:
        return {'bias': {}, 'n_fa': {}, 'n_ab': {}, 'total_fa': 0, 'total_ab': 0}

    fa_mask = aborted["fa_label"] == "FA_time_in"
    odors = sorted(aborted[odor_col].dropna().unique())
    bias = {}
    n_fa = {}
    n_ab = {}
    total_fa = int(fa_mask.sum())
    total_ab = len(aborted)
    for od in odors:
        at_od = aborted[odor_col] == od
        fa_at_od = fa_mask & at_od
        n_fa_od = int(fa_at_od.sum())
        n_ab_od = int(at_od.sum())
        n_fa[od] = n_fa_od
        n_ab[od] = n_ab_od
        bias[od] = (n_fa_od / n_ab_od) / (total_fa / total_ab) if n_ab_od > 0 and total_ab > 0 and total_fa > 0 else np.nan
        print(f"{od}: {n_fa_od}/{n_ab_od} FA, Bias: {bias[od]:.3f}")
    return {'bias': bias, 'n_fa': n_fa, 'n_ab': n_ab, 'total_fa': total_fa, 'total_ab': total_ab}

def FA_position_bias(results):
    print("FA Position Bias for FA Time In:")
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "fa_label" not in df.columns:
        return pd.Series(dtype=float)

    position_col = "last_odor_position" if "last_odor_position" in df.columns else "last_event_index"
    if position_col not in df.columns:
        return pd.Series(dtype=float)

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    if aborted.empty:
        return pd.Series(dtype=float)

    fa_mask = aborted["fa_label"] == "FA_time_in"
    positions = sorted(aborted[position_col].dropna().unique())
    bias = {}
    total_fa = int(fa_mask.sum())
    total_ab = len(aborted)
    for pos in positions:
        at_pos = aborted[position_col] == pos
        fa_at_pos = fa_mask & at_pos
        n_fa_pos = int(fa_at_pos.sum())
        n_ab_pos = int(at_pos.sum())
        pos_report = int(pos) + 1 if position_col == "last_event_index" else int(pos)
        bias[pos_report] = (n_fa_pos / n_ab_pos) / (total_fa / total_ab) if n_ab_pos > 0 and total_ab > 0 and total_fa > 0 else np.nan
        print(f"Position {pos_report}: {n_fa_pos}/{n_ab_pos} FA, Bias: {bias[pos_report]:.3f}")
    return pd.Series(bias).sort_index()

def sequence_completion_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Sequence Completion Rate: no trial_data")
        return 0, 0, np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    n_completed = int((~aborted_mask).sum())

    denom = int(df["global_trial_id"].notna().sum()) if "global_trial_id" in df.columns else len(df)
    rate = n_completed / denom if denom > 0 else np.nan
    print(f"Sequence Completion Rate: {n_completed}/{denom} = {rate:.3f}")
    return n_completed, denom, rate

def odorx_abortion_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "presentations" not in df.columns:
        return pd.Series(dtype=float)

    odor_col = "last_odor_name" if "last_odor_name" in df.columns else "last_odor"
    if odor_col not in df.columns:
        return pd.Series(dtype=float)

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]

    abortions = aborted[odor_col].dropna().value_counts().to_dict()

    presentations = {}
    for _, row in df.iterrows():
        pres_list = parse_json_column(row.get("presentations", []))
        if isinstance(pres_list, list):
            for pres in pres_list:
                od = pres.get("odor_name") if isinstance(pres, dict) else None
                if od is not None:
                    presentations[od] = presentations.get(od, 0) + 1

    all_odors = set(presentations.keys()).union(abortions.keys())
    rates = {}
    for od in sorted(all_odors):
        n_ab = abortions.get(od, 0)
        n_pres = presentations.get(od, 0)
        rates[od] = n_ab / n_pres if n_pres > 0 else np.nan
        print(f"{od}: {n_ab}/{n_pres} abortions, Rate: {rates[od]:.3f}")
    return pd.Series(rates).sort_index()

def hidden_rule_performance(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Performance: no trial_data")
        return 0, 0, np.nan

    success_mask = df["hidden_rule_success"].apply(_is_truthy) if "hidden_rule_success" in df.columns else pd.Series(False, index=df.index)
    reward_mask = df["response_time_category"] == "rewarded" if "response_time_category" in df.columns else pd.Series(False, index=df.index)
    num_mask = success_mask & reward_mask
    n_hr_rewarded = int(num_mask.sum())

    denom_mask = df["hit_hidden_rule"].apply(_is_truthy) if "hit_hidden_rule" in df.columns else pd.Series(False, index=df.index)
    denom = int(denom_mask.sum())

    rate = n_hr_rewarded / denom if denom > 0 else np.nan
    print(f"Hidden Rule Performance: {n_hr_rewarded}/{denom} = {rate:.3f}")
    return n_hr_rewarded, denom, rate

def hidden_rule_detection_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Detection Rate: no trial_data")
        return 0, 0, np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    success_mask = df["hidden_rule_success"].apply(_is_truthy) if "hidden_rule_success" in df.columns else pd.Series(False, index=df.index)
    num_mask = (~aborted_mask) & success_mask
    n_hr_completed = int(num_mask.sum())

    denom_mask = df["hit_hidden_rule"].apply(_is_truthy) if "hit_hidden_rule" in df.columns else pd.Series(False, index=df.index)
    denom = int(denom_mask.sum())

    rate = n_hr_completed / denom if denom > 0 else np.nan
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


def hidden_rule_counts_by_odor(results):
    """
    Aggregate HR trials by odor across outcome categories to support per-odor performance/detection.
    Returns a dict with hr_odors, hr_positions, and per-odor counts plus rates.
    """
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Counts by Odor: no trial_data")
        return {"hr_odors": [], "hr_positions": [], "by_odor": {}}

    hr_odors, hr_positions = _extract_hr_config(results)
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

    # Total presentations per odor (match odorx_abortion_rate logic)
    presentations = {}
    for _, row in df.iterrows():
        pres_list = parse_json_column(row.get("presentations", []))
        if isinstance(pres_list, list):
            for pres in pres_list:
                od = pres.get("odor_name") if isinstance(pres, dict) else None
                if od is not None and od in hr_set:
                    presentations[od] = presentations.get(od, 0) + 1

    def _fmt_rate(val):
        return f"{val:.3f}" if isinstance(val, (int, float, np.floating)) and not np.isnan(val) else "nan"

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

        print(
            f"Hidden Rule Odor {odor}: {rewarded} Rewarded, {unrewarded} Unrewarded, {timeout} Timeout, {total_presentations} Total Presentations."
        )
        print(
            f"  HR Odor {odor} Performance: {rewarded}/{completed_no_timeout} = {_fmt_rate(performance)}, "
            f"HR Odor {odor} Detection Rate: {completed_no_timeout}/{total_presentations} = {_fmt_rate(detection_rate)}"
        )

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

def choice_timeout_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Choice Timeout Rate: no trial_data/response_time_category")
        return 0, 0, np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    completed = df[~aborted_mask]

    n_tmo = int((completed["response_time_category"] == "timeout_delayed").sum())
    denom = len(completed)
    rate = n_tmo / denom if denom > 0 else np.nan
    print(f"Choice Timeout Rate: {n_tmo}/{denom} = {rate:.3f}")
    return n_tmo, denom, rate

def avg_sampling_time_odor_x(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        return pd.Series(dtype=float)

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    completed = df[~aborted_mask]
    if completed.empty:
        return pd.Series(dtype=float)

    odor_times = {}
    for _, row in completed.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        if isinstance(ppt, dict):
            for _, info in ppt.items():
                od = info.get("odor_name") if isinstance(info, dict) else None
                poke_ms = info.get("poke_time_ms") if isinstance(info, dict) else None
                if od is not None and poke_ms is not None:
                    odor_times.setdefault(od, []).append(poke_ms)

    avg_times = pd.Series({od: np.mean(times) for od, times in odor_times.items()}).sort_index()
    for odor, avg_time in avg_times.items():
        print(f"{odor} Average Sampling Time: {avg_time:.2f} ms")

    return avg_times

def avg_sampling_time_completed_sequence(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        return np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    completed = df[~aborted_mask]

    total_time = 0.0
    total_presentations = 0
    for _, row in completed.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        if isinstance(ppt, dict):
            for _, info in ppt.items():
                poke_ms = info.get("poke_time_ms") if isinstance(info, dict) else None
                if poke_ms is not None:
                    total_time += poke_ms
                    total_presentations += 1
    avg = total_time / total_presentations if total_presentations > 0 else np.nan
    print(f"Average Sampling Time (Completed Sequences): {avg:.2f} ms")
    return avg

def avg_sampling_time_aborted_sequence(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        return np.nan

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    if aborted.empty:
        return np.nan

    total_time = 0.0
    total_presentations = 0
    for _, row in aborted.iterrows():
        pres_list = parse_json_column(row.get("presentations", []))
        last_idx = row.get("last_event_index")
        if isinstance(pres_list, list):
            for pres in pres_list:
                idx = pres.get("index_in_trial") if isinstance(pres, dict) else None
                poke_ms = pres.get("poke_time_ms") if isinstance(pres, dict) else None
                if idx is not None and idx != last_idx and poke_ms is not None:
                    total_time += poke_ms
                    total_presentations += 1
    avg = total_time / total_presentations if total_presentations > 0 else np.nan
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

def abortion_rate_positionX(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        return pd.Series(dtype=float)

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    completed = df[~aborted_mask]

    position_col = "last_odor_position" if "last_odor_position" in df.columns else "last_event_index"
    if position_col not in df.columns:
        return pd.Series(dtype=float)

    abortions = aborted[position_col].dropna().value_counts().to_dict()

    reached = {}

    for _, row in aborted.iterrows():
        last_pos_val = row.get(position_col)
        if pd.notnull(last_pos_val):
            try:
                last_pos = int(last_pos_val)
            except Exception:
                continue
            for pos in range(1, last_pos + 1):
                reached[pos] = reached.get(pos, 0) + 1

    for _, row in completed.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        max_pos = None
        if isinstance(ppt, dict) and ppt:
            try:
                max_pos = max(int(k) for k in ppt.keys())
            except Exception:
                max_pos = None
        if max_pos is None and position_col in row:
            le = row.get(position_col)
            try:
                max_pos = int(le) if pd.notnull(le) else None
            except Exception:
                max_pos = None
        if max_pos is not None:
            for pos in range(1, int(max_pos) + 1):
                reached[pos] = reached.get(pos, 0) + 1

    all_positions = sorted(set(list(abortions.keys()) + list(reached.keys())))
    rates = {}
    for pos in all_positions:
        n_ab = abortions.get(pos, 0)
        n_reached = reached.get(pos, 0)
        rates[pos] = n_ab / n_reached if n_reached > 0 else np.nan
        print(f"Position {pos}: {n_ab}/{n_reached} abortions, Rate: {rates[pos]:.3f}")
    return pd.Series(rates).sort_index()

def avg_response_time(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "response_time_ms" not in df.columns:
        print("No response time data available.")
        return {}

    def avg_for_mask(mask):
        s = pd.to_numeric(df.loc[mask, "response_time_ms"], errors="coerce").dropna()
        return float(s.mean()) if not s.empty else np.nan, len(s)

    out = {}
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"), ("Reward Timeout", "timeout_delayed")]:
        avg, n = avg_for_mask(df["response_time_category"] == key)
        print(f"{label}: {avg:.1f} ms (n={n})" if not np.isnan(avg) else f"{label}: nan (n={n})")
        out[label] = avg

    mask_both = df["response_time_category"].isin(["rewarded", "unrewarded"])
    avg_both, n_both = avg_for_mask(mask_both)
    print(f"Average Response Time (Rewarded + Unrewarded): {avg_both:.1f} ms (n={n_both})" if not np.isnan(avg_both) else f"Average Response Time (Rewarded + Unrewarded): nan (n={n_both})")
    out["Average Response Time (Rewarded + Unrewarded)"] = avg_both

    return out

def FA_avg_response_times(results):
    out = {}
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "fa_label" not in df.columns or "fa_latency_ms" not in df.columns:
        return out

    fa_df = df[df["fa_label"].notna()]
    for label, pretty in [
        ("FA_time_in", "FA Time In"),
        ("FA_time_out", "FA Time Out"),
        ("FA_late", "FA Late"),
    ]:
        s = pd.to_numeric(fa_df.loc[fa_df["fa_label"] == label, "fa_latency_ms"], errors="coerce").dropna()
        avg = s.mean() if not s.empty else np.nan
        print(f"{pretty}: avg={avg:.1f} ms (n={len(s)})" if not np.isnan(avg) else f"{pretty}: nan (n={len(s)})")
        out[pretty] = float(avg) if not np.isnan(avg) else np.nan
    return out

def response_rate(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Response Rate: no trial_data/response_time_category")
        return 0, 0, np.nan

    n_rew = int((df["response_time_category"] == "rewarded").sum())
    n_unr = int((df["response_time_category"] == "unrewarded").sum())
    n_tmo = int((df["response_time_category"] == "timeout_delayed").sum())

    denom = n_rew + n_unr + n_tmo
    num = n_rew + n_unr
    rate = num / denom if denom > 0 else np.nan
    print(f"Response Rate: {num}/{denom} = {rate:.3f}")
    return num, denom, rate

def manual_vs_auto_stop_preference(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    completed = df[~aborted_mask]

    short = 0
    long = 0
    for _, row in completed.iterrows():
        vts = parse_json_column(row.get("position_valve_times", {}))
        if isinstance(vts, dict):
            for _, info in vts.items():
                dur = info.get("valve_duration_ms") if isinstance(info, dict) else None
                if dur is not None:
                    if dur <= 1000:
                        short += 1
                    elif dur >= 1000:
                        long += 1
    ratio = short / long if long > 0 else float('nan')
    print(f"Manual Stops: {short}")
    print(f"Auto Stops: {long}")
    print(f"Manual vs Auto Stop: {ratio:.2f}")
    return {"short_valve": short, "long_valve": long, "ratio": ratio}

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

def odor_initiation_bias(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "abortion_type" not in df.columns:
        return pd.Series(dtype=float)

    odor_col = "last_odor_name" if "last_odor_name" in df.columns else "last_odor"
    if odor_col not in df.columns:
        return pd.Series(dtype=float)

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted = df[aborted_mask]
    if aborted.empty:
        return pd.Series(dtype=float)

    init_mask = aborted["abortion_type"] == "initiation_abortion"
    odors = aborted[odor_col].dropna().unique()
    bias = {}
    total_init = int(init_mask.sum())
    total_ab = len(aborted)
    for od in sorted(odors):
        n_init_od = int(((aborted[odor_col] == od) & init_mask).sum())
        n_ab_od = int((aborted[odor_col] == od).sum())
        bias[od] = (n_init_od / n_ab_od) / (total_init / total_ab) if n_ab_od > 0 and total_ab > 0 and total_init > 0 else np.nan
        print(f"{od}: {n_init_od}/{n_ab_od} initiation abortions, Bias: {bias[od]:.3f}")
    return pd.Series(bias).sort_index()

def fa_abortion_stats(results, return_df=False):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "fa_label" not in df.columns:
        print("No FA abortion data available.")
        return None if not return_df else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    odor_col = "last_odor_name" if "last_odor_name" in df.columns else "last_odor"
    if odor_col not in df.columns:
        print("No FA abortion data available (missing odor column).")
        return None if not return_df else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    pos_col = "last_odor_position" if "last_odor_position" in df.columns else None
    if pos_col is None:
        print("No FA abortion data available (missing last_odor_position).")
        return None if not return_df else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    aborted_all = df[aborted_mask]
    completed = df[~aborted_mask]
    allowed_fa = {"FA_time_in", "FA_time_out", "FA_late"}
    fa_mask = aborted_all["fa_label"].isin(allowed_fa)
    fa_df = aborted_all[fa_mask]

    if aborted_all.empty:
        print("No aborted trials found.")
        return None if not return_df else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

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
    reached = {}
    # Aborted trials: all positions up to last aborted position count as reached
    for _, row in aborted_all.iterrows():
        last_pos_val = row.get(pos_col)
        if pd.notnull(last_pos_val):
            try:
                last_pos = int(last_pos_val)
            except Exception:
                continue
            for pos in range(1, last_pos + 1):
                reached[pos] = reached.get(pos, 0) + 1

    # Completed trials: use position_poke_times to infer reached positions; fallback to pos_col
    for _, row in completed.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        max_pos = None
        if isinstance(ppt, dict) and ppt:
            try:
                max_pos = max(int(k) for k in ppt.keys())
            except Exception:
                max_pos = None
        if max_pos is None and pos_col in row:
            le = row.get(pos_col)
            try:
                max_pos = int(le) if pd.notnull(le) else None
            except Exception:
                max_pos = None
        if max_pos is not None:
            for pos in range(1, int(max_pos) + 1):
                reached[pos] = reached.get(pos, 0) + 1

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

