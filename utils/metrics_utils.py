import sys
import os
project_root = os.path.abspath("/Users/joschua/repos/harris_lab/hypnose/hypnose-analysis")
if project_root not in sys.path:
    sys.path.append(project_root)
import os
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
import math
from pathlib import Path
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





# ================== Loading, Wrapper, and Helper Functions ==================

def load_session_results(subjid, date):
    """
    Load saved analysis results for a given subject and date.
    Returns a dict of DataFrames and metadata, matching classification keys.
    """
    base_dir = Path("/Volumes/harris/hypnose/derivatives")
    sub_str = f"sub-{str(subjid).zfill(3)}"
    date_str = str(date)

    # Find subject directory (may have multiple _id-*)
    subject_dirs = list(base_dir.glob(f"{sub_str}_id-*"))
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

    # Load all CSV tables
    tables = [
        "initiated_sequences","non_initiated_sequences","non_initiated_odor1_attempts",
        "completed_sequences","completed_sequences_with_response_times",
        "completed_sequence_rewarded","completed_sequence_unrewarded","completed_sequence_reward_timeout",
        "completed_sequences_HR","completed_sequence_HR_rewarded","completed_sequence_HR_unrewarded","completed_sequence_HR_reward_timeout",
        "completed_sequences_HR_missed","completed_sequence_HR_missed_rewarded","completed_sequence_HR_missed_unrewarded","completed_sequence_HR_missed_reward_timeout",
        "aborted_sequences","aborted_sequences_HR","aborted_sequences_detailed", "non_initiated_FA",
    ]
    results = {}
    for t in tables:
        f = results_dir / f"{t}.csv"
        if f.exists():
            results[t] = pd.read_csv(f)
        else:
            results[t] = pd.DataFrame()

    # Attach manifest and summary
    results["manifest"] = manifest
    results["summary"] = summary

    return results

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
    # --- Derive subjid, date, and output_dir from results ---
    base_dir = Path("/Volumes/harris/hypnose/derivatives")
    manifest = results.get("manifest", {})

    summary = results.get("summary", {})
    # Try to get subjid and date from manifest or summary
    subjid = (
        manifest.get("session", {}).get("subject_id")
        or summary.get("session", {}).get("subject_id")
        or manifest.get("session", {}).get("subjid")
        or summary.get("session", {}).get("subjid")
        or None
    )
    date = (
        manifest.get("session", {}).get("date")
        or summary.get("session", {}).get("date")
        or manifest.get("session", {}).get("session_date")
        or summary.get("session", {}).get("session_date")
        or None
    )
    # Try to get output_dir from manifest paths
    paths = manifest.get("paths", {})

    # Override paths with the local base_dir
    if base_dir is not None:
        base_dir = Path(base_dir)
        if "rawdata_dir" in paths:
            paths["rawdata_dir"] = str(base_dir / "rawdata")
        if "results_dir" in manifest:
            manifest["results_dir"] = str(base_dir / "derivatives" / paths.get("sub_folder", "") / paths.get("ses_folder", "") / "saved_analysis_results")

    out_dir = None
    if "rawdata_dir" in paths:
        # Ensure 'derivatives' is not duplicated
        rawdata_parent = Path(paths["rawdata_dir"]).parent
        if rawdata_parent.name == "derivatives":
            out_dir = rawdata_parent / paths.get("sub_folder", "") / paths.get("ses_folder", "") / "saved_analysis_results"
        else:
            out_dir = rawdata_parent / "derivatives" / paths.get("sub_folder", "") / paths.get("ses_folder", "") / "saved_analysis_results"
    elif "results_dir" in manifest:
        out_dir = Path(manifest["results_dir"])
    else:
        # fallback: use current working directory
        out_dir = Path.cwd()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Run metrics and capture output ---
    metrics = {}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("\n--- Decision Accuracy ---")
        metrics['decision_accuracy'] = decision_accuracy(results)
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

def merged_results_output_dir(subjids, dates, protocol, derivatives_dir=Path("/Volumes/harris/hypnose/derivatives")):
    """
    Determine the output directory for merged results based on subjids, dates, and protocol.
    """
    subjids = sorted(set(str(s) for s in subjids))
    dates = sorted(set(str(d) for d in dates))
    if len(subjids) == 1:
        # Single subject: save in that subject's folder under merged_results
        sub_str = f"sub-{str(subjids[0]).zfill(3)}"
        subj_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
        if not subj_dirs:
            raise FileNotFoundError(f"No subject directory found for {sub_str}")
        subj_dir = subj_dirs[0]
        merged_dir = subj_dir / "merged_results"
    else:
        # Multiple subjects: save in derivatives/merged/(protocol_merged|merged)
        merged_dir = derivatives_dir / "merged"
        if protocol:
            merged_dir = merged_dir / "protocol_merged"
        else:
            merged_dir = merged_dir / "merged"
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
    derivatives_dir = Path("/Volumes/harris/hypnose/derivatives")
    results = []
    results_dicts = []

    # Find all subject directories
    subj_dirs = list(derivatives_dir.glob("sub-*_id-*")) if subjids is None else [
        d for subjid in subjids
        for d in derivatives_dir.glob(f"sub-{str(subjid).zfill(3)}_id-*")
    ]
    if verbose:
        print(f"Found {len(subj_dirs)} subject directories.")

    for subj_dir in subj_dirs:
        subj_results = []  # Store results for this subject
        subj_dates = []  # Track processed dates for this subject

        # Find all session directories for this subject
        ses_dirs = list(subj_dir.glob("ses-*_date-*")) if dates is None else [
            d for date in dates for d in subj_dir.glob(f"ses-*_date-{date}")
        ]
        if not ses_dirs:
            continue
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            summary_path = results_dir / "summary.json"
            if not summary_path.exists():
                continue
            # Protocol filter
            if protocol is not None:
                try:
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    runs = summary.get("session", {}).get("runs", [])
                    if not runs or "stage" not in runs[0]:
                        continue
                    stage_name = runs[0]["stage"].get("stage_name", "")
                    if protocol not in stage_name:
                        continue
                except Exception as e:
                    if verbose:
                        print(f"Skipping {summary_path}: {e}")
                    continue
            # Extract subjid and date from path or summary
            subjid = subj_dir.name.split("_")[0].replace("sub-", "")
            date = ses_dir.name.split("_date-")[-1]
            # Run metrics
            try:
                session_results = load_session_results(subjid, date)
                metrics = run_all_metrics(
                    session_results,
                    save_txt=save_txt,
                    save_json=save_json
                )
                subj_results.append(session_results)  # Collect results for this subject
                subj_dates.append(date)  # Track processed dates for this subject
                results_dicts.append(session_results)  # Add to global results
                if verbose:
                    print(f"Processed subjid={subjid}, date={date}")
            except Exception as e:
                if verbose:
                    print(f"Failed for subjid={subjid}, date={date}: {e}")

        # --- Merge results for this subject ---
        if subj_results:
            pooled_results = pool_results_dicts(subj_results)
            # --- Capture pretty print output ---
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                merged_metrics = run_all_metrics(pooled_results, save_txt=False, save_json=False)
            pretty_print_str = buffer.getvalue()
            print(pretty_print_str)
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
        print(pretty_print_str)
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

    return results

# ================== Behavioral Metrics Functions =================

def decision_accuracy(results):
    comp_rew = results.get("completed_sequence_rewarded", pd.DataFrame())
    comp_unr = results.get("completed_sequence_unrewarded", pd.DataFrame())
    n_rew = len(comp_rew)
    n_unr = len(comp_unr)
    denom = n_rew + n_unr
    print(f"Decision Accuracy: {n_rew}/{denom} = {n_rew/denom if denom>0 else np.nan:.3f}")
    return n_rew, denom, n_rew / denom if denom > 0 else np.nan

def premature_response_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    n_fa = (ab_det["fa_label"] != "nFA").sum() if not ab_det.empty and "fa_label" in ab_det.columns else 0
    n_total = len(ab_det)
    print(f"Premature Response Rate: {n_fa}/{n_total} = {n_fa/n_total if n_total>0 else np.nan:.3f}")
    return n_fa, n_total, n_fa / n_total if n_total > 0 else np.nan

def response_contingent_FA_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    comp_rew = results.get("completed_sequence_rewarded", pd.DataFrame())
    comp_unr = results.get("completed_sequence_unrewarded", pd.DataFrame())
    n_fa = (ab_det["fa_label"] != "nFA").sum() if not ab_det.empty and "fa_label" in ab_det.columns else 0
    denom = n_fa + len(comp_rew) + len(comp_unr)
    print(f"Response-Contingent False Alarm Rate: {n_fa}/{denom} = {n_fa/denom if denom>0 else np.nan:.3f}")
    return n_fa, denom, n_fa / denom if denom > 0 else np.nan

def global_FA_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    ini = results.get("initiated_sequences", pd.DataFrame())
    n_fa = (ab_det["fa_label"] != "nFA").sum() if not ab_det.empty and "fa_label" in ab_det.columns else 0
    n_ini = len(ini)
    print(f"Global False Alarm Rate: {n_fa}/{n_ini} = {n_fa/n_ini if n_ini>0 else np.nan:.3f}")
    return n_fa, n_ini, n_fa / n_ini if n_ini > 0 else np.nan

def FA_odor_bias(results):
    print("FA Odor Bias for FA Time In:")
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    if ab_det.empty or "fa_label" not in ab_det.columns or "last_odor_name" not in ab_det.columns:
        return {'bias': {}, 'n_fa': {}, 'n_ab': {}, 'total_fa': 0, 'total_ab': 0}
    fa_mask = ab_det["fa_label"] == "FA_time_in"
    odors = sorted(ab_det["last_odor_name"].dropna().unique())
    bias = {}
    n_fa = {}
    n_ab = {}
    total_fa = fa_mask.sum()
    total_ab = len(ab_det)
    for od in odors:
        fa_at_od = fa_mask & (ab_det["last_odor_name"] == od)
        n_fa_od = fa_at_od.sum()
        n_ab_od = (ab_det["last_odor_name"] == od).sum()
        n_fa[od] = n_fa_od
        n_ab[od] = n_ab_od
        bias[od] = (n_fa_od / n_ab_od) / (total_fa / total_ab) if n_ab_od > 0 and total_ab > 0 and total_fa > 0 else np.nan
        print(f"{od}: {n_fa_od}/{n_ab_od} FA, Bias: {bias[od]:.3f}")
    return {'bias': bias, 'n_fa': n_fa, 'n_ab': n_ab, 'total_fa': total_fa, 'total_ab': total_ab}

def FA_position_bias(results):
    print("FA Position Bias for FA Time In:")
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    if ab_det.empty or "fa_label" not in ab_det.columns or "last_odor_position" not in ab_det.columns:
        return pd.Series(dtype=float)
    fa_mask = ab_det["fa_label"] == "FA_time_in" # use != "nFA" for all FA types
    positions = sorted(ab_det["last_odor_position"].dropna().unique())
    bias = {}
    total_fa = fa_mask.sum()
    total_ab = len(ab_det)
    for pos in positions:
        fa_at_pos = fa_mask & (ab_det["last_odor_position"] == pos)
        n_fa_pos = fa_at_pos.sum()
        n_ab_pos = (ab_det["last_odor_position"] == pos).sum()
        bias[pos] = (n_fa_pos / n_ab_pos) / (total_fa / total_ab) if n_ab_pos > 0 and total_ab > 0 and total_fa > 0 else np.nan
        print(f"Position {pos}: {n_fa_pos}/{n_ab_pos} FA, Bias: {bias[pos]:.3f}")
    return pd.Series(bias).sort_index()

def sequence_completion_rate(results):
    comp = results.get("completed_sequences", pd.DataFrame())
    ini = results.get("initiated_sequences", pd.DataFrame())
    print(f"Sequence Completion Rate: {len(comp)}/{len(ini)} = {len(comp)/len(ini) if len(ini)>0 else np.nan:.3f}")
    return len(comp), len(ini), len(comp) / len(ini) if len(ini) > 0 else np.nan

def odorx_abortion_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    comp = results.get("completed_sequences", pd.DataFrame())
    if ab_det.empty or "presentations" not in ab_det.columns or "last_odor_name" not in ab_det.columns:
        return pd.Series(dtype=float)

    # Numerator: abortions at odor X (last odor)
    abortions = ab_det["last_odor_name"].value_counts().to_dict()

    # Denominator: all presentations of odor X (aborted + completed)
    presentations = {}
    # Aborted trials
    for _, row in ab_det.iterrows():
        pres_list = parse_json_column(row.get("presentations", []))
        if isinstance(pres_list, list):
            for pres in pres_list:
                od = pres.get("odor_name")
                if od is not None:
                    presentations[od] = presentations.get(od, 0) + 1
    # Completed trials
    for _, row in comp.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        if isinstance(ppt, dict):
            for pos, info in ppt.items():
                od = info.get("odor_name")
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
    # Numerator: rewarded completed HR trials
    n_hr_rewarded = len(results.get("completed_sequence_HR_rewarded", pd.DataFrame()))
    # Denominator: any trial with HR presentation
    denom = (
        len(results.get("completed_sequence_HR_rewarded", pd.DataFrame())) +
        len(results.get("completed_sequence_HR_unrewarded", pd.DataFrame())) +
        len(results.get("completed_sequence_HR_reward_timeout", pd.DataFrame())) +
        len(results.get("completed_sequences_HR_missed", pd.DataFrame())) +
        len(results.get("aborted_sequences_HR", pd.DataFrame()))
    )
    print(f"Hidden Rule Performance: {n_hr_rewarded}/{denom} = {n_hr_rewarded/denom if denom>0 else np.nan:.3f}")
    return n_hr_rewarded, denom, n_hr_rewarded / denom if denom > 0 else np.nan

def hidden_rule_detection_rate(results):
    # Numerator: completed HR trials (rewarded, unrewarded, timeout) at HR position
    n_hr_completed = (
        len(results.get("completed_sequence_HR_rewarded", pd.DataFrame())) +
        len(results.get("completed_sequence_HR_unrewarded", pd.DataFrame())) +
        len(results.get("completed_sequence_HR_reward_timeout", pd.DataFrame()))
    )
    # Denominator: any trial with HR presentation
    denom = (
        len(results.get("completed_sequence_HR_rewarded", pd.DataFrame())) +
        len(results.get("completed_sequence_HR_unrewarded", pd.DataFrame())) +
        len(results.get("completed_sequence_HR_reward_timeout", pd.DataFrame())) +
        len(results.get("completed_sequences_HR_missed", pd.DataFrame())) +
        len(results.get("aborted_sequences_HR", pd.DataFrame()))
    )
    print(f"Hidden Rule Detection Rate: {n_hr_completed}/{denom} = {n_hr_completed/denom if denom>0 else np.nan:.3f}")
    return n_hr_completed, denom, n_hr_completed / denom if denom > 0 else np.nan

def choice_timeout_rate(results):
    comp_tmo = results.get("completed_sequence_reward_timeout", pd.DataFrame())
    comp = results.get("completed_sequences", pd.DataFrame())
    print(f"Choice Timeout Rate: {len(comp_tmo)}/{len(comp)} = {len(comp_tmo)/len(comp) if len(comp)>0 else np.nan:.3f}")
    return len(comp_tmo), len(comp), len(comp_tmo) / len(comp) if len(comp) > 0 else np.nan

def avg_sampling_time_odor_x(results):
    comp = results.get("completed_sequences", pd.DataFrame())
    if comp.empty:
        return pd.Series(dtype=float)
    odor_times = {}
    for _, row in comp.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        for pos, info in ppt.items():
            od = info.get("odor_name")
            poke_ms = info.get("poke_time_ms")
            if od is not None and poke_ms is not None:
                odor_times.setdefault(od, []).append(poke_ms)
    avg_times = pd.Series({od: np.mean(times) for od, times in odor_times.items()}).sort_index()
    for odor, avg_time in avg_times.items():
        print(f"{odor} Average Sampling Time: {avg_time:.2f} ms")

    return avg_times

def avg_sampling_time_completed_sequence(results):
    comp = results.get("completed_sequences", pd.DataFrame())
    total_time = 0.0
    total_presentations = 0
    for _, row in comp.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        for pos, info in ppt.items():
            poke_ms = info.get("poke_time_ms")
            if poke_ms is not None:
                total_time += poke_ms
                total_presentations += 1
    print(f"Average Sampling Time (Completed Sequences): {total_time/total_presentations if total_presentations>0 else np.nan:.2f} ms")
    return total_time / total_presentations if total_presentations > 0 else np.nan

def avg_sampling_time_aborted_sequence(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    total_time = 0.0
    total_presentations = 0
    if ab_det.empty or "presentations" not in ab_det.columns or "last_event_index" not in ab_det.columns:
        return np.nan
    for _, row in ab_det.iterrows():
        pres_list = parse_json_column(row.get("presentations", []))
        last_idx = row.get("last_event_index")
        if isinstance(pres_list, list):
            for pres in pres_list:
                idx = pres.get("index_in_trial")
                poke_ms = pres.get("poke_time_ms")
                if idx != last_idx and poke_ms is not None:
                    total_time += poke_ms
                    total_presentations += 1
    print(f"Average Sampling Time (Aborted Sequences): {total_time/total_presentations if total_presentations>0 else np.nan:.2f} ms")
    return total_time / total_presentations if total_presentations > 0 else np.nan

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
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    comp = results.get("completed_sequences", pd.DataFrame())
    comp_hr_missed = results.get("completed_sequences_HR_missed", pd.DataFrame())
    if ab_det.empty or "last_odor_position" not in ab_det.columns:
        return pd.Series(dtype=float)

    # Numerator: abortions at each position
    abortions = ab_det["last_odor_position"].value_counts().to_dict()

    # Denominator: number of trials that reached each position
    reached = {}

    # Aborted trials: count all positions up to and including the abortion position
    for _, row in ab_det.iterrows():
        last_pos = row.get("last_odor_position")
        if pd.notnull(last_pos):
            for pos in range(1, int(last_pos) + 1):
                reached[pos] = reached.get(pos, 0) + 1

    # Completed HR-missed: all 5 positions reached
    for _, row in comp_hr_missed.iterrows():
        for pos in range(1, 6):
            reached[pos] = reached.get(pos, 0) + 1

    # Completed (not HR-missed): only positions up to and including HR position
    for _, row in comp.iterrows():
        ppt = parse_json_column(row.get("position_poke_times", {}))
        if isinstance(ppt, dict) and ppt:
            max_pos = max([int(pos) for pos in ppt.keys() if str(pos).isdigit()])
            for pos in range(1, max_pos + 1):
                reached[pos] = reached.get(pos, 0) + 1
    # Calculate rates
    all_positions = sorted(set(list(abortions.keys()) + list(reached.keys())))
    rates = {}
    for pos in all_positions:
        n_ab = abortions.get(pos, 0)
        n_reached = reached.get(pos, 0)
        rates[pos] = n_ab / n_reached if n_reached > 0 else np.nan
        print(f"Position {pos}: {n_ab}/{n_reached} abortions, Rate: {rates[pos]:.3f}")
    return pd.Series(rates).sort_index()

def avg_response_time(results):
    comp_rt = results.get("completed_sequences_with_response_times", pd.DataFrame())
    if comp_rt.empty or "response_time_ms" not in comp_rt.columns or "response_time_category" not in comp_rt.columns:
        print("No response time data available.")
        return {}

    out = {}
    def print_avg(df, label):
        s = pd.to_numeric(df["response_time_ms"], errors="coerce").dropna()
        avg = s.mean() if not s.empty else np.nan
        print(f"{label}: {avg:.1f} ms (n={len(s)})")
        return float(avg) if not np.isnan(avg) else np.nan

    # Rewarded
    rew = comp_rt[comp_rt["response_time_category"] == "rewarded"]
    out["Rewarded"] = print_avg(rew, "Rewarded")

    # Unrewarded
    unr = comp_rt[comp_rt["response_time_category"] == "unrewarded"]
    out["Unrewarded"] = print_avg(unr, "Unrewarded")

    # Reward Timeout
    tmo = comp_rt[comp_rt["response_time_category"] == "timeout_delayed"]
    out["Reward Timeout"] = print_avg(tmo, "Reward Timeout")

    # Rewarded + Unrewarded (excluding timeouts)
    both = comp_rt[comp_rt["response_time_category"].isin(["rewarded", "unrewarded"])]
    out["Average Response Time (Rewarded + Unrewarded)"] = print_avg(both, "Average Response Time (Rewarded + Unrewarded)")

    return out

def FA_avg_response_times(results):
    out = {}
    # Non-initiated FA
    fa_noninit_df = results.get("non_initiated_FA", pd.DataFrame())
    if not fa_noninit_df.empty and "fa_label" in fa_noninit_df.columns and "fa_latency_ms" in fa_noninit_df.columns:
        print("Non-Initiated FA Response Times:")
        for label, pretty in [
            ("FA_time_in", "FA Time In"),
            ("FA_time_out", "FA Time Out"),
            ("FA_late", "FA Late"),
        ]:
            s = pd.to_numeric(fa_noninit_df.loc[fa_noninit_df["fa_label"] == label, "fa_latency_ms"], errors="coerce").dropna()
            avg = s.mean() if not s.empty else np.nan
            print(f"  - {pretty}: avg={avg:.1f} ms, median={s.median():.1f} ms, n={len(s)}" if not s.empty else f"  - {pretty}: n=0")
            out[f"Non-Initiated {pretty}"] = float(avg) if not np.isnan(avg) else np.nan
        print()

    # Aborted trials FA
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    if not ab_det.empty and "fa_label" in ab_det.columns and "fa_latency_ms" in ab_det.columns:
        print("Aborted Trials FA Response Times:")
        for label, pretty in [
            ("FA_time_in", "FA Time In"),
            ("FA_time_out", "FA Time Out"),
            ("FA_late", "FA Late"),
        ]:
            s = pd.to_numeric(ab_det.loc[ab_det["fa_label"] == label, "fa_latency_ms"], errors="coerce").dropna()
            avg = s.mean() if not s.empty else np.nan
            print(f"  - {pretty}: avg={avg:.1f} ms, median={s.median():.1f} ms, n={len(s)}" if not s.empty else f"  - {pretty}: n=0")
            out[f"Aborted {pretty}"] = float(avg) if not np.isnan(avg) else np.nan
        print()
    return out

def response_rate(results):
    comp = results.get("completed_sequences", pd.DataFrame())
    comp_tmo = results.get("completed_sequence_reward_timeout", pd.DataFrame())
    n_resp = len(comp) - len(comp_tmo)
    print(f"Response Rate: {n_resp}/{len(comp)} = {n_resp/len(comp) if len(comp)>0 else np.nan:.3f}")
    return n_resp, len(comp), n_resp / len(comp) if len(comp) > 0 else np.nan

def manual_vs_auto_stop_preference(results):
    comp = results.get("completed_sequences", pd.DataFrame())
    short = 0
    long = 0
    for _, row in comp.iterrows():
        vts = parse_json_column(row.get("position_valve_times", {}))
        for pos, info in vts.items():
            dur = info.get("valve_duration_ms")
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
    n_fa = (fa_noninit_df["fa_label"] != "nFA").sum()
    print(f"Non-Initiated FA Rate: {n_fa}/{len(fa_noninit_df)} = {n_fa/len(fa_noninit_df) if len(fa_noninit_df)>0 else np.nan:.3f}")
    return n_fa, len(fa_noninit_df), n_fa / len(fa_noninit_df) if len(fa_noninit_df) > 0 else np.nan

def non_initiation_odor_bias(results):
    non_ini = results.get("non_initiated_sequences", pd.DataFrame())
    pos1 = results.get("non_initiated_odor1_attempts", pd.DataFrame())
    ini = results.get("initiated_sequences", pd.DataFrame())

    # Only consider first odor attempts
    non_ini = non_ini[non_ini["odor_position"] == 1] if "odor_position" in non_ini.columns else non_ini
    pos1 = pos1[pos1["odor_position"] == 1] if "odor_position" in pos1.columns else pos1

    # Numerator: non-initiated trials with this odor as first odor
    all_non_init = pd.concat([non_ini, pos1], ignore_index=True)
    count_odors = all_non_init["odor_name"].value_counts()

    # Denominator: all trials with this odor as first odor (initiated or not)
    # Get from both initiated and non-initiated
    first_odors = []

    # Initiated
    for idx, row in ini.iterrows():
        presentations = row.get("presentations", [])
        if isinstance(presentations, str):
            import json
            try:
                presentations = json.loads(presentations.replace("'", '"'))
            except Exception:
                presentations = []
        if presentations and isinstance(presentations, list):
            for pres in presentations:
                if pres.get("position") == 1:
                    first_odors.append(pres.get("odor_name"))
                    break

    # Non-initiated (baseline and pos1)
    for df in [non_ini, pos1]:
        if not df.empty and "odor_name" in df.columns:
            first_odors.extend(df["odor_name"].dropna().tolist())

    total_first_odors = pd.Series(first_odors).value_counts()

    # Global rates for normalization
    total_noninit = len(all_non_init)
    total_trials = len(total_first_odors) and total_first_odors.sum() or 0
    global_rate = total_noninit / total_trials if total_trials > 0 else np.nan

    # Calculate bias for each odor
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
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    if ab_det.empty or "last_odor_name" not in ab_det.columns or "abortion_type" not in ab_det.columns:
        return pd.Series(dtype=float)
    odors = ab_det["last_odor_name"].dropna().unique()
    bias = {}
    total_init = (ab_det["abortion_type"] == "initiation_abortion").sum()
    total_ab = len(ab_det)
    for od in sorted(odors):
        n_init_od = ((ab_det["last_odor_name"] == od) & (ab_det["abortion_type"] == "initiation_abortion")).sum()
        n_ab_od = (ab_det["last_odor_name"] == od).sum()
        bias[od] = (n_init_od / n_ab_od) / (total_init / total_ab) if n_ab_od > 0 and total_ab > 0 and total_init > 0 else np.nan
        print(f"{od}: {n_init_od}/{n_ab_od} initiation abortions, Bias: {bias[od]:.3f}")
    return pd.Series(bias).sort_index()

def fa_abortion_stats(results, return_df=False):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    if ab_det.empty:
        print("No aborted_sequences_detailed data.")
        return

    subtype_labels = [
        ("FA_time_in", "FA Time In"),
        ("FA_time_out", "FA Time Out"),
        ("FA_late", "FA Late"),
    ]

    # Odor+Position table
    rows = []
    odors = sorted(ab_det["last_odor_name"].dropna().unique())
    positions = sorted(ab_det["last_odor_position"].dropna().unique())
    for odor in odors:
        for pos in positions:
            df = ab_det[(ab_det["last_odor_name"] == odor) & (ab_det["last_odor_position"] == pos)]
            n_total = len(df)
            if n_total == 0:
                continue
            fa_labels = df["fa_label"].fillna("").astype(str).str.strip()
            row = {
                "Odor": odor,
                "Position": pos,
                "Total Abortions": n_total,
            }
            n_fa = fa_labels.isin([s[0] for s in subtype_labels]).sum()
            row["FA Abortion Rate"] = f"{n_fa}/{n_total} ({n_fa/n_total:.2f})"
            for subtype, pretty in subtype_labels:
                count = (fa_labels == subtype).sum()
                row[pretty] = f"{count} ({count/n_total:.2f})"
            rows.append(row)
    df_out = pd.DataFrame(rows)

    # Per-odor table
    odor_rows = []
    for odor in odors:
        df = ab_det[ab_det["last_odor_name"] == odor]
        n_total = len(df)
        if n_total == 0:
            continue
        fa_labels = df["fa_label"].fillna("").astype(str).str.strip()
        row = {
            "Odor": odor,
            "Total Abortions": n_total,
        }
        n_fa = fa_labels.isin([s[0] for s in subtype_labels]).sum()
        row["FA Abortion Rate"] = f"{n_fa}/{n_total} ({n_fa/n_total:.2f})"
        for subtype, pretty in subtype_labels:
            count = (fa_labels == subtype).sum()
            row[pretty] = f"{count} ({count/n_total:.2f})"
        odor_rows.append(row)
    df_odor = pd.DataFrame(odor_rows)

    # Per-position table
    pos_rows = []
    for pos in positions:
        df = ab_det[ab_det["last_odor_position"] == pos]
        n_total = len(df)
        if n_total == 0:
            continue
        fa_labels = df["fa_label"].fillna("").astype(str).str.strip()
        row = {
            "Position": pos,
            "Total Abortions": n_total,
        }
        n_fa = fa_labels.isin([s[0] for s in subtype_labels]).sum()
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


# ================== Loading saved metrics and Visualization =================

def _extract_metric_value(metrics: dict, var_path: str):
    """
    Extract a numeric value from metrics dict given a dot-path.
    Examples:
      - "decision_accuracy" -> uses the 3rd element (value) if tuple/list (num, denom, value)
      - "avg_response_time.Rewarded" -> nested dict lookup
    Returns float or np.nan if not found/unsupported.
    """
    try:
        parts = var_path.split(".")
        cur = metrics.get(parts[0], None)
        for p in parts[1:]:
            if isinstance(cur, dict):
                cur = cur.get(p, None)
            else:
                # unsupported path deeper into non-dict
                return float("nan")
        # Resolve final value
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            return float(cur)
        if isinstance(cur, (list, tuple)) and len(cur) >= 3:
            # assume (numerator, denominator, value)
            val = cur[2]
            return float(val) if val is not None else float("nan")
        # Some dicts may hold numbers directly keyed by categories (needs explicit subkey in var_path)
        return float(cur) if isinstance(cur, (int, float)) else float("nan")
    except Exception:
        return float("nan")
    
def _iter_subject_dirs(derivatives_dir: Path, subjids: Optional[Iterable[int]]):
    if subjids is None:
        for d in sorted(derivatives_dir.glob("sub-*_id-*")):
            name = d.name.split("_")[0].replace("sub-", "")
            try:
                yield int(name), d
            except Exception:
                continue
    else:
        for sid in subjids:
            sub_str = f"sub-{str(int(sid)).zfill(3)}"
            dirs = sorted(derivatives_dir.glob(f"{sub_str}_id-*"))
            if dirs:
                yield int(sid), dirs[0]

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

def _load_protocol_from_summary(results_dir: Path) -> str:
    try:
        with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        runs = summary.get("session", {}).get("runs", [])
        if runs and isinstance(runs, list):
            stage = runs[0].get("stage", {}) if isinstance(runs[0], dict) else {}
            name = stage.get("stage_name") or stage.get("name")
            return str(name) if name else "Unknown"
    except Exception:
        pass
    return "Unknown"

def _ensure_metrics_json(subjid: int, date: Union[int, str], results_dir: Path, compute_if_missing: bool) -> Optional[dict]:
    """
    Return metrics dict by loading metrics_{subjid}_{date}.json if present,
    else compute via run_all_metrics if compute_if_missing=True.
    """
    subjid_i = int(subjid)
    date_s = str(date)
    metrics_path = results_dir / f"metrics_{subjid_i}_{date_s}.json"
    if metrics_path.exists():
        try:
            return json.load(open(metrics_path, "r", encoding="utf-8"))
        except Exception:
            pass
    if compute_if_missing:
        try:
            session_results = load_session_results(subjid_i, date_s)
            return run_all_metrics(session_results, save_txt=True, save_json=True)
        except Exception:
            return None
    return None

def plot_behavior_metrics(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    variables: Optional[Iterable[str]] = None,
    *,
    protocol_filter: Optional[str] = None,
    compute_if_missing: bool = False,
    verbose: bool = True
):
    """
    Plot selected metrics over sessions for one or more subjects.

    - X-axis: union of all available dates across selected subjects (categorical, no time gaps).
    - Y-axis: metric value.
    - One figure per variable.
    - Marker shape encodes subject; dot color encodes protocol; connecting lines are thin black.
    - Protocol filtering optional (substring match).
    - Values are read from metrics_{subjid}_{date}.json; if missing and compute_if_missing=True, metrics are computed.

    Parameters:
    - subjids: List of subject IDs to include, or None to include all subjects with matching dates.
    - dates: List of specific dates (e.g., [20250101, 20250102]) or a date range (e.g., (20250101, 20250202)).
    - variables: List of metric names or dot-paths to plot.
    - protocol_filter: Optional substring to filter sessions by protocol.
    - compute_if_missing: If True, compute metrics if missing.
    - verbose: If True, print progress and warnings.

    Returns:
    - List of matplotlib Figure objects.
    """
    if not variables:
        raise ValueError("Please provide `variables` (list of metric names or dot-paths).")

    rows = []
    derivatives_dir = Path("/Volumes/harris/hypnose/derivatives")

    # Gather sessions
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses in ses_dirs:
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            # Protocol (for coloring)
            protocol = _load_protocol_from_summary(results_dir)
            if protocol_filter and (protocol is None or protocol_filter not in str(protocol)):
                continue
            # Load metrics (json or compute)
            metrics = _ensure_metrics_json(sid, date_str, results_dir, compute_if_missing)
            if metrics is None:
                if verbose:
                    print(f"[plot_behavior_metrics] Skipping sub-{sid:03d} {date_str}: metrics JSON missing.")
                continue
            for var in variables:
                val = _extract_metric_value(metrics, var)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    rows.append({
                        "subjid": int(sid),
                        "date": int(date_str) if str(date_str).isdigit() else date_str,
                        "date_str": str(date_str),
                        "protocol": str(protocol) if protocol else "Unknown",
                        "variable": var,
                        "value": float(val)
                    })

    if not rows:
        if verbose:
            print("[plot_behavior_metrics] No data found for the given filters.")
        return []

    df = pd.DataFrame(rows)
    # Union of all dates across all subjects
    unique_dates = sorted(df["date"].unique())
    date_to_x = {d: i for i, d in enumerate(unique_dates)}
    df["x"] = df["date"].map(date_to_x)

    # Subject -> marker mapping
    markers_cycle = ['o', '^', 's', 'X', 'D', 'P', 'v', '>', '<', '*', 'h', 'H', '8', 'p', 'x']
    unique_subj = sorted(df["subjid"].unique())
    subj_to_marker = {sid: markers_cycle[i % len(markers_cycle)] for i, sid in enumerate(unique_subj)}

    # Protocol -> color mapping
    unique_protocols = [p for p in sorted(df["protocol"].unique()) if p and p != "Unknown"]
    # Add Unknown at end if present
    if "Unknown" in df["protocol"].unique():
        unique_protocols.append("Unknown")
    cmap = cm.get_cmap("tab10", max(10, len(unique_protocols)))
    prot_to_color = {p: cmap(i % cmap.N) for i, p in enumerate(unique_protocols)}
    if "Unknown" in prot_to_color:
        prot_to_color["Unknown"] = (0.6, 0.6, 0.6, 1.0)

    figs = []
    # One plot per variable
    for var in variables:
        df_var = df[df["variable"] == var]
        if df_var.empty:
            if verbose:
                print(f"[plot_behavior_metrics] No data for variable '{var}'.")
            continue

        fig, ax = plt.subplots(figsize=(10, 4.5))
        # Plot each subject: black connecting line + colored markers per protocol
        for sid in unique_subj:
            dsub = df_var[df_var["subjid"] == sid].sort_values("x")
            if dsub.empty:
                continue
            ax.plot(dsub["x"], dsub["value"], color="black", linewidth=1.0, alpha=0.8, zorder=1)
            # Scatter with subject marker and protocol color
            colors = dsub["protocol"].map(lambda p: prot_to_color.get(p, (0.6, 0.6, 0.6, 1.0)))
            ax.scatter(
                dsub["x"], dsub["value"],
                c=list(colors),
                marker=subj_to_marker[sid],
                edgecolors="black",
                linewidths=0.5,
                s=55,
                zorder=2,
                label=f"sub-{sid:03d}"
            )

        # X axis: categorical dates
        ax.set_xticks(range(len(unique_dates)))
        ax.set_xticklabels([str(d) for d in unique_dates], rotation=45, ha="right")
        ax.set_xlim(-0.5, len(unique_dates) - 0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel(var.replace("_", " ").title())
        ax.set_title(f"{var}")

        # Grid
        ax.grid(True, which="both", axis="both", alpha=0.25)

        # Build separate legends: subjects (markers) and protocols (colors)
        subject_handles = [
            Line2D([0], [0],
                   marker=subj_to_marker[sid],
                   color="black", linestyle="",
                   markerfacecolor="white",
                   markeredgecolor="black",
                   markersize=7,
                   label=f"sub-{sid:03d}")
            for sid in unique_subj
        ]
        protocol_handles = [
            Line2D([0], [0],
                   marker='o',
                   color='none', linestyle="",
                   markerfacecolor=prot_to_color[p],
                   markeredgecolor="black",
                   markersize=7,
                   label=p)
            for p in unique_protocols
        ]

        # Place legends
        if subject_handles:
            leg1 = ax.legend(handles=subject_handles, title="Subjects", loc="upper left", bbox_to_anchor=(1.02, 1.0))
            ax.add_artist(leg1)
        if protocol_handles:
            ax.legend(handles=protocol_handles, title="Protocols", loc="lower left", bbox_to_anchor=(1.02, 0.0))

        plt.tight_layout()
        figs.append(fig)

    return figs