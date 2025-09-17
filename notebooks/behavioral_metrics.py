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
# ================== Behavioral Metrics Functions =================

def decision_accuracy(results):
    comp_rew = results.get("completed_sequence_rewarded", pd.DataFrame())
    comp_unr = results.get("completed_sequence_unrewarded", pd.DataFrame())
    n_rew = len(comp_rew)
    n_unr = len(comp_unr)
    denom = n_rew + n_unr
    print(f"Decision Accuracy: {n_rew}/{denom} = {n_rew/denom if denom>0 else np.nan:.3f}")
    return n_rew / denom if denom > 0 else np.nan

def premature_response_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    n_fa = (ab_det["fa_label"] != "nFA").sum() if not ab_det.empty and "fa_label" in ab_det.columns else 0
    n_total = len(ab_det)
    print(f"Premature Response Rate: {n_fa}/{n_total} = {n_fa/n_total if n_total>0 else np.nan:.3f}")
    return n_fa / n_total if n_total > 0 else np.nan

def response_contingent_FA_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    comp_rew = results.get("completed_sequence_rewarded", pd.DataFrame())
    comp_unr = results.get("completed_sequence_unrewarded", pd.DataFrame())
    n_fa = (ab_det["fa_label"] != "nFA").sum() if not ab_det.empty and "fa_label" in ab_det.columns else 0
    denom = n_fa + len(comp_rew) + len(comp_unr)
    print(f"Response-Contingent False Alarm Rate: {n_fa}/{denom} = {n_fa/denom if denom>0 else np.nan:.3f}")
    return n_fa / denom if denom > 0 else np.nan

def global_FA_rate(results):
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    ini = results.get("initiated_sequences", pd.DataFrame())
    n_fa = (ab_det["fa_label"] != "nFA").sum() if not ab_det.empty and "fa_label" in ab_det.columns else 0
    n_ini = len(ini)
    print(f"Global False Alarm Rate: {n_fa}/{n_ini} = {n_fa/n_ini if n_ini>0 else np.nan:.3f}")
    return n_fa / n_ini if n_ini > 0 else np.nan

def FA_odor_bias(results):
    print("FA Odor Bias for FA Time In:")
    ab_det = results.get("aborted_sequences_detailed", pd.DataFrame())
    if ab_det.empty or "fa_label" not in ab_det.columns or "last_odor_name" not in ab_det.columns:
        return pd.Series(dtype=float)
    fa_mask = ab_det["fa_label"] == "FA_time_in" # use != "nFA" for all FA types
    odors = sorted(ab_det["last_odor_name"].dropna().unique())
    bias = {}
    total_fa = fa_mask.sum()
    total_ab = len(ab_det)
    for od in odors:
        fa_at_od = fa_mask & (ab_det["last_odor_name"] == od)
        n_fa_od = fa_at_od.sum()
        n_ab_od = (ab_det["last_odor_name"] == od).sum()
        bias[od] = (n_fa_od / n_ab_od) / (total_fa / total_ab) if n_ab_od > 0 and total_ab > 0 and total_fa > 0 else np.nan
        print(f"{od}: {n_fa_od}/{n_ab_od} FA, Bias: {bias[od]:.3f}")
    return pd.Series(bias).sort_index()

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
    return len(comp) / len(ini) if len(ini) > 0 else np.nan

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
    return n_hr_rewarded / denom if denom > 0 else np.nan

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
    return n_hr_completed / denom if denom > 0 else np.nan

def choice_timeout_rate(results):
    comp_tmo = results.get("completed_sequence_reward_timeout", pd.DataFrame())
    comp = results.get("completed_sequences", pd.DataFrame())
    print(f"Choice Timeout Rate: {len(comp_tmo)}/{len(comp)} = {len(comp_tmo)/len(comp) if len(comp)>0 else np.nan:.3f}")
    return len(comp_tmo) / len(comp) if len(comp) > 0 else np.nan

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
        return

    def print_avg(df, label):
        s = pd.to_numeric(df["response_time_ms"], errors="coerce").dropna()
        avg = s.mean() if not s.empty else np.nan
        print(f"{label}: {avg:.1f} ms (n={len(s)})")

    # Rewarded
    rew = comp_rt[comp_rt["response_time_category"] == "rewarded"]
    print_avg(rew, "Rewarded")

    # Unrewarded
    unr = comp_rt[comp_rt["response_time_category"] == "unrewarded"]
    print_avg(unr, "Unrewarded")

    # Reward Timeout
    tmo = comp_rt[comp_rt["response_time_category"] == "timeout_delayed"]
    print_avg(tmo, "Reward Timeout")

    # Rewarded + Unrewarded (excluding timeouts)
    both = comp_rt[comp_rt["response_time_category"].isin(["rewarded", "unrewarded"])]
    print_avg(both, "Average Response Time (Rewarded + Unrewarded)")

def FA_avg_response_times(results):
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
            if not s.empty:
                print(f"  - {pretty}: avg={s.mean():.1f} ms, median={s.median():.1f} ms, n={len(s)}")
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
            if not s.empty:
                print(f"  - {pretty}: avg={s.mean():.1f} ms, median={s.median():.1f} ms, n={len(s)}")
        print()

def response_rate(results):
    comp = results.get("completed_sequences", pd.DataFrame())
    comp_tmo = results.get("completed_sequence_reward_timeout", pd.DataFrame())
    n_resp = len(comp) - len(comp_tmo)
    print(f"Response Rate: {n_resp}/{len(comp)} = {n_resp/len(comp) if len(comp)>0 else np.nan:.3f}")
    return n_resp / len(comp) if len(comp) > 0 else np.nan

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
    return n_fa / len(fa_noninit_df) if len(fa_noninit_df) > 0 else np.nan

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