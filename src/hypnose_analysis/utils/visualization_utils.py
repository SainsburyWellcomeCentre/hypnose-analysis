import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Iterable, Optional, Union, Tuple
from hypnose_analysis.utils.metrics_utils import (
    load_session_results,
    run_all_metrics,
    parse_json_column,
    _populate_legacy_results_from_trial_data,
)
from datetime import timedelta, datetime
from hypnose_analysis.utils.classification_utils import load_all_streams, load_experiment
from hypnose_analysis.paths import (
    get_data_root,
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import re
import numpy as np
import json


def _load_table_with_trial_data(results_dir: Path, name: str) -> pd.DataFrame:
    """Load table by name with fallback to trial_data-derived reconstruction."""
    csv_path = results_dir / f"{name}.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    trial_df = pd.DataFrame()
    pq = results_dir / "trial_data.parquet"
    if pq.exists():
        try:
            trial_df = pd.read_parquet(pq)
        except Exception:
            pass
    if trial_df.empty:
        tcsv = results_dir / "trial_data.csv"
        if tcsv.exists():
            try:
                trial_df = pd.read_csv(tcsv)
            except Exception:
                pass

    if isinstance(trial_df, pd.DataFrame) and not trial_df.empty:
        res = {"trial_data": trial_df}
        _populate_legacy_results_from_trial_data(res)
        return res.get(name, pd.DataFrame()).copy()

    return pd.DataFrame()

# Load metric results for visualization (NOTE: Previously in metrics_utils.py) ==============================================================================

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


# =========================================================== Metrics Plotting Functions =============================================================================

def plot_behavior_metrics(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    variables: Optional[Iterable[str]] = None,
    *,
    protocol_filter: Optional[str] = None,
    compute_if_missing: bool = False,
    verbose: bool = True,
    black_white: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    plot_HR_separately: bool = False,
    show_title: bool = True,
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
    - y_range: Optional tuple (ymin, ymax); if provided, sets y-limits for each plot.
    - plot_HR_separately: If True and plotting hidden_rule_detection_rate, also plot per-HR-odor detection alongside total.
    - show_title: If False, omit the plot title (useful for tighter layouts).

    Returns:
    - List of matplotlib Figure objects.
    """
    if not variables:
        raise ValueError("Please provide `variables` (list of metric names or dot-paths).")

    rows = []
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    # Gather sessions
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for session_num, ses in enumerate(ses_dirs, start=1):
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
                        "session_num": session_num,
                        "date": int(date_str) if str(date_str).isdigit() else date_str,
                        "date_str": str(date_str),
                        "protocol": str(protocol) if protocol else "Unknown",
                        "variable": var,
                        "value": float(val),
                        "series": "Total",
                    })

                # If requested, add per-HR-odor detection rates
                if plot_HR_separately and var == "hidden_rule_detection_rate":
                    hr_block = metrics.get("hidden_rule_by_odor", {}) if isinstance(metrics, dict) else {}
                    by_odor = hr_block.get("by_odor", {}) if isinstance(hr_block, dict) else {}
                    for odor, stats in by_odor.items():
                        dr = None
                        if isinstance(stats, dict):
                            dr = stats.get("detection_rate")
                        if isinstance(dr, (int, float)) and not np.isnan(dr):
                            rows.append({
                                "subjid": int(sid),
                                "session_num": session_num,
                                "date": int(date_str) if str(date_str).isdigit() else date_str,
                                "date_str": str(date_str),
                                "protocol": str(protocol) if protocol else "Unknown",
                                "variable": var,
                                "value": float(dr),
                                "series": str(odor),
                            })

    if not rows:
        if verbose:
            print("[plot_behavior_metrics] No data found for the given filters.")
        return []

    df = pd.DataFrame(rows)
    if "series" not in df.columns:
        df["series"] = "Total"
    
    # Subject -> marker mapping
    markers_cycle = ['o', '^', 's', 'X', 'D', 'P', 'v', '>', '<', '*', 'h', 'H', '8', 'p', 'x']
    unique_subj = sorted(df["subjid"].unique())
    subj_to_marker = {sid: markers_cycle[i % len(markers_cycle)] for i, sid in enumerate(unique_subj)}

    # Protocol -> color mapping (or mono if black_white)
    prot_to_color = {}
    unique_protocols = []
    if not black_white:
        for p in df["protocol"]:
            if p not in unique_protocols and p and p != "Unknown":
                unique_protocols.append(p)
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

        fig, ax = plt.subplots(figsize=(12, 9))

        # Series handling: when plotting HR detection, allow per-odor series; otherwise single "Total"
        series_values = sorted(df_var["series"].unique()) if "series" in df_var.columns else ["Total"]

        # Define default palette for per-HR-odor series (C->B color, F->A color)
        odor_a_color = '#FF6B6B'  # same as odor A in plot_decision_accuracy_by_odor
        odor_b_color = '#4ECDC4'  # same as odor B in plot_decision_accuracy_by_odor

        def _map_series_color(series_label: str):
            lbl = str(series_label).lower()
            letters = [ch for ch in lbl if ch.isalpha()]
            last = letters[-1] if letters else ""
            if lbl == "total":
                return "black"
            if last == "f":  # HR odor F -> odor A color
                return odor_a_color
            if last == "c":  # HR odor C -> odor B color
                return odor_b_color
            if last == "a":
                return odor_a_color
            if last == "b":
                return odor_b_color
            return None

        hr_series_mode = (plot_HR_separately and var == "hidden_rule_detection_rate")

        if hr_series_mode:
            # Always use colored series with solid lines; Total is thick black
            series_to_color = {s: (_map_series_color(s) or "black") for s in series_values}
            series_to_ls = {s: "-" for s in series_values}
            series_to_lw = {s: (3.0 if s == "Total" else 1.8) for s in series_values}
        elif black_white:
            series_to_color = {s: (0, 0, 0, 1.0) for s in series_values}
            linestyle_cycle = ["-", "--", ":", "-."]
            series_to_ls = {s: linestyle_cycle[i % len(linestyle_cycle)] for i, s in enumerate(series_values)}
            series_to_lw = {s: (2.5 if s == "Total" else 1.2) for s in series_values}
        else:
            series_cmap = cm.get_cmap("tab20", max(3, len(series_values)))
            series_to_color = {}
            for i, s in enumerate(series_values):
                mapped = _map_series_color(s)
                series_to_color[s] = mapped if mapped is not None else series_cmap(i % series_cmap.N)
            series_to_ls = {s: "-" for s in series_values}
            series_to_lw = {s: (2.5 if s == "Total" else 1.5) for s in series_values}

        # Plot each series per subject
        for series in series_values:
            df_series = df_var[df_var.get("series", "Total") == series]
            for sid in unique_subj:
                dsub = df_series[df_series["subjid"] == sid].sort_values("session_num")
                if dsub.empty:
                    continue
                color = series_to_color.get(series, "black")
                ls = series_to_ls.get(series, "-")
                lw = series_to_lw.get(series, 1.0)
                ax.plot(dsub["session_num"], dsub["value"], color=color, linestyle=ls, linewidth=lw, alpha=0.8, zorder=1)
                # Scatter with subject marker; color by series to distinguish odors
                ax.scatter(
                    dsub["session_num"], dsub["value"],
                    c=[color] * len(dsub),
                    marker=subj_to_marker[sid],
                    edgecolors="black",
                    linewidths=0.5,
                    s=40,
                    zorder=2,
                )

        # X-axis: session numbers with sparse labels
        session_nums = sorted(df_var["session_num"].unique())
        n_sessions = len(session_nums)
        max_session = session_nums[-1] if session_nums else 0
        
        # Determine tick spacing (every 5-10 sessions)
        if n_sessions <= 10:
            tick_spacing = 2
        elif n_sessions <= 30:
            tick_spacing = 5
        elif n_sessions <= 80:
            tick_spacing = 10
        elif n_sessions <= 100:
            tick_spacing = 20
        else:
            tick_spacing = 50
        
        # Create x-axis ticks and labels
        x_ticks = [i for i in session_nums if i % tick_spacing == 0]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
        ax.set_xlim([0.5, max_session + 0.5])
        if y_range is not None and len(y_range) == 2:
            ax.set_ylim(y_range)
        
        # Format title: split by "_" and capitalize each word
        title_formatted = " ".join(word.capitalize() for word in var.split(".")[0].split("_")) + (f" ({var.split('.')[1].capitalize()})" if '.' in var else "")

        
        ax.set_xlabel("Days", fontsize=30, fontweight='bold')
        ax.set_ylabel(var.replace("_", " ").title(), fontsize=30, fontweight='bold')
        if show_title:
            ax.set_title(title_formatted, fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)
        
        # Make axes bold
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Remove upper and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # No grid
        ax.grid(False)

        # Build legends: subjects (markers) and either protocols or series (colors)
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

        series_handles = []
        if len(series_values) > 1:
            for s in series_values:
                color = series_to_color.get(s, (0, 0, 0, 1))
                ls = series_to_ls.get(s, "-")
                lw = series_to_lw.get(s, 1.0)
                series_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        color=color,
                        linestyle=ls,
                        linewidth=lw,
                        markerfacecolor='white' if black_white else color,
                        markeredgecolor="black",
                        markersize=7,
                        label=s,
                    )
                )

        protocol_handles = []
        if not series_handles and not black_white:
            protocol_handles = [
                Line2D([0], [0],
                       marker='o',
                       color='none', linestyle="",
                       markerfacecolor=prot_to_color.get(p, (0, 0, 0, 1)),
                       markeredgecolor="black",
                       markersize=7,
                       label=p)
                for p in unique_protocols
            ]

        # Place legends
        if subject_handles:
            leg1 = ax.legend(handles=subject_handles, title="Subjects", loc="upper left", bbox_to_anchor=(1.02, 1.0))
            ax.add_artist(leg1)
        if series_handles:
            ax.legend(handles=series_handles, title="HR Series", loc="lower left", bbox_to_anchor=(1.02, 0.0))
        elif protocol_handles:
            ax.legend(handles=protocol_handles, title="Protocols", loc="lower left", bbox_to_anchor=(1.02, 0.0))

        plt.tight_layout()
        figs.append(fig)

    return figs

def plot_decision_accuracy_by_odor(subjid, dates=None, figsize=(12, 6), save_path=None, plot_choice_acc=False, plot_AB=True):
    """
    Plot decision accuracy by odor (A, B) and total over dates.
    Optionally include global choice accuracy as a separate line.
    Fast version using pre-computed metrics with existing helper functions.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (12, 6))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    plot_choice_acc : bool, optional
        If True, also plot global choice accuracy as a dark grey line (default: False)
    plot_AB : bool, optional
        If True, plot odor-specific accuracies for A and B (default: True). If False, omit A/B lines.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    rows = []
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    def _normalize_odor_label(odor_raw):
        """Map assorted odor keys to canonical labels (A/B/Total/other)."""
        if isinstance(odor_raw, (int, float)) and not np.isnan(odor_raw):
            val = int(odor_raw)
            if val in (0, 1):
                return "A" if val == 0 else "B"
        if isinstance(odor_raw, str):
            raw = odor_raw.strip()
            lower = raw.lower()
            base = lower.replace("odor", "").replace("_", "").replace(" ", "")
            if base in {"a", "1", "01"}:
                return "A"
            if base in {"b", "2", "02"}:
                return "B"
            if lower in {"total", "overall"}:
                return "Total"
        return str(odor_raw)

    def _collect_odor_acc_rows(acc_block, date_int):
        """Handle both legacy flat dicts and new nested decision_accuracy_by_odor blocks."""
        collected = []

        def add_from_dict(dct):
            for odor, acc in dct.items():
                if isinstance(acc, (int, float)) and not np.isnan(acc):
                    collected.append({
                        "date": date_int,
                        "odor": _normalize_odor_label(odor),
                        "accuracy": float(acc)
                    })

        if not isinstance(acc_block, dict):
            return collected

        if "decision_accuracy_ab" in acc_block:
            add_from_dict(acc_block.get("decision_accuracy_ab", {}))
        if "decision_accuracy_total" in acc_block:
            add_from_dict(acc_block.get("decision_accuracy_total", {}))

        # If neither of the new-schema keys are present, assume legacy flat mapping
        if not collected:
            add_from_dict(acc_block)

        return collected

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses in ses_dirs:
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            metrics = _ensure_metrics_json(sid, date_str, results_dir, compute_if_missing=False)
            if metrics is None:
                continue
            
            # Extract DA by odor using the helper function
            acc_by_odor = metrics.get('decision_accuracy_by_odor', {})
            acc_total = _extract_metric_value(metrics, 'decision_accuracy')
            
            # Add odor-specific accuracies (supports legacy flat dict and new nested schema)
            rows.extend(_collect_odor_acc_rows(acc_by_odor, int(date_str)))
            
            # Add total accuracy
            if isinstance(acc_total, (int, float)) and not np.isnan(acc_total):
                rows.append({
                    "date": int(date_str),
                    "odor": "Total",
                    "accuracy": float(acc_total)
                })
            
            # Add global choice accuracy if requested
            if plot_choice_acc:
                gca = metrics.get('global_choice_accuracy', None)
                if isinstance(gca, (tuple, list)) and len(gca) >= 3:
                    gca_value = gca[2]
                    if isinstance(gca_value, (int, float)) and not np.isnan(gca_value):
                        rows.append({
                            "date": int(date_str),
                            "odor": "Global Choice Accuracy",
                            "accuracy": float(gca_value)
                        })
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    unique_dates = sorted(df["date"].unique())
    date_to_x = {d: i for i, d in enumerate(unique_dates)}
    df["x"] = df["date"].map(date_to_x)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'Total': 'black', 'Global Choice Accuracy': 'darkgreen'}
    linewidths = {'A': 1.5, 'B': 1.5, 'Total': 4, 'Global Choice Accuracy': 3.5}
    markers = {'A': 'o', 'B': 'o', 'Total': 's', 'Global Choice Accuracy': '^'}
    linestyles = {'A': '-', 'B': '-', 'Total': '-', 'Global Choice Accuracy': '--'}
    
    # Determine which odors to plot (restricted set)
    unique_odors = set(df["odor"].unique())
    odors_to_plot = []
    if plot_AB:
        for base in ["A", "B"]:
            if base in unique_odors:
                odors_to_plot.append(base)
    if "Total" in unique_odors:
        odors_to_plot.append("Total")
    if plot_choice_acc and "Global Choice Accuracy" in unique_odors:
        odors_to_plot.append("Global Choice Accuracy")
    
    for odor in odors_to_plot:
        subset = df[df["odor"] == odor]
        if subset.empty:
            continue
        ax.plot(subset["x"].values, subset["accuracy"].values, 
                label=odor,
                color=colors.get(odor, '#999999'),
                linewidth=linewidths.get(odor, 1.5),
                linestyle=linestyles.get(odor, '-'),
                marker=markers.get(odor, 'o'),
                markersize=4 if odor not in ('Total', 'Global Choice Accuracy') else 6,
                alpha=0.7 if odor not in ('Total', 'Global Choice Accuracy') else 0.8,
                zorder=10 if odor in ('Total', 'Global Choice Accuracy') else 1)
    
    ax.set_xlabel('Days', fontsize=30, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=30, fontweight='bold')
    ax.tick_params(axis='both', labelsize=26)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.legend(loc='best', fontsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
        
    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    title = f"Subject {str(subjid).zfill(3)} - Decision Accuracy by Odor"
    if plot_choice_acc:
        title += " (with Global Choice Accuracy)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_sampling_times_analysis(subjid, dates=None, figsize=(16, 18)):
    """
    Plot sampling times (poke durations) by position and by odor for completed and aborted trials.
    OPTIMIZED: Vectorized JSON parsing instead of row-by-row loops.
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for session_num, ses in enumerate(ses_dirs, start=1):
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            comp = _load_table_with_trial_data(results_dir, "completed_sequences")
            aborted = _load_table_with_trial_data(results_dir, "aborted_sequences_detailed")
            
            # ============ COMPLETED TRIALS - VECTORIZED ============
            if not comp.empty and "position_poke_times" in comp.columns:
                # Vectorize: parse ALL JSON at once
                def extract_poke_times(json_str):
                    """Extract list of (position, poke_ms, odor) tuples from JSON"""
                    try:
                        data = parse_json_column(json_str)
                        if not isinstance(data, dict):
                            return []
                        
                        results = []
                        for pos_str, info in data.items():
                            if isinstance(info, dict):
                                poke_ms = info.get("poke_time_ms")
                                odor = info.get("odor_name")
                                if poke_ms is not None and isinstance(poke_ms, (int, float)) and poke_ms > 0:
                                    try:
                                        results.append((int(pos_str), float(poke_ms), str(odor) if odor else None))
                                    except (ValueError, TypeError):
                                        pass
                        return results
                    except Exception:
                        return []
                
                # Apply to all rows at once and flatten
                all_poke_times = comp["position_poke_times"].apply(extract_poke_times)
                for poke_list in all_poke_times:
                    for position, poke_ms, odor in poke_list:
                        rows.append({
                            "trial_type": "completed",
                            "position": position,
                            "odor": odor,
                            "poke_time_ms": poke_ms,
                            "session_num": session_num,
                            "date": int(date_str) if str(date_str).isdigit() else date_str,
                        })
            
            # ============ ABORTED TRIALS - VECTORIZED ============
            if not aborted.empty and "presentations" in aborted.columns:
                # Vectorize: parse ALL JSON at once
                def extract_abort_poke_times(presentations_str, last_event_idx):
                    """Extract list of (position, poke_ms, odor) tuples excluding abort event"""
                    try:
                        pres_list = parse_json_column(presentations_str)
                        if not isinstance(pres_list, list):
                            return []
                        
                        results = []
                        for pres in pres_list:
                            if isinstance(pres, dict):
                                idx = pres.get("index_in_trial")
                                if idx == last_event_idx:
                                    continue
                                
                                poke_ms = pres.get("poke_time_ms")
                                pos = pres.get("position")
                                odor = pres.get("odor_name")
                                
                                if poke_ms is not None and isinstance(poke_ms, (int, float)) and poke_ms > 0:
                                    try:
                                        results.append((int(pos) if pos is not None else None, 
                                                      float(poke_ms), 
                                                      str(odor) if odor else None))
                                    except (ValueError, TypeError):
                                        pass
                        return results
                    except Exception:
                        return []
                
                # Vectorized apply
                all_abort_times = aborted.apply(
                    lambda row: extract_abort_poke_times(row["presentations"], row.get("last_event_index")),
                    axis=1
                )
                
                for poke_list in all_abort_times:
                    for position, poke_ms, odor in poke_list:
                        rows.append({
                            "trial_type": "aborted",
                            "position": position,
                            "odor": odor,
                            "poke_time_ms": poke_ms,
                            "session_num": session_num,
                            "date": int(date_str) if str(date_str).isdigit() else date_str,
                        })
    
    if not rows:
        print("No data found")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # ============ PLOT 1: Completed by Position ============
    ax = axes[0, 0]
    df_comp_pos = df[(df["trial_type"] == "completed") & (df["position"].notna())].copy()
    
    if not df_comp_pos.empty:
        positions = sorted(df_comp_pos["position"].unique())
        
        means = []
        stds = []
        
        for pos in positions:
            values = df_comp_pos[df_comp_pos["position"] == pos]["poke_time_ms"].values
            
            # Scatter with jitter
            x_jitter = np.random.normal(pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='steelblue')
            
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Mean points with error bars (no line)
        ax.scatter(positions, means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(positions)
    
    ax.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Poke Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'Completed Trials: Sampling Time by Position\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 2: Aborted by Position ============
    ax = axes[0, 1]
    df_abort_pos = df[(df["trial_type"] == "aborted") & (df["position"].notna())].copy()
    
    if not df_abort_pos.empty:
        positions = sorted(df_abort_pos["position"].unique())
        
        means = []
        stds = []
        
        for pos in positions:
            values = df_abort_pos[df_abort_pos["position"] == pos]["poke_time_ms"].values
            
            # Scatter with jitter
            x_jitter = np.random.normal(pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='coral')
            
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Mean points with error bars (no line)
        ax.scatter(positions, means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(positions)
    
    ax.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Poke Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'Aborted Trials: Sampling Time by Position\n(excl. abort position)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 3: Completed by Odor ============
    ax = axes[1, 0]
    df_comp_odor = df[(df["trial_type"] == "completed") & (df["odor"].notna())].copy()
    
    if not df_comp_odor.empty:
        odors = sorted(df_comp_odor["odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        means = []
        stds = []
        
        for odor in odors:
            values = df_comp_odor[df_comp_odor["odor"] == odor]["poke_time_ms"].values
            x_pos = odor_to_x[odor]
            
            # Scatter with jitter
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='steelblue')
            
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Mean points with error bars (no line)
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(range(len(odors)), means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Poke Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'Completed Trials: Sampling Time by Odor\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 4: Aborted by Odor ============
    ax = axes[1, 1]
    df_abort_odor = df[(df["trial_type"] == "aborted") & (df["odor"].notna())].copy()
    
    if not df_abort_odor.empty:
        odors = sorted(df_abort_odor["odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        means = []
        stds = []
        
        for odor in odors:
            values = df_abort_odor[df_abort_odor["odor"] == odor]["poke_time_ms"].values
            x_pos = odor_to_x[odor]
            
            # Scatter with jitter
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='coral')
            
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Mean points with error bars (no line)
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(range(len(odors)), means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Poke Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title(f'Aborted Trials: Sampling Time by Odor\n(excl. abort odor)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 5: Average Poke Time per Position over Sessions ============
    ax = axes[2, 0]
    df_pos_series = df[(df["trial_type"] == "completed") & (df["position"].notna()) & (df["session_num"].notna())].copy()

    if not df_pos_series.empty:
        grouped = df_pos_series.groupby(["session_num", "position"]).poke_time_ms.mean().reset_index()
        positions = sorted(grouped["position"].unique())

        # Dark-to-light blue gradient for positions
        pos_palette = [
            '#0b3c68',  # dark
            '#155d8a',
            '#1f7eac',
            '#3c99c7',
            '#65b4d7',  # light
        ]

        for i, pos in enumerate(positions):
            pos_data = grouped[grouped["position"] == pos].sort_values("session_num")
            color = pos_palette[i % len(pos_palette)]
            ax.plot(pos_data["session_num"], pos_data["poke_time_ms"],
                    label=f"Pos {pos}",
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    alpha=0.85)

        ax.set_xlabel("Session", fontsize=11, fontweight='bold')
        ax.set_ylabel("Average Poke Time (ms)", fontsize=11, fontweight='bold')
        ax.set_title(f'Average Poke Time per Position Across Sessions\n(Subject {str(subjid).zfill(3)})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        ax.legend(loc='best', fontsize=9)
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    # ============ PLOT 6: Average Poke Time per Odor over Sessions ============
    ax = axes[2, 1]
    df_odor_series = df[(df["trial_type"] == "completed") & (df["odor"].notna()) & (df["session_num"].notna())].copy()

    if not df_odor_series.empty:
        grouped = df_odor_series.groupby(["session_num", "odor"]).poke_time_ms.mean().reset_index()
        odors = sorted(grouped["odor"].unique())

        def _odor_color(odor_label: str):
            raw = str(odor_label).strip()
            lower = raw.lower()
            base = lower.replace("odor", "").replace("_", "").replace(" ", "")
            if base in {"a", "1", "01"}:
                return '#FF6B6B'
            if base in {"f"}:
                return '#E63946'  # slightly deeper red
            if base in {"b", "2", "02"}:
                return '#4ECDC4'
            if base in {"c", "3", "03"}:
                return '#1D9AB3'  # slightly deeper blue
            return '#888888'

        for odor in odors:
            odor_data = grouped[grouped["odor"] == odor].sort_values("session_num")
            ax.plot(odor_data["session_num"], odor_data["poke_time_ms"],
                    label=str(odor),
                    color=_odor_color(odor),
                    linewidth=2.2,
                    marker="o",
                    markersize=5,
                    alpha=0.9)

        ax.set_xlabel("Session", fontsize=11, fontweight='bold')
        ax.set_ylabel("Average Sampling Time (ms)", fontsize=11, fontweight='bold')
        ax.set_title(f'Average Sampling Time per Odor Across Sessions\n(Subject {str(subjid).zfill(3)})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        ax.legend(loc='best', fontsize=9)
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()
    return fig, axes

def plot_abortion_and_fa_rates(
    subjid, 
    dates=None, 
    figsize=(18, 14),
    include_noninitiated_in_fa_odor=True,
    fa_types='FA_time_in'  # NEW PARAMETER
):
    """
    Plot FA rates, abortion rates, and FA ratio by position and odor across sessions.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : list, tuple, or None
        Dates to include
    figsize : tuple
        Figure size
    include_noninitiated_in_fa_odor : bool
        If True, include non-initiated FAs in FA Rate per Odor.
    fa_types : str or list, optional
        Which FA types to include:
        - 'FA_Time_In' : only FA_Time_In
        - 'FA_Time_In,FA_Time_Out' : multiple specific types (comma-separated)
        - 'All' : all FA types starting with 'FA_'
        (default: 'FA_Time_In')
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # DEFINE fa_filter_fn HERE - BEFORE THE LOOPS
    if isinstance(fa_types, str):
        if fa_types.lower() == 'all':
            fa_filter_fn = lambda x: x.astype(str).str.startswith('FA_', na=False)
        else:
            # Handle comma-separated list like 'FA_Time_In,FA_Time_Out'
            fa_type_list = [t.strip() for t in fa_types.split(',')]
            fa_filter_fn = lambda x: x.astype(str).isin(fa_type_list)
    elif isinstance(fa_types, list):
        fa_filter_fn = lambda x: x.astype(str).isin(fa_types)
    else:
        fa_filter_fn = lambda x: x.astype(str) == str(fa_types)
    
    rows = []
    fa_port_rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses in ses_dirs:
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            # Load metrics JSON for FA rates
            metrics = _ensure_metrics_json(sid, date_str, results_dir, compute_if_missing=False)
            if metrics is None:
                # No metrics JSON - we'll still try to get FA stats from CSVs below
                metrics = {}
            
            
            # Extract FA rates from metrics (no need to load raw CSVs)
            fa_stats = metrics.get("fa_abortion_stats", {})

            # FA rate per odor (FA Time In only)
            fa_by_odor = fa_stats.get("by_odor", [])
            if isinstance(fa_by_odor, list):
                for item in fa_by_odor:
                    if isinstance(item, dict) and "Odor" in item:
                        odor = item["Odor"]
                        total_ab = item.get("Total Abortions")
                        fa_time_in_str = item.get("FA Time In", "")
                        # Extract count from format like "5 (0.50)"
                        if "(" in fa_time_in_str and total_ab is not None:
                            try:
                                fa_time_in_count = int(fa_time_in_str.split()[0])
                                fa_ratio = fa_time_in_count / total_ab
                                rows.append({
                                    "date": int(date_str),
                                    "metric_type": "fa_rate",
                                    "category": "odor",
                                    "position_or_odor": str(odor),
                                    "rate": fa_ratio
                                })
                            except (ValueError, IndexError):
                                continue

            # FA rate per position (FA Time In only)
            fa_by_position = fa_stats.get("by_position", [])
            if isinstance(fa_by_position, list):
                for item in fa_by_position:
                    if isinstance(item, dict) and "Position" in item:
                        pos = item["Position"]
                        total_ab = item.get("Total Abortions")
                        fa_time_in_str = item.get("FA Time In", "")
                        # Extract count from format like "5 (0.50)"
                        if "(" in fa_time_in_str and total_ab is not None:
                            try:
                                pos_int = int(pos)
                                fa_time_in_count = int(fa_time_in_str.split()[0])
                                fa_ratio = fa_time_in_count / total_ab
                                rows.append({
                                    "date": int(date_str),
                                    "metric_type": "fa_rate",
                                    "category": "position",
                                    "position_or_odor": pos_int,
                                    "rate": fa_ratio
                                })
                            except (ValueError, IndexError):
                                continue
            
            # Non-Initiated FA before position 1 (add as "Non-Initiated" to position category)
            fa_noninit_rate = metrics.get("non_initiated_FA_rate", None)
            if isinstance(fa_noninit_rate, (tuple, list)) and len(fa_noninit_rate) == 3:
                n_fa, n_total, rate_noninit = fa_noninit_rate
                rows.append({
                    "date": int(date_str),
                    "metric_type": "fa_rate",
                    "category": "position",
                    "position_or_odor": "Non-Initiated",
                    "rate": rate_noninit
                })

            # ============ FA PORT RATIO - Calculate from CSVs (ensures correct include_noninitiated_in_fa_odor behavior) ============
            ab_det_path = results_dir / "aborted_sequences_detailed.csv"
            fa_noninit_path = results_dir / "non_initiated_FA.csv"
            
            ab_det = pd.DataFrame()
            fa_noninit = pd.DataFrame()
            
            # Load aborted sequences with FA data
            if ab_det_path.exists():
                try:
                    ab_det = pd.read_csv(ab_det_path)
                    # Filter by fa_label if column exists
                    if "fa_label" in ab_det.columns:
                        ab_det = ab_det[fa_filter_fn(ab_det["fa_label"])].copy()
                    # Keep only needed columns
                    needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                    ab_det = ab_det[[col for col in needed_cols if col in ab_det.columns]]
                except Exception:
                    ab_det = pd.DataFrame()
            
            # Load non-initiated FA data if requested
            if include_noninitiated_in_fa_odor and fa_noninit_path.exists():
                try:
                    fa_noninit = pd.read_csv(fa_noninit_path)
                    # Filter by fa_label if column exists
                    if "fa_label" in fa_noninit.columns:
                        fa_noninit = fa_noninit[fa_filter_fn(fa_noninit["fa_label"])].copy()
                    # Keep only needed columns (note: uses 'odor_name' not 'last_odor_name')
                    needed_cols = ['fa_label', 'odor_name', 'fa_port']
                    fa_noninit = fa_noninit[[col for col in needed_cols if col in fa_noninit.columns]]
                    # Rename 'odor_name' to 'last_odor_name' for consistency with aborted data
                    if 'odor_name' in fa_noninit.columns:
                        fa_noninit = fa_noninit.rename(columns={'odor_name': 'last_odor_name'})
                except Exception:
                    fa_noninit = pd.DataFrame()
            
            # Combine data based on include_noninitiated_in_fa_odor parameter
            try:
                if include_noninitiated_in_fa_odor:
                    fa_all = pd.concat([ab_det, fa_noninit], ignore_index=True)
                else:
                    fa_all = ab_det.copy()
                
                # Calculate FA port ratio per odor
                if not fa_all.empty and "fa_port" in fa_all.columns and "last_odor_name" in fa_all.columns:
                    for odor in sorted(fa_all["last_odor_name"].dropna().unique()):
                        fa_odor = fa_all[fa_all["last_odor_name"] == odor]
                        n_a = (fa_odor["fa_port"] == 1).sum()
                        n_b = (fa_odor["fa_port"] == 2).sum()
                        n_total = n_a + n_b
                        ratio_a = (n_a - n_b) / n_total if n_total > 0 else np.nan
                        fa_port_rows.append({
                            "date": int(date_str),
                            "odor": str(odor),
                            "fa_ratio_a": ratio_a
                        })
            except Exception:
                pass
            # ============ ABORTION RATES - from metrics JSON ============
            
            # Abortion rate per position
            ab_pos_data = metrics.get("abortion_rate_positionX", {})
            if isinstance(ab_pos_data, dict):
                for pos, rate in ab_pos_data.items():
                    if rate is not None and isinstance(rate, (int, float)):
                        try:
                            rows.append({
                                "date": int(date_str),
                                "metric_type": "abortion_rate",
                                "category": "position",
                                "position_or_odor": int(pos),
                                "rate": float(rate)
                            })
                        except (ValueError, TypeError):
                            continue
            
            # Abortion rate per odor
            ab_odor_data = metrics.get("odorx_abortion_rate", {})
            if isinstance(ab_odor_data, dict):
                for odor, rate in ab_odor_data.items():
                    if rate is not None and isinstance(rate, (int, float)):
                        rows.append({
                            "date": int(date_str),
                            "metric_type": "abortion_rate",
                            "category": "odor",
                            "position_or_odor": str(odor),
                            "rate": float(rate)
                        })
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    df_port = pd.DataFrame(fa_port_rows) if fa_port_rows else pd.DataFrame()
    
    # Create figure with 5 subplots (3 rows: top 2x2, bottom 1x2 centered)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])  # Spans full width
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    # ============ PLOT 1: FA Rate per Position (with Non-Initiated) ============
    ax = ax1
    df_fa_pos = df[(df["metric_type"] == "fa_rate") & (df["category"] == "position")].copy()
    
    if not df_fa_pos.empty:
        # Sort positions: Non-Initiated first, then 1, 2, 3, 4, 5
        positions = ["Non-Initiated"] + sorted([p for p in df_fa_pos["position_or_odor"].unique() if p != "Non-Initiated"])
        position_to_x = {pos: i for i, pos in enumerate(positions)}
        
        for pos in positions:
            rates = df_fa_pos[df_fa_pos["position_or_odor"] == pos]["rate"].values
            x_pos = position_to_x[pos]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='steelblue')
        
        means = [df_fa_pos[df_fa_pos["position_or_odor"] == pos]["rate"].mean() for pos in positions]
        sems = [df_fa_pos[df_fa_pos["position_or_odor"] == pos]["rate"].sem() for pos in positions]
        
        ax.scatter(range(len(positions)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(positions)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions)
    
    ax.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('FA Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'FA Rate per Position\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 2: FA Rate per Odor ============
    ax = ax2
    df_fa_odor = df[(df["metric_type"] == "fa_rate") & (df["category"] == "odor")].copy()
    
    if not df_fa_odor.empty:
        odors = sorted(df_fa_odor["position_or_odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        for odor in odors:
            rates = df_fa_odor[df_fa_odor["position_or_odor"] == odor]["rate"].values
            x_pos = odor_to_x[odor]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='steelblue')
        
        means = [df_fa_odor[df_fa_odor["position_or_odor"] == odor]["rate"].mean() for odor in odors]
        sems = [df_fa_odor[df_fa_odor["position_or_odor"] == odor]["rate"].sem() for odor in odors]
        
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(odors)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor', fontsize=11, fontweight='bold')
    ax.set_ylabel('FA Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'FA Rate per Odor\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 3: Abortion Rate per Position ============
    ax = ax3
    df_ab_pos = df[(df["metric_type"] == "abortion_rate") & (df["category"] == "position")].copy()
    
    if not df_ab_pos.empty:
        positions = sorted(df_ab_pos["position_or_odor"].unique())
        
        for pos in positions:
            rates = df_ab_pos[df_ab_pos["position_or_odor"] == pos]["rate"].values
            x_jitter = np.random.normal(pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='coral')
        
        means = [df_ab_pos[df_ab_pos["position_or_odor"] == pos]["rate"].mean() for pos in positions]
        sems = [df_ab_pos[df_ab_pos["position_or_odor"] == pos]["rate"].sem() for pos in positions]
        
        ax.scatter(positions, means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(positions, means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(positions)
    
    ax.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Abortion Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'Abortion Rate per Position\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # ============ PLOT 4: Abortion Rate per Odor ============
    ax = ax4
    df_ab_odor = df[(df["metric_type"] == "abortion_rate") & (df["category"] == "odor")].copy()
    
    if not df_ab_odor.empty:
        odors = sorted(df_ab_odor["position_or_odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        for odor in odors:
            rates = df_ab_odor[df_ab_odor["position_or_odor"] == odor]["rate"].values
            x_pos = odor_to_x[odor]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='coral')
        
        means = [df_ab_odor[df_ab_odor["position_or_odor"] == odor]["rate"].mean() for odor in odors]
        sems = [df_ab_odor[df_ab_odor["position_or_odor"] == odor]["rate"].sem() for odor in odors]
        
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(odors)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Abortion Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'Abortion Rate per Odor\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')

    
    # ============ PLOT 5: FA Ratio (A-B) / (A+B) per Odor (full width) ============
    ax = ax5
    if not df_port.empty:
        odors = sorted(df_port["odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        for odor in odors:
            ratios = df_port[df_port["odor"] == odor]["fa_ratio_a"].values
            x_pos = odor_to_x[odor]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(ratios))
            ax.scatter(x_jitter, ratios, alpha=0.4, s=20, color='steelblue')
        
        means = [df_port[df_port["odor"] == odor]["fa_ratio_a"].mean() for odor in odors]
        sems = [df_port[df_port["odor"] == odor]["fa_ratio_a"].sem() for odor in odors]
        
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(odors)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor', fontsize=11, fontweight='bold')
    ax.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
    ax.set_title(f'FA Ratio (A-B)/(A+B) per Odor\n(Subject {str(subjid).zfill(3)})', 
                fontsize=12, fontweight='bold')
    ax.set_ylim([-1.1, 1.1])
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    return fig, axes

def plot_response_times_completed_vs_fa(subjid, dates=None, figsize=(12, 8), y_limit=20000):
    """
    Scatter plot comparing average response times for completed sequences vs FA Time In abortions.
    Both metrics on the same plot sharing Y-axis for easy comparison.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (12, 8))
    y_limit : float, optional
        Maximum Y-axis value to display (default: 20000). Points above this are excluded.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses in ses_dirs:
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            metrics = _ensure_metrics_json(sid, date_str, results_dir, compute_if_missing=False)
            if metrics is None:
                continue
            
            # Get average response time for completed sequences
            avg_resp_times = metrics.get("avg_response_time", {})
            completed_rt = avg_resp_times.get("Average Response Time (Rewarded + Unrewarded)")
            if completed_rt is not None and not np.isnan(completed_rt):
                rows.append({
                    "date": int(date_str),
                    "response_type": "Completed Sequences",
                    "response_time_ms": float(completed_rt)
                })
            
            # Get average response time for FA Time In abortions
            fa_resp_times = metrics.get("FA_avg_response_times", {})
            fa_time_in_rt = fa_resp_times.get("Aborted FA Time In")
            if fa_time_in_rt is not None and not np.isnan(fa_time_in_rt):
                rows.append({
                    "date": int(date_str),
                    "response_type": "FA Time In Abortions",
                    "response_time_ms": float(fa_time_in_rt)
                })
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    
    # Filter by y_limit
    df_filtered = df[df["response_time_ms"] <= y_limit].copy()
    
    if df_filtered.empty:
        print(f"No data found below y_limit={y_limit}")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    response_types = ["Completed Sequences", "FA Time In Abortions"]
    x_positions = [0, 1]
    colors = ['steelblue', 'coral']
    
    for x_pos, resp_type, color in zip(x_positions, response_types, colors):
        df_subset = df_filtered[df_filtered["response_type"] == resp_type].copy()
        
        if not df_subset.empty:
            values = df_subset["response_time_ms"].values
            
            # Scatter with jitter
            x_jitter = np.random.normal(x_pos, 0.08, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=80, color=color, zorder=3)
            
            # Calculate mean and std
            mean_rt = values.mean()
            std_rt = values.std()
            
            # Plot mean point with error bars
            ax.scatter([x_pos], [mean_rt], color='darkred', s=150, zorder=5, marker='D',
                      edgecolors='black', linewidth=2, label='Mean ± SD' if x_pos == 0 else "")
            ax.errorbar([x_pos], [mean_rt], yerr=std_rt, fmt='none', ecolor='darkred',
                       capsize=8, capthick=2, linewidth=2.5, zorder=4)
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(response_types, fontsize=12, fontweight='bold')
    ax.set_ylabel('Response Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, y_limit])
    ax.set_title(f'Average Response Times Comparison\n(Subject {str(subjid).zfill(3)})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    return fig, ax

def plot_fa_ratio_a_over_sessions(
    subjid,
    dates=None,
    figsize=(14, 10),
    include_noninitiated=True
):
    """
    Plot FA Ratio A/(A+B) over sessions for each odor (OPTIMIZED).
    
    Parameters similar to original, but now loads only necessary data.
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    fa_data = {}  # {odor: [(session_num, ratio, n_a, n_b, n_total), ...]}
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses in enumerate(ses_dirs, start=1):
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            ab_det = _load_table_with_trial_data(results_dir, "aborted_sequences_detailed")
            if not ab_det.empty:
                needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                ab_det = ab_det[[col for col in needed_cols if col in ab_det.columns]]

            fa_noninit = pd.DataFrame()
            if include_noninitiated:
                fa_noninit = _load_table_with_trial_data(results_dir, "non_initiated_FA")
                if not fa_noninit.empty:
                    needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                    fa_noninit = fa_noninit[[col for col in needed_cols if col in fa_noninit.columns]]
            
            # Combine only if we have data
            if ab_det.empty and fa_noninit.empty:
                continue
            
            # Filter and combine FA data
            try:
                fa_list = []
                
                if not ab_det.empty and 'fa_label' in ab_det.columns:
                    fa_ab = ab_det[ab_det['fa_label'].astype(str) == 'FA_time_in']
                    if not fa_ab.empty:
                        fa_list.append(fa_ab)
                
                if not fa_noninit.empty and 'fa_label' in fa_noninit.columns:
                    fa_ni = fa_noninit[fa_noninit['fa_label'].astype(str) == 'FA_time_in']
                    if not fa_ni.empty:
                        fa_list.append(fa_ni)
                
                if not fa_list:
                    continue
                
                fa_all = pd.concat(fa_list, ignore_index=True)
            except Exception as e:
                continue
            
            if fa_all.empty or 'fa_port' not in fa_all.columns or 'last_odor_name' not in fa_all.columns:
                continue
            
            # Calculate FA port ratio per odor
            try:
                for odor in sorted(fa_all['last_odor_name'].dropna().unique()):
                    fa_odor = fa_all[fa_all['last_odor_name'] == odor]
                    n_a = (fa_odor['fa_port'] == 1).sum()
                    n_b = (fa_odor['fa_port'] == 2).sum()
                    n_total = n_a + n_b
                    ratio_a = n_a / n_total if n_total > 0 else np.nan
                    
                    if odor not in fa_data:
                        fa_data[odor] = []
                    fa_data[odor].append({
                        'session_num': session_num,
                        'date': int(date_str),
                        'ratio_a': ratio_a,
                        'n_a': n_a,
                        'n_b': n_b,
                        'n_total': n_total
                    })
            except Exception as e:
                continue
    
    if not fa_data:
        print("No FA data found")
        return {}
    
    # Create one figure per odor
    figs = {}
    odor_list = sorted(fa_data.keys())
    
    for odor in odor_list:
        data = fa_data[odor]
        data = sorted(data, key=lambda x: x["session_num"])
        
        x_positions = np.arange(len(data))
        session_nums = [d["session_num"] for d in data]
        ratios = [d["ratio_a"] for d in data]
        n_a_list = [d["n_a"] for d in data]
        n_total_list = [d["n_total"] for d in data]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(x_positions, ratios, color='black', linewidth=1.0, alpha=0.6, zorder=1)
        ax.scatter(x_positions, ratios, s=40, color='black', alpha=0.8, 
                  edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
        
        y_text = 1.08
        for x_pos, n_a, n_total in zip(x_positions, n_a_list, n_total_list):
            ax.text(x_pos, y_text, f"{n_a}/{n_total}", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   transform=ax.get_xaxis_transform())
        
        ax.set_xlim([-0.5, len(data) - 0.5])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(sn) for sn in session_nums])
        
        ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('FA Ratio A / (A+B)', fontsize=12, fontweight='bold')
        ax.set_title(f'FA Ratio Odor {odor}\n(Subject {str(subjid).zfill(3)})',
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(False)
        
        plt.tight_layout()
        figs[odor] = fig
    
    return figs


# =========================================================== Movement & Behavior Plotting Functions ================================================================


def plot_cumulative_rewards(subjids, dates, split_days=False, figsize=(12, 6), title=None, save_path=None):
    """
    Plot cumulative rewards with inter-session gap collapsing.
    
    OPTIMIZATION: Skip load_session_results() for 15+ DataFrames.
    Load ONLY: completed_sequence_rewarded CSV + manifest.json
    This gives ~10-15x speedup while keeping all visual features.
    
    Parameters:
    -----------
    subjids : int or list
        Subject ID(s)
    dates : list, tuple, or None
        Dates to include
    split_days : bool, optional
        If True, reset cumulative count per day (default: False)
    figsize : tuple, optional
        Figure size (default: (12, 6))
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Save path for figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Ensure subjids is a list
    if not isinstance(subjids, (list, tuple)):
        subjids = [subjids]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(range(len(subjids)))

    for subj_idx, subjid in enumerate(subjids):
        all_rewarded = []
        session_info = []

        # Find subject directory
        base_path = get_rawdata_root()
        server_root = get_server_root()
        derivatives_dir = get_derivatives_root()
        subj_str = f"sub-{str(subjid).zfill(3)}"
        subj_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*"))
        if not subj_dirs:
            print(f"Warning: No subject directory found for {subj_str}")
            continue
        subj_dir = subj_dirs[0]

        # Use _filter_session_dirs to get session directories
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses in ses_dirs:
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            rewarded_trials = _load_table_with_trial_data(results_dir, "completed_sequence_rewarded")
            if rewarded_trials.empty:
                continue
            try:
                rewarded_trials['sequence_start'] = pd.to_datetime(rewarded_trials['sequence_start'])
            except Exception:
                pass
            rewarded_trials['date'] = date_str
            all_rewarded.append(rewarded_trials)
            
            # OPTIMIZATION 2: Load ONLY manifest.json for session timing
            manifest_path = results_dir / "summary.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    runs = manifest.get('session', {}).get('runs', [])
                    if runs:
                        session_info.append({
                            'date': date_str,
                            'runs': runs,
                        })
                except Exception:
                    pass
        
        if not all_rewarded:
            print(f"Warning: No rewarded trials found for subject {subjid}")
            continue
        
        # Combine all dates
        combined = pd.concat(all_rewarded, ignore_index=True)
        combined = combined.sort_values('sequence_start').reset_index(drop=True)
        
        # Subject-specific session boundaries and gaps
        subj_session_starts = []
        subj_gaps = []
        
        # Calculate continuous time axis with collapsed inter-session gaps for this subject
        if session_info:
            # Build time offset mapping for each session
            time_offset = 0  # Cumulative offset to add to timestamps
            session_offsets = {}  # Maps session date to its time offset
            
            for sess_idx, sess in enumerate(session_info):
                runs = sess['runs']
                session_date = sess['date']
                
                if sess_idx == 0:
                    # First session: no offset, use actual start time
                    first_run_start = pd.to_datetime(runs[0]['start_time']).tz_localize(None)
                    global_start_time = first_run_start
                    session_offsets[session_date] = 0
                else:
                    # Get end of previous session
                    prev_sess = session_info[sess_idx - 1]
                    prev_last_run = prev_sess['runs'][-1]
                    prev_end = pd.to_datetime(prev_last_run['end_time']).tz_localize(None)
                    
                    # Get start of current session
                    curr_start = pd.to_datetime(runs[0]['start_time']).tz_localize(None)
                    
                    # Calculate the gap between sessions (in seconds)
                    inter_session_gap = (curr_start - prev_end).total_seconds()
                    
                    # Add this gap to our cumulative offset (we want to subtract it from timestamps)
                    time_offset += inter_session_gap - 1  # Keep 1 second buffer
                    session_offsets[session_date] = time_offset
                    
                    # Mark session boundary (time where new session "starts" in plot)
                    boundary_seconds = (prev_end - global_start_time).total_seconds() - session_offsets[prev_sess['date']] + 1
                    subj_session_starts.append(boundary_seconds)
                
                # Calculate gaps within this session
                for run in runs:
                    if 'gap_to_next_run' in run and run['gap_to_next_run']:
                        gap_str = run['gap_to_next_run']
                        try:
                            # Parse format like "0:27:53.342496"
                            parts = gap_str.split(':')
                            if len(parts) == 3:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = float(parts[2])
                                gap_duration = hours * 3600 + minutes * 60 + seconds
                            else:
                                gap_duration = float(gap_str)
                            
                            # Gap starts at end_time of this run
                            run_end = pd.to_datetime(run['end_time']).tz_localize(None)
                            gap_start_seconds = (run_end - global_start_time).total_seconds() - session_offsets[session_date]
                            gap_end_seconds = gap_start_seconds + gap_duration
                            
                            subj_gaps.append((gap_start_seconds, gap_end_seconds))
                        except Exception:
                            pass
            
            # Apply time offsets to trial data
            combined['time_seconds'] = combined.apply(
                lambda row: (row['sequence_start'] - global_start_time).total_seconds() - session_offsets.get(row['date'], 0),
                axis=1
            )
        else:
            # Fallback if no manifest info
            global_start_time = combined['sequence_start'].iloc[0]
            combined['time_seconds'] = (combined['sequence_start'] - global_start_time).dt.total_seconds()
        
        # Add grey shading for gaps between runs (subject-specific)
        for gap_start, gap_end in subj_gaps:
            ax.axvspan(gap_start, gap_end, alpha=0.2, color='gray', zorder=0)
        
        # Add vertical dashed lines at session boundaries (subject-specific, colored)
        for boundary in subj_session_starts:
            ax.axvline(x=boundary, color=colors[subj_idx], linestyle='--', linewidth=1, alpha=0.7)
        
        if split_days:
            # Reset count for each day
            combined['day_group'] = combined['date'].astype(str)
            combined['cumulative_rewards'] = combined.groupby('day_group').cumcount() + 1
            
            # Plot each day separately (no connecting lines)
            for day in combined['day_group'].unique():
                day_data = combined[combined['day_group'] == day]
                ax.plot(day_data['time_seconds'], 
                       day_data['cumulative_rewards'],
                       color=colors[subj_idx],
                       marker='o',
                       markersize=3,
                       label=f'Subject {subjid}' if day == combined['day_group'].iloc[0] else None)
        else:
            # Continuous accumulation across days
            combined['cumulative_rewards'] = range(1, len(combined) + 1)
            ax.plot(combined['time_seconds'], 
                   combined['cumulative_rewards'],
                   color=colors[subj_idx],
                   marker='o',
                   markersize=3,
                   label=f'Subject {subjid}')
    
    ax.set_xlabel('Time (seconds from session start)')
    ax.set_ylabel('Cumulative Rewards')
    ax.set_title(title if title else 'Accumulated Rewards Over Time')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def plot_movement_trace(subjid, date, smooth_window=10, linewidth=1, alpha=0.5, figsize=(10, 10), 
                       xlim=None, ylim=None, invert_y=True, title=None, save_path=None):
    """
    Plot animal movement trace from ezTrack location tracking CSV.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    smooth_window : int, optional
        Number of frames for moving average smoothing (default: 10)
        Set to 1 for no smoothing
    linewidth : float, optional
        Width of the trace line (default: 1)
    alpha : float, optional
        Transparency of the trace (0-1, default: 0.5)
    figsize : tuple, optional
        Figure size (width, height) (default: (10, 10))
    xlim : tuple, optional
        X-axis limits (min, max). If None, auto-scales to data
    ylim : tuple, optional
        Y-axis limits (min, max). If None, auto-scales to data
    invert_y : bool, optional
        Whether to invert Y-axis to match video coordinates (default: True)
    title : str, optional
        Plot title. If None, uses default
    save_path : str or Path, optional
        If provided, saves the plot to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Build path to combined tracking CSV
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # Find subject directory
    sub_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]
    
    # Find session directory
    date_str = str(date)
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]
    
    # Find combined tracking CSV
    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for combined tracking file (exclude macOS metadata files)
    tracking_files = [f for f in results_dir.glob(f"*_combined_tracking_with_timestamps.csv") 
                    if not f.name.startswith('._')]
    if not tracking_files:
        raise FileNotFoundError(
            f"No combined tracking file found in {results_dir}\n"
            f"Run add_timestamps_to_tracking({subjid}, {date}) first to create it."
        )

    csv_path = tracking_files[0]
    print(f"Loading tracking data from: {csv_path.name}")

    # Load the tracking data with encoding handling
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
        print("Note: Used latin1 encoding")

    # Extract X and Y coordinates
    x = df['X'].values
    y = df['Y'].values
    
    # Apply moving average smoothing
    if smooth_window > 1:
        x_smooth = pd.Series(x).rolling(window=smooth_window, center=True, min_periods=1).mean().values
        y_smooth = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean().values
    else:
        x_smooth = x
        y_smooth = y
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the trace
    ax.plot(x_smooth, y_smooth, color='black', linewidth=linewidth, alpha=alpha)
    
    # Set axis properties
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(title if title else f'Animal Movement Trace - Subject {subjid}, {date_str}')
    
    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Invert y-axis to match video coordinates (origin at top-left)
    if invert_y:
        ax.invert_yaxis()
    
    # Equal aspect ratio for proper spatial representation
    ax.set_aspect('equal', adjustable='box')
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add grid for reference
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def load_tracking_with_behavior(subjid, date):
    """
    Load combined tracking data and behavior results for a session.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    
    Returns:
    --------
    dict containing:
        - 'tracking': pd.DataFrame with tracking data (X, Y, time)
        - 'behavior': dict from load_session_results()
        - 'tracking_labeled': pd.DataFrame with added 'in_trial' column
    """
    # Load tracking data
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    sub_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]
    
    date_str = str(date)
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]
    
    results_dir = session_dir / "saved_analysis_results"
    tracking_files = [f for f in results_dir.glob(f"*_combined_tracking_with_timestamps.csv") 
                      if not f.name.startswith('._')]
    if not tracking_files:
        raise FileNotFoundError(
            f"No combined tracking file found. Run add_timestamps_to_tracking({subjid}, {date}) first."
        )
    
    # Load tracking CSV
    try:
        tracking = pd.read_csv(tracking_files[0], encoding='utf-8')
    except UnicodeDecodeError:
        tracking = pd.read_csv(tracking_files[0], encoding='latin1')
    
    # Convert time column to datetime
    tracking['time'] = pd.to_datetime(tracking['time'])
    
    # Load behavior data
    behavior = load_session_results(subjid, date)
    
    # Label tracking frames as in_trial or not
    tracking_labeled = tracking.copy()
    tracking_labeled['in_trial'] = False
    tracking_labeled['trial_type'] = None
    
    # Get all completed trials (can customize this to use different trial types)
    trials = behavior.get('completed_sequences', pd.DataFrame())
    
    if not trials.empty:
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        
        # Mark frames that fall within any trial
        for idx, trial in trials.iterrows():
            mask = (tracking_labeled['time'] >= trial['sequence_start']) & \
                   (tracking_labeled['time'] <= trial['sequence_end'])
            tracking_labeled.loc[mask, 'in_trial'] = True
            tracking_labeled.loc[mask, 'trial_type'] = trial.get('trial_type', 'trial')
    
    return {
        'tracking': tracking,
        'behavior': behavior,
        'tracking_labeled': tracking_labeled
    }


def plot_movement_by_trial_state(subjid, date, smooth_window=10, linewidth=1, alpha=0.6, 
                                 figsize=(10, 10), xlim=None, ylim=None, invert_y=True,
                                 in_trial_color='blue', out_trial_color='gray',
                                 title=None, save_path=None, show=True):
    """
    Plot animal movement trace colored by trial state (in-trial vs between-trials).
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    smooth_window : int, optional
        Number of frames for moving average smoothing (default: 10)
    linewidth : float, optional
        Width of the trace line (default: 1)
    alpha : float, optional
        Transparency of the trace (0-1, default: 0.6)
    figsize : tuple, optional
        Figure size (width, height) (default: (10, 10))
    xlim, ylim : tuple, optional
        Axis limits (min, max). If None, auto-scales
    invert_y : bool, optional
        Whether to invert Y-axis (default: True)
    in_trial_color : str, optional
        Color for in-trial segments (default: 'blue')
    out_trial_color : str, optional
        Color for between-trial segments (default: 'gray')
    title : str, optional
        Plot title
    save_path : str or Path, optional
        If provided, saves the plot
    show : bool, optional
        If True, displays the plot (default: True)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load data
    data = load_tracking_with_behavior(subjid, date)
    df = data['tracking_labeled']
    
    # Apply smoothing
    if smooth_window > 1:
        df['X_smooth'] = df['X'].rolling(window=smooth_window, center=True, min_periods=1).mean()
        df['Y_smooth'] = df['Y'].rolling(window=smooth_window, center=True, min_periods=1).mean()
    else:
        df['X_smooth'] = df['X']
        df['Y_smooth'] = df['Y']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot in segments based on trial state
    # This ensures continuous lines within each state
    current_state = None
    segment_x = []
    segment_y = []
    
    for idx, row in df.iterrows():
        if row['in_trial'] != current_state:
            # State changed, plot accumulated segment
            if segment_x:
                color = in_trial_color if current_state else out_trial_color
                label = 'In trial' if current_state else 'Between trials'
                # Only add label once per state
                if current_state is not None:
                    existing_labels = [t.get_label() for t in ax.lines]
                    if label in existing_labels:
                        label = None
                
                ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, 
                       alpha=alpha, label=label)
            
            # Start new segment
            segment_x = [row['X_smooth']]
            segment_y = [row['Y_smooth']]
            current_state = row['in_trial']
        else:
            # Continue current segment
            segment_x.append(row['X_smooth'])
            segment_y.append(row['Y_smooth'])
    
    # Plot final segment
    if segment_x:
        color = in_trial_color if current_state else out_trial_color
        label = 'In trial' if current_state else 'Between trials'
        existing_labels = [t.get_label() for t in ax.lines]
        if label in existing_labels:
            label = None
        ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, 
               alpha=alpha, label=label)
    
    # Set properties
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(title if title else f'Movement by Trial State - Subject {subjid}, {date}')
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if invert_y:
        ax.invert_yaxis()
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def _load_tracking_and_behavior(subjid, date, tracking_source='auto'):
    """
    Load combined tracking CSV and behavior results for a session.
    Supports both ezTrack (*_combined_tracking_with_timestamps.csv) 
    and SLEAP (*_combined_sleap_tracking_timestamps.csv) formats.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date
    tracking_source : str, optional
        'auto' (default): prefer SLEAP if available, else ezTrack
        'sleap': only load SLEAP
        'eztrack': only load ezTrack
    
    Returns:
    --------
    tracking_df, behavior_dict
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()

    sub_str = f"sub-{str(subjid).zfill(3)}"
    date_str = str(date)

    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]

    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]

    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find tracking files
    sleap_files = [f for f in results_dir.glob("*_combined_sleap_tracking_timestamps.csv")
                   if not f.name.startswith("._")]
    eztrack_files = [f for f in results_dir.glob("*_combined_tracking_with_timestamps.csv")
                     if not f.name.startswith("._")]
    
    csv_path = None
    source_used = None
    
    if tracking_source == 'auto':
        # Prefer SLEAP if available
        if sleap_files:
            csv_path = sleap_files[0]
            source_used = 'sleap'
        elif eztrack_files:
            csv_path = eztrack_files[0]
            source_used = 'eztrack'
    elif tracking_source == 'sleap':
        if sleap_files:
            csv_path = sleap_files[0]
            source_used = 'sleap'
    elif tracking_source == 'eztrack':
        if eztrack_files:
            csv_path = eztrack_files[0]
            source_used = 'eztrack'
    
    if csv_path is None:
        raise FileNotFoundError(
            f"No tracking file found for {tracking_source}. "
            f"Available: {len(sleap_files)} SLEAP, {len(eztrack_files)} ezTrack"
        )

    try:
        tracking = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        tracking = pd.read_csv(csv_path, encoding='latin1')

    tracking['time'] = pd.to_datetime(tracking['time'], errors='coerce')

    # For SLEAP data: use 'centroid_x' and 'centroid_y' if available, else 'X'/'Y'
    if source_used == 'sleap':
        if 'centroid_x' in tracking.columns and 'centroid_y' in tracking.columns:
            tracking['X'] = tracking['centroid_x']
            tracking['Y'] = tracking['centroid_y']
        elif 'X' not in tracking.columns:
            # Try to find any x/y columns
            x_cols = [c for c in tracking.columns if 'x' in c.lower() and 'score' not in c.lower()]
            y_cols = [c for c in tracking.columns if 'y' in c.lower() and 'score' not in c.lower()]
            if x_cols and y_cols:
                tracking['X'] = tracking[x_cols[0]]
                tracking['Y'] = tracking[y_cols[0]]

    behavior = load_session_results(subjid, date)
    
    print(f"Loaded {source_used.upper()} tracking: {len(tracking)} frames from {csv_path.name}")
    
    return tracking, behavior

def plot_movement_with_behavior(
    subjid, date,
    mode="simple",                # "simple" | "trial_state" | "last_odor" | "time_windows" | "trial_windows"
    time_windows=None,            # list of ("HH:MM:SS","HH:MM:SS")
    trial_windows=None,           # list of (start, end). negatives allowed, e.g. (-20, None) = last 20..last
    smooth_window=10, linewidth=1, alpha=0.6,
    figsize=(10, 10), xlim=None, ylim=None, invert_y=True,
    last_odor_colors=None,        # {'A':'red','B':'blue','other':'gray'}
    title=None, save_path=None, show=True
):
    """
    Minimal modes:
      - simple: baseline trace
      - trial_state: in-trial vs outside-trial
      - last_odor: within-trial colored by last odor (A vs B)
      - time_windows: plot only movement within provided clock-time windows (can be multiple)
      - trial_windows: plot only trials in provided windows; supports negatives from the end
    Also auto-creates per-condition facet plots when multiple categories/windows exist.
    """
    assert mode in {"simple", "trial_state", "by_odor", "by_odor_rew", "by_odor_outcome", "time_windows", "trial_windows", "trial_windows_rew"}
    tracking, behavior = _load_tracking_and_behavior(subjid, date)
    def _infer_last_odor_column(trials: pd.DataFrame) -> str | None:
        """
        Try to find a column that represents the last odor identity.
        Returns column name or None.
        """
        cols = set(trials.columns.str.lower())

        # Direct matches
        for cand in ["last_odor", "lastodor", "last_odor_name", "final_odor", "finalodor"]:
            for c in trials.columns:
                if c.lower() == cand:
                    return c

        # If there are odorN columns (odor1, odor2, ...), we will derive per-row later
        has_odorN = any(re.match(r"^odor\d+$", c.lower()) for c in trials.columns)
        if has_odorN:
            return None  # signal to derive from odorN columns

        # If there's an 'odors' list/JSON column
        for c in trials.columns:
            if c.lower() == "odors":
                return c

        return None
    def _last_odor_series(trials: pd.DataFrame) -> pd.Series:
        """
        Build a per-trial Series with the last odor identity.
        Handles:
        - explicit last_odor-like columns,
        - odorN columns (takes last non-null),
        - 'odors' list/JSON.
        Falls back to 'other' if not resolvable.
        """
        if trials.empty:
            return pd.Series([], dtype=object, index=trials.index)

        col = _infer_last_odor_column(trials)
        s = pd.Series(index=trials.index, dtype=object)

        if col is not None and col.lower() not in {"odors"}:
            # Direct column present
            s = trials[col].astype(object).fillna("other")
            return s

        # Try odorN columns
        odorN_cols = sorted(
            [c for c in trials.columns if re.match(r"^odor\d+$", c.lower())],
            key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0
        )
        if odorN_cols:
            def last_non_null(row):
                vals = [row[c] for c in odorN_cols if pd.notna(row[c])]
                return vals[-1] if vals else "other"
            return trials.apply(last_non_null, axis=1)

        # Try 'odors' as list/JSON
        if col is not None and col.lower() == "odors":
            def from_list(v):
                try:
                    if isinstance(v, str):
                        # maybe JSON-like
                        import json
                        v2 = json.loads(v)
                    else:
                        v2 = v
                    if isinstance(v2, (list, tuple)) and len(v2) > 0:
                        return v2[-1]
                except Exception:
                    pass
                return "other"
            return trials[col].apply(from_list)

        return s.fillna("other")
    # Smoothing
    if smooth_window > 1:
        tracking['X_smooth'] = pd.Series(tracking['X']).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        tracking['Y_smooth'] = pd.Series(tracking['Y']).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
    else:
        tracking['X_smooth'] = tracking['X']
        tracking['Y_smooth'] = tracking['Y']

    fig, ax = plt.subplots(figsize=figsize)

    def _plot_segments_by_mask(df, mask, color, label=None, axes=None):
        # Plot continuous segments where mask is True
        target_ax = axes if axes is not None else ax
        m = mask.fillna(False).astype(bool)
        if m.sum() == 0:
            return
        seg_id = (m != m.shift(1, fill_value=False)).cumsum()
        first = True
        for _, g in df[m].groupby(seg_id[m]):
            target_ax.plot(g['X_smooth'].values, g['Y_smooth'].values,
                           color=color, linewidth=linewidth, alpha=alpha,
                           label=(label if first else None))
            first = False

    facet_plots = []  # collect per-condition masks to facet later

    if mode == "simple":
        ax.plot(tracking['X_smooth'].values, tracking['Y_smooth'].values,
                color='black', linewidth=linewidth, alpha=alpha, label='Movement trace')

    elif mode == "trial_state":
        trials = behavior.get('initiated_sequences', pd.DataFrame())
        in_trial = pd.Series(False, index=tracking.index)
        if not trials.empty:
            trials = trials.copy()
            trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
            trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
            t_time = tracking['time']
            for _, tr in trials.iterrows():
                in_trial |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))

        colors = {'in': 'blue', 'out': 'gray'}
        _plot_segments_by_mask(tracking, in_trial, colors['in'], label='In trial')
        _plot_segments_by_mask(tracking, ~in_trial, colors['out'], label='Between trials')
        facet_plots = [
            ('In trial', in_trial, colors['in']),
            ('Between trials', ~in_trial, colors['out']),
        ]

        
    elif mode in ["by_odor", "by_odor_rew"]:
        if mode == "by_odor":
            comps = behavior.get('completed_sequences', pd.DataFrame())
        elif mode == "by_odor_rew":
            comps = behavior.get('completed_sequence_rewarded', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; last_odor plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")
        
        if last_odor_colors is None:
            last_odor_colors = {'OdorA': 'red', 'OdorB': 'blue', 'other': 'lightgray'}

        # Map each tracking frame to its odor category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        
        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
        
        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        
        unique_odors = sorted(trial_odor_filtered.unique())
        
        # Plot combined view with all odors
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            _plot_segments_by_mask(tracking, full_mask, color, label=f"Odors: {odor}", axes=ax)
        
        # Create facet plots for individual odors
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            facet_plots.append((f"{odor}", full_mask, color))

    elif mode == "by_odor_outcome":
        comps = behavior.get('completed_sequences', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; by_odor_outcome plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")

        # Try to infer the rewarded/outcome column
        rewarded_col = None
        for cand in ['rewarded', 'is_rewarded', 'outcome', 'correct', 'success']:
            if cand in comps.columns:
                rewarded_col = cand
                break
        if rewarded_col is None:
            # Fallback: if completed_sequence_rewarded exists, mark those trials as rewarded
            rewarded_trials = behavior.get('completed_sequence_rewarded', pd.DataFrame())
            if not rewarded_trials.empty and 'sequence_start' in rewarded_trials.columns:
                rewarded_starts = set(pd.to_datetime(rewarded_trials['sequence_start']))
                comps['__rewarded'] = comps['sequence_start'].isin(rewarded_starts)
                rewarded_col = '__rewarded'
            else:
                raise ValueError("No rewarded/outcome column found in completed_sequences and cannot infer from completed_sequence_rewarded.")

        if last_odor_colors is None:
            last_odor_colors = {'OdorA': 'red', 'OdorB': 'blue', 'other': 'lightgray'}

        # Map each tracking frame to its odor and outcome category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        trial_outcome = pd.Series('', index=tracking.index, dtype=object)

        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
            is_rewarded = bool(tr[rewarded_col])
            trial_outcome.loc[mask] = 'rewarded' if is_rewarded else 'not_rewarded'

        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        trial_outcome_filtered = trial_outcome[in_trial_mask]

        unique_odors = sorted(trial_odor_filtered.unique())

        # Plot combined view: correct in color, incorrect/timeout in grey
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            # Rewarded trials for this odor
            rewarded_mask = odor_mask & (trial_outcome_filtered == 'rewarded')
            # Not rewarded trials for this odor
            not_rewarded_mask = odor_mask & (trial_outcome_filtered == 'not_rewarded')

            full_rewarded_mask = pd.Series(False, index=tracking.index)
            full_rewarded_mask.loc[rewarded_mask.index[rewarded_mask]] = True
            full_not_rewarded_mask = pd.Series(False, index=tracking.index)
            full_not_rewarded_mask.loc[not_rewarded_mask.index[not_rewarded_mask]] = True

            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            _plot_segments_by_mask(tracking, full_rewarded_mask, color, label=f"{odor} (correct)", axes=ax)
            _plot_segments_by_mask(tracking, full_not_rewarded_mask, 'lightgray', label=f"{odor} (incorrect/timeout)", axes=ax)

        # Create facet plots for each odor/outcome
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            rewarded_mask = odor_mask & (trial_outcome_filtered == 'rewarded')
            not_rewarded_mask = odor_mask & (trial_outcome_filtered == 'not_rewarded')

            full_rewarded_mask = pd.Series(False, index=tracking.index)
            full_rewarded_mask.loc[rewarded_mask.index[rewarded_mask]] = True
            full_not_rewarded_mask = pd.Series(False, index=tracking.index)
            full_not_rewarded_mask.loc[not_rewarded_mask.index[not_rewarded_mask]] = True

            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            # Use a slightly darker grey for incorrect/timeout
            dark_grey = '#888888'
            facet_plots.append((
                f"{odor} by Outcome",
                [  # list of (mask, color, label)
                    (full_rewarded_mask, color, "Correct"),
                    (full_not_rewarded_mask, dark_grey, "Incorrect/Timeout")
                ]
            ))

    elif mode == "time_windows":
        if not time_windows:
            raise ValueError("time_windows must be provided for mode='time_windows'.")
        # Normalize input
        if isinstance(time_windows, tuple):
            time_windows = [time_windows]
        if isinstance(time_windows, str):
            parts = [p.strip() for p in time_windows.split(",")]
            if len(parts) != 2:
                raise ValueError("time_windows string must be 'HH:MM:SS, HH:MM:SS'")
            time_windows = [tuple(parts)]

        t = tracking['time']
        tz = t.dt.tz if hasattr(t.dt, 'tz') else None
        unique_dates = sorted(pd.to_datetime(t.dt.date).unique())
        cmap = plt.cm.Set2
        facet_plots = []
        for i, (ts, te) in enumerate(time_windows):
            color = cmap(i % 8)
            mask = pd.Series(False, index=t.index)
            for d in unique_dates:
                start_dt = pd.to_datetime(f"{pd.to_datetime(d).date()} {ts}")
                end_dt = pd.to_datetime(f"{pd.to_datetime(d).date()} {te}")
                if tz is not None:
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.tz_localize(tz)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.tz_localize(tz)
                mask |= ((t >= start_dt) & (t <= end_dt))
            _plot_segments_by_mask(tracking, mask, color, label=f"Window {i+1}: {ts}-{te}")
            facet_plots.append((f"Window {i+1}: {ts}-{te}", mask, color))

    elif mode in ["trial_windows", "trial_windows_rew"]:
        if mode == "trial_windows":
            trials = behavior.get('initiated_sequences', pd.DataFrame())
        elif mode == "trial_windows_rew":
            trials = behavior.get('completed_sequence_rewarded', pd.DataFrame())
        if trials.empty:
            raise ValueError(f"{mode} requires appropriate trial data.")
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        # Sort by time
        trials = trials.sort_values('sequence_start').reset_index(drop=True)

        # Normalize input to list
        if isinstance(trial_windows, tuple):
            trial_windows = [trial_windows]

        n = len(trials)

        cmap = plt.cm.Dark2
        facet_plots = []
        t_time = tracking['time']
        for i, (start_idx, end_idx) in enumerate(trial_windows):
            # If end_idx is 0 or None, select to the end
            if end_idx in [0, None]:
                sel = trials.iloc[start_idx:]
            else:
                sel = trials.iloc[start_idx:end_idx]
            
            if sel.empty:
                continue
            
            mask = pd.Series(False, index=tracking.index)
            for _, tr in sel.iterrows():
                mask |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))
            
            color = cmap(i % 8)
            # Display label using actual indices for clarity
            actual_indices = sel.index.tolist()
            label = f"Trials {actual_indices[0]}-{actual_indices[-1]}"
            _plot_segments_by_mask(tracking, mask, color, label=label)
            facet_plots.append((label, mask, color))

    # Axes/styling (overlay)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    if title is None:
        title = f"Movement - Subject {subjid}, {date} ({mode})"
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    # Faceted per-condition plots when multiple categories/windows exist
    if len(facet_plots) > 1:
        n = len(facet_plots)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        facet_fig, facet_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*1.2, figsize[1]*1.2))
        # Ensure facet_axes is always 2D
        if nrows == 1 and ncols == 1:
            facet_axes = np.array([[facet_axes]])
        elif nrows == 1 or ncols == 1:
            facet_axes = facet_axes.reshape(nrows, ncols)
        
        facet_axes_flat = facet_axes.flatten()
        
        for i, facet in enumerate(facet_plots):
            ax_i = facet_axes_flat[i]
            label = facet[0]
            mask_or_list = facet[1]
            # If mask_or_list is a list, it's the new format (by_odor_outcome)
            if isinstance(mask_or_list, list):
                for mask, color, sublabel in mask_or_list:
                    _plot_segments_by_mask(tracking, mask, color, label=sublabel, axes=ax_i)
            else:
                # Old format: (label, mask, color)
                mask = mask_or_list
                color = facet[2]
                _plot_segments_by_mask(tracking, mask, color, label=label, axes=ax_i)
            ax_i.set_title(label)
            ax_i.set_xlabel('X Position (px)')
            ax_i.set_ylabel('Y Position (px)')
            if xlim:
                ax_i.set_xlim(xlim)
            if ylim:
                ax_i.set_ylim(ylim)
            if invert_y:
                ax_i.invert_yaxis()
            ax_i.set_aspect('equal', adjustable='box')
            ax_i.grid(True, alpha=0.3)
        
        # Hide unused axes if any
        for j in range(len(facet_plots), len(facet_axes_flat)):
            facet_axes_flat[j].axis('off')
        
        facet_fig.suptitle(f"Per-condition views - Subject {subjid}, {date} ({mode})")
        facet_fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()

    return fig, ax


def plot_choice_history(subjid, dates=None, figsize=(16, 8), title=None, save_path=None):
    """
    Plot choice history over time for one or more sessions.
    
    - Y-axis: Choice direction (A=red up, B=blue down)
    - X-axis: Time
    - Rewarded trials: solid line with circle marker at end
    - Completed unrewarded trials: dotted line, no marker
    - Aborted trials: grey line going up
    - Hidden Rule trials: yellow line
      - HR rewarded: solid yellow line with circle marker
      - HR missed/unrewarded: dotted yellow line, no marker
    - Multiple sessions: time gaps collapsed, session boundaries marked with grey dotted lines
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : list, tuple, or None
        Specific dates [20250101, 20250102] or range (20250101, 20250110)
        If None, plots all available dates
    figsize : tuple, optional
        Figure size (default: (16, 8))
    title : str, optional
        Plot title. If None, uses default
    save_path : str or Path, optional
        If provided, saves the plot to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    subj_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subject_dir = subject_dirs[0]
    
    # Get session directories
    ses_dirs = _filter_session_dirs(subject_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")
    
    # Collect all trials across sessions
    all_trials = []
    
    for session_idx, ses_dir in enumerate(ses_dirs):
        date_str = ses_dir.name.split("_date-")[-1]
        
        try:
            results = load_session_results(subjid, date_str)
        except Exception as e:
            print(f"Warning: Could not load session {date_str}: {e}")
            continue
        
        # Load completed and aborted trials
        comp_rew = results.get('completed_sequence_rewarded', pd.DataFrame())
        comp_unr = results.get('completed_sequence_unrewarded', pd.DataFrame())
        comp_tmo = results.get('completed_sequence_reward_timeout', pd.DataFrame())
        aborted = results.get('aborted_sequences', pd.DataFrame())
        
        # Load Hidden Rule trials
        hr_rewarded = results.get('completed_sequence_HR_rewarded', pd.DataFrame())
        hr_unrewarded = results.get('completed_sequence_HR_unrewarded', pd.DataFrame())
        hr_timeout = results.get('completed_sequence_HR_reward_timeout', pd.DataFrame())
        hr_missed = results.get('completed_sequences_HR_missed', pd.DataFrame())
        aborted_hr = results.get('aborted_sequences_HR', pd.DataFrame())
        
        # Process completed rewarded trials
        if not comp_rew.empty:
            for idx, row in comp_rew.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'rewarded',
                    'is_hr': False,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process completed unrewarded trials
        if not comp_unr.empty:
            for idx, row in comp_unr.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'unrewarded',
                    'is_hr': False,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process timeout trials
        if not comp_tmo.empty:
            for idx, row in comp_tmo.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'timeout',
                    'is_hr': False,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process aborted trials
        if not aborted.empty:
            for idx, row in aborted.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor_name', 'Unknown'),
                    'trial_type': 'aborted',
                    'is_hr': False,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process HR rewarded trials
        if not hr_rewarded.empty:
            for idx, row in hr_rewarded.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'hr_rewarded',
                    'is_hr': True,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process HR unrewarded trials
        if not hr_unrewarded.empty:
            for idx, row in hr_unrewarded.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'hr_unrewarded',
                    'is_hr': True,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process HR timeout trials
        if not hr_timeout.empty:
            for idx, row in hr_timeout.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'hr_timeout',
                    'is_hr': True,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process HR missed trials
        if not hr_missed.empty:
            for idx, row in hr_missed.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'hr_missed',
                    'is_hr': True,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
        
        # Process aborted HR trials
        if not aborted_hr.empty:
            for idx, row in aborted_hr.iterrows():
                all_trials.append({
                    'sequence_start': pd.to_datetime(row['sequence_start']),
                    'last_odor': row.get('last_odor', 'Unknown'),
                    'trial_type': 'hr_aborted',
                    'is_hr': True,
                    'date_str': date_str,
                    'session_idx': session_idx
                })
    
    if not all_trials:
        print(f"No trials found for subject {subjid}")
        return None, None
    
    trials_df = pd.DataFrame(all_trials)
    trials_df = trials_df.sort_values('sequence_start').reset_index(drop=True)
    
    # Set global start time from first trial
    global_start_time = trials_df['sequence_start'].iloc[0]
    
    # Calculate time offsets for each session to collapse inter-session gaps
    session_time_offsets = {}
    session_boundaries = []
    
    time_offset = 0
    
    for session_idx in sorted(trials_df['session_idx'].unique()):
        session_data = trials_df[trials_df['session_idx'] == session_idx]
        
        if session_idx == 0:
            session_time_offsets[session_idx] = 0
        else:
            prev_session_data = trials_df[trials_df['session_idx'] == session_idx - 1]
            
            if not prev_session_data.empty and not session_data.empty:
                prev_end = prev_session_data['sequence_start'].max()
                curr_start = session_data['sequence_start'].min()
                
                gap = (curr_start - prev_end).total_seconds()
                time_offset += gap
                session_time_offsets[session_idx] = time_offset
                
                prev_time_in_plot = (prev_end - global_start_time).total_seconds() - session_time_offsets[session_idx - 1]
                session_boundaries.append((prev_time_in_plot, session_idx))
    
    # Calculate plot time for each trial
    trials_df['time_in_plot'] = trials_df.apply(
        lambda row: (row['sequence_start'] - global_start_time).total_seconds() - session_time_offsets[row['session_idx']],
        axis=1
    )
    
    # Extract odor letter (e.g., 'OdorA' -> 'A')
    def extract_odor_letter(odor_str):
        if pd.isna(odor_str):
            return 'Unknown'
        odor_str = str(odor_str)
        if odor_str.startswith('Odor'):
            return odor_str.replace('Odor', '')
        return odor_str
    
    trials_df['odor_letter'] = trials_df['last_odor'].apply(extract_odor_letter)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    odor_colors = {
        'A': '#E53935',      # Bright red
        'B': '#00796B',      # Darker teal
        'HR': '#FFD700'      # Gold/yellow
    }
    
    odor_direction = {'A': 1, 'B': -1}  # A goes up, B goes down
    
    # Plot each trial
    for idx, trial in trials_df.iterrows():
        x = trial['time_in_plot']
        odor = trial['odor_letter']
        trial_type = trial['trial_type']
        is_hr = trial['is_hr']
        
        if odor not in odor_colors and odor != 'Unknown':
            odor = 'Unknown'
        
        # Determine color based on trial type
        if is_hr:
            color = odor_colors['HR']
        else:
            color = odor_colors.get(odor, '#999999')
        
        # Determine direction from odor
        direction = odor_direction.get(odor, 1)
        
        # Determine line style and marker based on reward status
        if trial_type == 'rewarded' or trial_type == 'hr_rewarded':
            linestyle = '-'
            linewidth = 2
            alpha = 0.85
            has_marker = True
        elif trial_type in ['unrewarded', 'timeout', 'hr_unrewarded', 'hr_timeout', 'hr_missed']:
            linestyle = ':'
            linewidth = 2
            alpha = 0.5
            has_marker = False
        else:
            # aborted or hr_aborted
            linestyle = '-'
            linewidth = 1.5
            alpha = 0.6
            has_marker = False
        
        # Plot the trial
        if trial_type == 'aborted':
            # Regular aborted: grey line
            ax.plot([x, x], [0, 0.6], color='#888888', linewidth=linewidth, alpha=alpha, zorder=1)
            ax.scatter([x], [0.6], color='#888888', s=15, marker='^', alpha=alpha, zorder=2)
        
        elif trial_type == 'hr_aborted':
            # HR aborted: yellow line with direction from odor
            y_end = 1.0 * direction
            ax.plot([x, x], [0, y_end], color=color, linewidth=linewidth, alpha=alpha, zorder=1, linestyle=linestyle)
            ax.scatter([x], [y_end], color=color, s=15, marker='^', alpha=alpha, zorder=2)
        
        else:
            # Completed trials (regular or HR)
            y_end = 1.0 * direction
            
            # HR trials on top
            line_zorder = 2 if is_hr else 1
            marker_zorder = 4 if is_hr else 3
            
            ax.plot([x, x], [0, y_end], color=color, linewidth=linewidth, 
                   linestyle=linestyle, alpha=alpha, zorder=line_zorder)
            
            if has_marker:
                ax.scatter([x], [y_end], color=color, s=40, marker='o', 
                          edgecolors='black', linewidth=0.8, zorder=marker_zorder)
    
    # Draw session boundaries
    for boundary_time, session_idx in session_boundaries:
        ax.axvline(x=boundary_time, color='grey', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
    
    # Format axes
    ax.axhline(y=0, color='black', linewidth=2, alpha=0.8)
    ax.set_ylim([-1.5, 1.5])
    
    x_min = trials_df['time_in_plot'].min()
    x_max = trials_df['time_in_plot'].max()
    x_padding = (x_max - x_min) * 0.05
    ax.set_xlim([x_min - x_padding, x_max + x_padding])
    
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['B', 'A'], fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Choice', fontsize=12, fontweight='bold')
    
    if title is None:
        title = f"Choice History - Subject {str(subjid).zfill(3)}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#E53935', lw=2.5, linestyle='-', label='Odor A (regular)'),
        Line2D([0], [0], color='#00796B', lw=2.5, linestyle='-', label='Odor B (regular)'),
        Line2D([0], [0], color='#FFD700', lw=2.5, linestyle='-', label='Hidden Rule (HR)'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Rewarded (solid)'),
        Line2D([0], [0], color='black', lw=2, linestyle=':', label='Unrewarded/Missed (dotted)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=5, markeredgecolor='black', label='Rewarded marker', linestyle='none'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)
    
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax



# ================================= Debugging / Testing ================================= #


def get_fa_ratio_a_stats(subjid, dates=None, odors=['C', 'F']):
    """
    Get FA ratio A/(A+B) statistics for specified odors across sessions.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : list, tuple, or None
        Specific dates to include, e.g., [20250811, 20250812] or (20250811, 20251010) for range
    odors : list, optional
        List of odors to include (default: ['C', 'F'])
    
    Returns:
    --------
    DataFrame with columns: date, odor, fa_ratio_a, n_fa_a, n_fa_b, n_total
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses in enumerate(ses_dirs, start=1):
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            
            if not results_dir.exists():
                continue
            
            ab_det = _load_table_with_trial_data(results_dir, "aborted_sequences_detailed")
            if not ab_det.empty:
                needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                ab_det = ab_det[[col for col in needed_cols if col in ab_det.columns]]

            fa_noninit = _load_table_with_trial_data(results_dir, "non_initiated_FA")
            if not fa_noninit.empty:
                needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                fa_noninit = fa_noninit[[col for col in needed_cols if col in fa_noninit.columns]]
            
            # Combine only if we have data
            if ab_det.empty and fa_noninit.empty:
                continue
            
            # Filter and combine FA data (matching plot_fa_ratio_a_over_sessions logic)
            try:
                fa_list = []
                
                if not ab_det.empty and 'fa_label' in ab_det.columns:
                    fa_ab = ab_det[ab_det['fa_label'].astype(str).str.startswith('FA_', na=False)]
                    if not fa_ab.empty:
                        fa_list.append(fa_ab)
                
                if not fa_noninit.empty and 'fa_label' in fa_noninit.columns:
                    fa_ni = fa_noninit[fa_noninit['fa_label'].astype(str).str.startswith('FA_', na=False)]
                    if not fa_ni.empty:
                        fa_list.append(fa_ni)
                
                if not fa_list:
                    continue
                
                fa_all = pd.concat(fa_list, ignore_index=True)
            except Exception as e:
                continue
            
            if fa_all.empty or 'fa_port' not in fa_all.columns or 'last_odor_name' not in fa_all.columns:
                continue
            
            # Calculate FA port ratio for ALL odors (not just the requested ones)
            # Then filter afterward for flexibility
            try:
                for odor in sorted(fa_all['last_odor_name'].dropna().unique()):
                    # Only include requested odors
                    if str(odor) not in [str(o) for o in odors]:
                        continue
                    
                    fa_odor = fa_all[fa_all['last_odor_name'] == odor]
                    n_a = (fa_odor['fa_port'] == 1).sum()
                    n_b = (fa_odor['fa_port'] == 2).sum()
                    n_total = n_a + n_b
                    ratio_a = n_a / n_total if n_total > 0 else np.nan
                    
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor": str(odor),
                        "fa_ratio_a": ratio_a,
                        "n_fa_a": n_a,
                        "n_fa_b": n_b,
                        "n_total": n_total
                    })
            except Exception as e:
                continue
    
    if not rows:
        print(f"No FA data found for subject {subjid} with odors {odors}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"FA Ratio A/(A+B) Summary - Subject {str(subjid).zfill(3)}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}\n")
    
    return df


def plot_fa_ratio_by_hr_position(
    subjid,
    dates=None,
    figsize=(16, 10),
    fa_types='FA_time_in', 
    print_statistics=False,
    exclude_last_pos=False,
    last_odor_num=5
):
    """
    Plot FA Ratio (A-B)/(A+B) by hidden rule odor position across sessions.
    
    For each session and each HR odor, calculates:
    1. FA on HR Odor at HR position
    2. FA at the next odor in sequence (position-independent)
    3. Total FA at or after HR position
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (16, 10))
    fa_types : str or list, optional
        Which FA types to include:
        - 'FA_time_in' : only FA_time_in
        - 'FA_time_in,FA_time_out' : multiple specific types (comma-separated)
        - 'All' : all FA types starting with 'FA_'
        (default: 'FA_time_in')
    print_statistics: bool, optional
        Whether to print a statistic summary table with counts for each
        FA type and position (default: False).
    exclude_last_pos: bool, optional
        If True, exclude FAs where last_odor_position == last_odor_num from all calculations.
        If False (default), include all positions.
    last_odor_num: int
        defines what position last odor is for possible exclusion of rewarded odors 
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # Parse FA type filter
    if isinstance(fa_types, str):
        if fa_types.lower() == 'all':
            fa_filter_fn = lambda fa_label: str(fa_label).startswith('FA_') if pd.notna(fa_label) else False
        else:
            types_list = [t.strip().lower() for t in fa_types.split(',')]
            fa_filter_fn = lambda fa_label: str(fa_label).lower() in types_list if pd.notna(fa_label) else False
    else:
        fa_filter_fn = lambda fa_label: True
    
    rows = []  # {date, session_num, odor_num, hr_odor, category, port_a, port_b, total, ratio}
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses in enumerate(ses_dirs, 1):
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            
            if not results_dir.exists():
                continue
            
            summary_path = results_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                hr_odors = summary.get("params", {}).get("hidden_rule_odors", [])
                if not hr_odors:
                    continue

                df_hr = _load_table_with_trial_data(results_dir, "aborted_sequences_HR")
                df_ab = _load_table_with_trial_data(results_dir, "aborted_sequences_detailed")
                if df_hr.empty or df_ab.empty:
                    continue
                
                # Match HR trials with aborted sequences
                if "sequence_start" not in df_hr.columns or "sequence_start" not in df_ab.columns:
                    continue
                
                hr_with_fa = df_hr[df_hr["sequence_start"].isin(df_ab["sequence_start"])].copy()
                
                # Merge to get FA details
                merged = hr_with_fa.merge(
                    df_ab[["sequence_start", "fa_label", "last_odor_name", "fa_port", "last_odor_position"]],
                    on="sequence_start",
                    how="left"
                )
                
                # Add HR position info and odor_sequence from HR data
                hr_cols_to_merge = ["sequence_start"]
                if "hidden_rule_positions" in df_hr.columns:
                    hr_cols_to_merge.append("hidden_rule_positions")
                if "odor_sequence" in df_hr.columns:
                    hr_cols_to_merge.append("odor_sequence")
                
                if len(hr_cols_to_merge) > 1:
                    merged = merged.merge(
                        df_hr[hr_cols_to_merge],
                        on="sequence_start",
                        how="left",
                        suffixes=('', '_hr')
                    )
                
                # Filter for actual FAs (not nFA) and apply FA type filter
                merged_fa = merged[
                    (merged["fa_label"] != "nFA") & 
                    (merged["fa_label"].apply(fa_filter_fn))
                ].copy()
                
                # Optionally exclude FAs at the specified last_odor_position
                if exclude_last_pos:
                    merged_fa = merged_fa[merged_fa["last_odor_position"] != last_odor_num].copy()
                
                if merged_fa.empty:
                    continue
                
                if "hidden_rule_positions" not in merged_fa.columns:
                    continue
                
                # Helper functions
                def count_ports(data):
                    if data.empty:
                        return 0, 0, 0
                    port_a = (data["fa_port"] == 1).sum()
                    port_b = (data["fa_port"] == 2).sum()
                    total = port_a + port_b
                    return int(port_a), int(port_b), int(total)
                
                def get_hr_position(hr_pos_str):
                    if pd.isna(hr_pos_str):
                        return None
                    try:
                        pos_list = json.loads(str(hr_pos_str))
                        if isinstance(pos_list, list) and len(pos_list) > 0:
                            return int(pos_list[0])
                    except:
                        pass
                    return None
                
                def has_hr_odor_in_sequence(odor_seq, hr_odor):
                    if pd.isna(odor_seq):
                        return False
                    try:
                        seq_list = json.loads(str(odor_seq))
                        return hr_odor in seq_list if isinstance(seq_list, list) else False
                    except:
                        return hr_odor in str(odor_seq)
                
                # Analyze each HR odor
                for odor_num, hr_odor in enumerate(hr_odors, 1):
                    # Filter to trials where this HR odor appears in the sequence
                    if "odor_sequence" in merged_fa.columns:
                        fa_for_this_hr = merged_fa[
                            merged_fa["odor_sequence"].apply(lambda seq: has_hr_odor_in_sequence(seq, hr_odor))
                        ].copy()
                    else:
                        fa_for_this_hr = merged_fa.copy()
                    
                    if fa_for_this_hr.empty:
                        continue
                    
                    # Extract HR position
                    fa_for_this_hr["hr_position"] = fa_for_this_hr["hidden_rule_positions"].apply(get_hr_position)
                    fa_for_this_hr = fa_for_this_hr[fa_for_this_hr["hr_position"].notna()]
                    
                    if fa_for_this_hr.empty:
                        continue
                    
                    # Category 1: FA on HR odor itself at HR position
                    fa_on_hr_odor = fa_for_this_hr[
                        (fa_for_this_hr["last_odor_name"] == hr_odor) & 
                        (fa_for_this_hr["last_odor_position"] == fa_for_this_hr["hr_position"])
                    ].copy()
                    a1, b1, t1 = count_ports(fa_on_hr_odor)
                    ratio1 = (a1 - b1) / t1 if t1 > 0 else np.nan
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor_num": odor_num,
                        "hr_odor": hr_odor,
                        "category": f"On {hr_odor}",
                        "port_a": a1,
                        "port_b": b1,
                        "total": t1,
                        "ratio": ratio1
                    })
                    
                    # Category 2: FA at next odor after HR odor (position-based, not odor-based)
                    # Find the position of the HR odor first
                    fa_one_after = fa_for_this_hr[
                        (fa_for_this_hr["last_odor_position"] == fa_for_this_hr["hr_position"] + 1)
                    ].copy()
                    a2, b2, t2 = count_ports(fa_one_after)
                    ratio2 = (a2 - b2) / t2 if t2 > 0 else np.nan
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor_num": odor_num,
                        "hr_odor": hr_odor,
                        "category": f"After {hr_odor}",
                        "port_a": a2,
                        "port_b": b2,
                        "total": t2,
                        "ratio": ratio2
                    })
                    
                    # Category 3: Total FA at or after HR position
                    fa_total = fa_for_this_hr[
                        (fa_for_this_hr["last_odor_position"] >= fa_for_this_hr["hr_position"])
                    ].copy()
                    a3, b3, t3 = count_ports(fa_total)
                    ratio3 = (a3 - b3) / t3 if t3 > 0 else np.nan
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor_num": odor_num,
                        "hr_odor": hr_odor,
                        "category": f"Total {hr_odor}",
                        "port_a": a3,
                        "port_b": b3,
                        "total": t3,
                        "ratio": ratio3
                    })
                
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
    
    if not rows:
        print("No data found for FA ratio analysis by HR position")
        return None, None
    
    df = pd.DataFrame(rows)
    
    # Get unique HR odors and create subplots: 2 rows (scatter + line) per odor
    unique_odors = sorted(df["hr_odor"].unique())
    n_odors = len(unique_odors)
    
    fig, axes = plt.subplots(2, n_odors, figsize=(figsize[0], figsize[1] * 1.5))
    if n_odors == 1:
        axes = axes.reshape(2, 1)
    
    for ax_idx, hr_odor in enumerate(unique_odors):
        # ===== TOP ROW: Scatter plot by category =====
        ax_scatter = axes[0, ax_idx]
        
        df_odor = df[df["hr_odor"] == hr_odor].copy()
        
        # Define X positions for the 3 categories
        categories = [f"On {hr_odor}", f"After {hr_odor}", f"Total {hr_odor}"]
        x_positions = {cat: i for i, cat in enumerate(categories)}
        
        # Plot each session as a dot
        for cat_idx, category in enumerate(categories):
            df_cat = df_odor[df_odor["category"] == category]
            
            if not df_cat.empty:
                ratios = df_cat["ratio"].dropna()
                if not ratios.empty:
                    x_jitter = np.random.normal(cat_idx, 0.08, size=len(ratios))
                    ax_scatter.scatter(x_jitter, ratios, alpha=0.5, s=40, color='steelblue')
        
        ax_scatter.set_xticks(range(len(categories)))
        ax_scatter.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax_scatter.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
        ax_scatter.set_ylim([-1.1, 1.1])
        ax_scatter.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_scatter.grid(True, alpha=0.3, axis='y')
        ax_scatter.set_title(f'HR Odor: {hr_odor} - By Category\n(Subject {str(subjid).zfill(3)})', 
                    fontsize=12, fontweight='bold')
        
        # ===== BOTTOM ROW: Line plot across sessions =====
        ax_line = axes[1, ax_idx]
        
        # Sort by date to get consecutive sessions
        df_odor_sorted = df_odor.sort_values(by="date")
        
        # Create session mapping: consecutive integers 0 to end
        unique_dates = sorted(df_odor_sorted["date"].unique())
        date_to_session = {d: i for i, d in enumerate(unique_dates)}
        df_odor_sorted["session_idx"] = df_odor_sorted["date"].map(date_to_session)
        
        # Define line properties for each category
        line_config = {
            f"On {hr_odor}": {"color": "blue", "label": f"On {hr_odor}"},
            f"After {hr_odor}": {"color": "green", "label": f"After {hr_odor}"},
            f"Total {hr_odor}": {"color": "black", "label": f"Total {hr_odor}"}
        }
        
        # Plot line for each category
        for category, config in line_config.items():
            df_cat = df_odor_sorted[df_odor_sorted["category"] == category].sort_values(by="session_idx")
            
            if not df_cat.empty and not df_cat["ratio"].isna().all():
                # Get data with values
                df_cat_valid = df_cat[df_cat["ratio"].notna()].copy()
                
                if not df_cat_valid.empty:
                    # Check if there are any gaps (missing sessions)
                    session_indices = df_cat_valid["session_idx"].values
                    all_sessions_present = len(session_indices) == (session_indices[-1] - session_indices[0] + 1)
                    
                    # Use dotted line if there are gaps in the data
                    linestyle = '-' if all_sessions_present else ':'
                    
                    ax_line.plot(df_cat_valid["session_idx"], df_cat_valid["ratio"], 
                                color=config["color"], 
                                label=config["label"],
                                linewidth=2,
                                linestyle=linestyle,
                                marker='o',
                                markersize=5,
                                alpha=0.7)
        
        ax_line.set_xlabel('Session Number', fontsize=11, fontweight='bold')
        ax_line.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
        ax_line.set_ylim([-1.1, 1.1])
        ax_line.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_line.grid(True, alpha=0.3)
        
        # Set x-axis to whole numbers
        max_session = int(df_odor_sorted["session_idx"].max())
        ax_line.set_xticks(range(0, max_session + 1))
        
        ax_line.legend(loc='best', fontsize=9)
        ax_line.set_title(f'HR Odor: {hr_odor} - Across Sessions\n(Subject {str(subjid).zfill(3)})', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if print_statistics:
        # Display summary table sorted by metric, then ratio ascending
        print("\n" + "="*100)
        print("SUMMARY TABLE (SORTED BY METRIC, THEN BY RATIO)")
        print(f"Note: Only showing dates where FA data was found on HR trials (Subject {str(subjid).zfill(3)})")
        print("="*100)
        
        # Create metric name by combining HR odor and category
        df_display = df.copy()
        df_display["metric"] = df_display["hr_odor"] + " - " + df_display["category"]
        
        # Select and order columns
        df_summary = df_display[["date", "metric", "port_a", "port_b", "total", "ratio"]].copy()
        df_summary = df_summary.sort_values(by=["metric", "ratio"], na_position='last')
        
        # Format ratio display
        df_summary["ratio"] = df_summary["ratio"].apply(
            lambda x: f"{x:+.3f}" if not pd.isna(x) else "N/A"
        )
        
        # Display table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df_summary.to_string(index=False))
        
        # Statistics per metric
        print("\n" + "="*100)
        print("STATISTICS (PER METRIC)")
        print("="*100)
        
        # Convert ratio back to numeric for stats
        df_stats = df_display[["metric", "ratio"]].copy()
        
        for metric in sorted(df_stats["metric"].unique()):
            metric_data = df_stats[df_stats["metric"] == metric]["ratio"].dropna()
            
            if len(metric_data) > 0:
                mean_val = metric_data.mean()
                min_val = metric_data.min()
                max_val = metric_data.max()
                std_val = metric_data.std()
                
                print(f"{metric}:")
                print(f"  Mean:  {mean_val:+.3f}")
                print(f"  Min:   {min_val:+.3f}")
                print(f"  Max:   {max_val:+.3f}")
                print(f"  Std:   {std_val:.3f}")
                print()
    
    return fig, axes


def plot_fa_ratio_by_abort_odor(
    subjid,
    dates=None,
    figsize=(18, 8),
    fa_types='FA_time_in'
):
    """
    Plot FA Ratio (A-B)/(A+B) by abortion odor, comparing HR and non-HR aborted sequences.
    
    For each odor where abortion occurred, compares:
    1. Aborted HR trials where abortion happens AFTER the HR odor (not on the HR)
    2. Aborted non-HR trials (no HR present in sequence)
    
    Only includes trials that match the FA type filter. FA Ratio is calculated for each category.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (14, 8))
    fa_types : str or list, optional
        Which FA types to include:
        - 'FA_time_in' : only FA_time_in
        - 'FA_time_in,FA_time_out' : multiple specific types (comma-separated)
        - 'All' : all FA types starting with 'FA_'
        (default: 'FA_time_in')
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # Parse FA type filter
    if isinstance(fa_types, str):
        if fa_types.lower() == 'all':
            fa_filter_fn = lambda fa_label: str(fa_label).startswith('FA_') if pd.notna(fa_label) else False
        else:
            types_list = [t.strip().lower() for t in fa_types.split(',')]
            fa_filter_fn = lambda fa_label: str(fa_label).lower() in types_list if pd.notna(fa_label) else False
    else:
        fa_filter_fn = lambda fa_label: True
    
    rows = []  # {date, odor, hr_odor, category, port_a, port_b, total, ratio}
    
    # Statistics tracking
    stats = {
        'total_no_hr': 0,
        'total_no_hr_fa': 0,
        'total_hr': 0,
        'total_hr_fa': 0
    }
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses in enumerate(ses_dirs, 1):
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"
            
            if not results_dir.exists():
                continue
            
            summary_path = results_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    summary = json.load(f)

                hr_odors = summary.get("params", {}).get("hidden_rule_odors", [])
                if not hr_odors:
                    continue

                df_hr = _load_table_with_trial_data(results_dir, "aborted_sequences_HR")
                df_ab = _load_table_with_trial_data(results_dir, "aborted_sequences_detailed")
                if df_hr.empty or df_ab.empty:
                    continue
                
                # ===== PROCESS ABORTED HR TRIALS (abortion after HR) =====
                if "sequence_start" in df_hr.columns and "sequence_start" in df_ab.columns:
                    # Get HR trials with FA
                    hr_with_fa = df_hr[df_hr["sequence_start"].isin(df_ab["sequence_start"])].copy()
                    
                    if not hr_with_fa.empty:
                        # Merge with FA details
                        merged_hr = hr_with_fa.merge(
                            df_ab[["sequence_start", "fa_label", "last_odor_name", "fa_port", "last_odor_position"]],
                            on="sequence_start",
                            how="left"
                        )
                        
                        # Add HR position info
                        if "hidden_rule_positions" in df_hr.columns:
                            merged_hr = merged_hr.merge(
                                df_hr[["sequence_start", "hidden_rule_positions"]],
                                on="sequence_start",
                                how="left",
                                suffixes=('', '_hr')
                            )
                        
                        # Filter for actual FAs and apply FA type filter
                        merged_hr = merged_hr[
                            (merged_hr["fa_label"] != "nFA") & 
                            (merged_hr["fa_label"].apply(fa_filter_fn))
                        ].copy()
                        
                        stats['total_hr'] += len(hr_with_fa)
                        stats['total_hr_fa'] += len(merged_hr)
                        
                        if not merged_hr.empty:
                            # Filter to abortions that happen AFTER the HR (not on the HR)
                            def get_hr_position(hr_pos_str):
                                if pd.isna(hr_pos_str):
                                    return None
                                try:
                                    pos_list = json.loads(str(hr_pos_str))
                                    if isinstance(pos_list, list) and len(pos_list) > 0:
                                        return int(pos_list[0])
                                except:
                                    pass
                                return None
                            
                            merged_hr["hr_position"] = merged_hr["hidden_rule_positions"].apply(get_hr_position)
                            
                            # Keep only trials where abortion happens AFTER HR position
                            before_after_filter = len(merged_hr)
                            merged_hr = merged_hr[
                                merged_hr["last_odor_position"] > merged_hr["hr_position"]
                            ].copy()
                            stats['total_hr_fa_after_pos'] = stats.get('total_hr_fa_after_pos', 0) + len(merged_hr)
                            stats['total_hr_fa_lost_to_position'] = stats.get('total_hr_fa_lost_to_position', 0) + (before_after_filter - len(merged_hr))
                            
                            if not merged_hr.empty:
                                # Group by last odor and HR odor
                                for last_odor in merged_hr["last_odor_name"].unique():
                                    odor_data = merged_hr[merged_hr["last_odor_name"] == last_odor]
                                    
                                    for hr_odor in hr_odors:
                                        # Check if this HR odor is in the sequence for this trial
                                        odor_matches = []
                                        
                                        if "odor_sequence" in odor_data.columns:
                                            def has_hr_odor(odor_seq, target_hr):
                                                if pd.isna(odor_seq):
                                                    return False
                                                try:
                                                    seq_list = json.loads(str(odor_seq))
                                                    return target_hr in seq_list if isinstance(seq_list, list) else False
                                                except:
                                                    return target_hr in str(odor_seq)
                                            
                                            odor_matches = odor_data[
                                                odor_data["odor_sequence"].apply(lambda seq: has_hr_odor(seq, hr_odor))
                                            ]
                                        else:
                                            odor_matches = odor_data
                                        
                                        if not odor_matches.empty:
                                            port_a = (odor_matches["fa_port"] == 1).sum()
                                            port_b = (odor_matches["fa_port"] == 2).sum()
                                            total = port_a + port_b
                                            ratio = (port_a - port_b) / total if total > 0 else np.nan
                                            
                                            rows.append({
                                                "date": int(date_str),
                                                "odor": last_odor,
                                                "category": hr_odor,
                                                "port_a": port_a,
                                                "port_b": port_b,
                                                "total": total,
                                                "ratio": ratio
                                            })
                
                # ===== PROCESS ABORTED NON-HR TRIALS =====
                # Get trials that are NOT in HR file (no HR present)
                if "sequence_start" in df_ab.columns:
                    ab_no_hr = df_ab[~df_ab["sequence_start"].isin(df_hr["sequence_start"].values)].copy()
                    
                    stats['total_no_hr'] += len(ab_no_hr)
                    
                    # Filter for actual FAs and apply FA type filter
                    ab_no_hr = ab_no_hr[
                        (ab_no_hr["fa_label"] != "nFA") & 
                        (ab_no_hr["fa_label"].apply(fa_filter_fn))
                    ].copy()
                    
                    stats['total_no_hr_fa'] += len(ab_no_hr)
                    
                    if not ab_no_hr.empty:
                        # Track how many go into breakdown
                        before_breakdown = len(ab_no_hr)
                        # Group by last odor
                        for last_odor in ab_no_hr["last_odor_name"].unique():
                            odor_data = ab_no_hr[ab_no_hr["last_odor_name"] == last_odor]
                            
                            port_a = (odor_data["fa_port"] == 1).sum()
                            port_b = (odor_data["fa_port"] == 2).sum()
                            total = port_a + port_b
                            ratio = (port_a - port_b) / total if total > 0 else np.nan
                            
                            rows.append({
                                "date": int(date_str),
                                "odor": last_odor,
                                "category": "No HR",
                                "port_a": port_a,
                                "port_b": port_b,
                                "total": total,
                                "ratio": ratio
                            })
                        stats['total_no_hr_in_breakdown'] = stats.get('total_no_hr_in_breakdown', 0) + sum(
                            row['total'] for row in rows if row.get('category') == 'No HR' and row.get('date') == int(date_str)
                        )
            
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
    
    if not rows:
        print("No data found for FA ratio by abort odor")
        return None, None
    
    df = pd.DataFrame(rows)
    
    # Get unique odors and filter out rewarded odors (OdorA, OdorB)
    all_unique_odors = sorted(df["odor"].unique())
    rewarded_odors = ['OdorA', 'OdorB']
    unique_odors = [odor for odor in all_unique_odors if odor not in rewarded_odors]
    
    # Still print stats for all odors including rewarded ones
    n_odors = len(unique_odors)
    
    # Create subplots: one per odor
    fig, axes = plt.subplots(1, n_odors, figsize=(figsize[0] * 0.85, figsize[1] * 0.9) if n_odors > 2 else figsize)
    if n_odors == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes)
    
    # Define category order
    category_order = []
    if "No HR" in df["category"].unique():
        category_order.append("No HR")
    category_order.extend(sorted([c for c in df["category"].unique() if c != "No HR"]))
    
    # Create session gradient colormap: dark blue for recent, light blue for older
    unique_dates_sorted = sorted(df["date"].unique())
    n_sessions = len(unique_dates_sorted)
    
    # Create color map: most recent = dark blue, oldest = light blue
    if n_sessions == 1:
        colors_for_dates = {unique_dates_sorted[0]: '#00008B'}  # Dark blue
    else:
        # Linear interpolation from light to dark blue
        blue_light = np.array([0.68, 0.85, 1.0])      # Light blue
        blue_dark = np.array([0.0, 0.0, 0.55])        # Dark blue
        colors_for_dates = {}
        for idx, date in enumerate(unique_dates_sorted):
            t = idx / (n_sessions - 1)  # 0 for oldest, 1 for newest
            color = blue_light * (1 - t) + blue_dark * t
            colors_for_dates[date] = color
    
    # Debug: Show how many sessions we have data from
    print(f"\nDEBUG: Data aggregated from {len(unique_dates_sorted)} sessions on dates: {sorted(unique_dates_sorted)}")
    print(f"DEBUG: Color mapping: {unique_dates_sorted} → Most recent (dark) to oldest (light)")
    print(f"DEBUG: Total rows in breakdown dataframe: {len(df)}")
    
    
    # Plot for each odor
    for ax_idx, odor in enumerate(unique_odors):
        ax = axes[ax_idx] if n_odors > 1 else axes[0]
        
        df_odor = df[df["odor"] == odor].copy()
        
        # For this specific odor, only include categories that have data
        categories_with_data = sorted([c for c in df_odor["category"].unique()])
        if not categories_with_data:
            continue
        
        x_positions = {cat: i for i, cat in enumerate(categories_with_data)}
        
        # Scatter plot with session gradient coloring
        for category in categories_with_data:
            df_cat = df_odor[df_odor["category"] == category]
            
            if not df_cat.empty:
                # Plot each date separately with its own color
                for date in unique_dates_sorted:
                    df_date = df_cat[df_cat["date"] == date]
                    if df_date.empty:
                        continue
                    
                    ratios = df_date["ratio"].dropna()
                    if not ratios.empty:
                        x_pos = x_positions[category]
                        # Add small jitter to spread out points
                        x_jitter = np.random.normal(x_pos, 0.06, size=len(ratios))
                        color = colors_for_dates[date]
                        ax.scatter(x_jitter, ratios, alpha=0.7, s=80, color=color, 
                                  edgecolors='none', label=f'{date}' if ax_idx == 0 else '')
        
        # Add black line for aggregate mean for each category that actually has data in this odor
        # Line width scales with number of categories (smaller when fewer categories)
        line_half_width = 0.15 if len(categories_with_data) > 1 else 0.08
        for category in categories_with_data:
            df_cat = df_odor[df_odor["category"] == category]
            all_ratios = df_cat["ratio"].dropna()
            if len(all_ratios) > 0:
                mean_ratio = all_ratios.mean()
                x_pos = x_positions[category]
                # Only draw line if we have data at this position
                ax.plot([x_pos - line_half_width, x_pos + line_half_width], [mean_ratio, mean_ratio], 
                       color='black', linewidth=3, alpha=0.8, zorder=10)
        
        ax.set_xticks(range(len(categories_with_data)))
        ax.set_xticklabels(categories_with_data, fontsize=10, fontweight='bold', rotation=0)
        ax.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
        ax.set_ylim([-1.1, 1.1])
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f'{odor}', fontsize=12, fontweight='bold')
        
        # Set x-axis limits with padding
        n_cats = len(categories_with_data)
        ax.set_xlim(-0.5, n_cats - 0.5)
        ax.margins(y=0)  # Only apply margins to y-axis, not x-axis
    
    # Create a legend for the sessions (on the first subplot)
    if n_odors > 0:
        # Create custom legend entries
        legend_elements = []
        for date in reversed(unique_dates_sorted):  # Reverse so newest is first
            label = f'{date}'
            if date == unique_dates_sorted[-1]:
                label += ' (recent)'
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=colors_for_dates[date], 
                                         markersize=8, label=label, alpha=0.7))
        
        fig.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                  title='Sessions', title_fontsize=10, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space for legend
    
    # Print statistics
    print("\n" + "="*100)
    print("FA RATIO BY ABORTION ODOR - STATISTICS")
    print("="*100)
    print(f"\nAborted Sequences WITHOUT Hidden Rule:")
    print(f"  Total aborted: {stats['total_no_hr']}")
    print(f"  Matching FA filter: {stats['total_no_hr_fa']}")
    print(f"  In breakdown by odor: {stats.get('total_no_hr_in_breakdown', 'unknown')}")
    
    # Calculate actual HR breakdown count
    hr_breakdown_count = sum(row['total'] for row in rows if row.get('category') != 'No HR')
    
    print(f"\nAborted Sequences WITH Hidden Rule (abortion AFTER HR):")
    print(f"  Total aborted: {stats['total_hr']}")
    print(f"  Matching FA filter: {stats['total_hr_fa']}")
    print(f"  After position filter (after HR): {stats.get('total_hr_fa_after_pos', 'unknown')}")
    print(f"  In breakdown table: {hr_breakdown_count}")
    print(f"\nDISCREPANCY ANALYSIS:")
    print(f"  Non-HR: FA filter count ({stats['total_no_hr_fa']}) vs breakdown count ({stats.get('total_no_hr_in_breakdown', 'unknown')})")
    print(f"  HR: FA filter count ({stats['total_hr_fa']}) vs breakdown count ({hr_breakdown_count})")
    print(f"  Missing HR trials in breakdown: {stats['total_hr_fa'] - hr_breakdown_count}")
    
    print(f"\n" + "-"*100)
    print("BREAKDOWN BY ODOR AND CATEGORY (including rewarded odors OdorA, OdorB in stats):")
    print("-"*100)
    
    # Group by odor and show per-date breakdown for ALL odors
    for odor in all_unique_odors:
        is_rewarded = odor in rewarded_odors
        odor_label = f"{odor}" + (" [REWARDED - not plotted]" if is_rewarded else "")
        print(f"\n{odor_label}:")
        df_odor = df[df["odor"] == odor]
        
        for category in category_order:
            df_cat = df_odor[df_odor["category"] == category]
            
            if not df_cat.empty:
                # Show aggregate across all dates
                port_a_total = df_cat["port_a"].sum()
                port_b_total = df_cat["port_b"].sum()
                total_trials = df_cat["total"].sum()
                ratio_agg = (port_a_total - port_b_total) / total_trials if total_trials > 0 else np.nan
                
                ratio_str = f"{ratio_agg:+.3f}" if not pd.isna(ratio_agg) else "N/A"
                print(f"  {category:<12} - Ratio: {ratio_str}  Port A: {int(port_a_total)}, Port B: {int(port_b_total)}, Total: {int(total_trials)}")
                
                # Show per-date breakdown
                for idx, row in df_cat.iterrows():
                    date_val = int(row['date'])
                    ratio_str_date = f"{row['ratio']:+.3f}" if not pd.isna(row['ratio']) else "N/A"
                    print(f"      → {date_val}: Port A: {int(row['port_a'])}, Port B: {int(row['port_b'])}, Total: {int(row['total'])}")
            else:
                print(f"  {category:<12} - No data")
    
    print("="*100)
        
    # Show summary totals
    print("\nSUMMARY BY CATEGORY (across all odors and dates):")
    print("-"*100)
    
    total_no_hr_all = df[df["category"] == "No HR"]["total"].sum()
    total_hr_all = df[df["category"] != "No HR"]["total"].sum()
    
    print(f"No HR trials total: {int(total_no_hr_all)}")
    print(f"HR trials total: {int(total_hr_all)}")

    return fig, axes
