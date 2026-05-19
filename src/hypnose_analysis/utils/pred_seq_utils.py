"""Utilities for predicted sequence analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union
import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hypnose_analysis.helpers import _filter_session_dirs, _iter_subject_dirs
from hypnose_analysis.paths import get_derivatives_root
from hypnose_analysis.utils.save_utils import save_figure
from matplotlib.patches import Patch


def _parse_json_value(val):
	if isinstance(val, (dict, list)):
		return val
	if not isinstance(val, str):
		return None
	try:
		return json.loads(val)
	except Exception:
		try:
			return ast.literal_eval(val)
		except Exception:
			return None


def _normalize_date(value):
	if value is None:
		return None
	digits = "".join(ch for ch in str(value) if ch.isdigit())
	return int(digits) if digits.isdigit() else str(value)


def _collect_sessions(subjids, dates):
	derivatives_dir = get_derivatives_root()
	for subjid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
		if subjid is None:
			continue
		date_vals = []
		results_dirs = []
		for ses_dir in _filter_session_dirs(subj_dir, dates):
			date_part = ses_dir.name.split("_date-")[-1]
			date_val = _normalize_date(date_part)
			if date_val is None:
				continue
			date_vals.append(date_val)
			results_dirs.append(ses_dir / "saved_analysis_results")
		if results_dirs:
			yield subjid, date_vals, results_dirs


def _load_trial_data(results_dir: Path) -> pd.DataFrame:
	parquet_path = results_dir / "trial_data.parquet"
	if not parquet_path.exists():
		raise FileNotFoundError(f"Missing trial_data.parquet at {parquet_path}")
	return pd.read_parquet(parquet_path)


def _sequence_label(seq):
	if not isinstance(seq, (list, tuple)):
		return None
	parts = []
	for item in seq:
		text = str(item)
		if text.startswith("Odor"):
			parts.append(text.replace("Odor", "", 1))
		else:
			parts.append(text)
	return "-".join(parts) if parts else None


def _sequence_len_ok(seq, *, min_len: int = 3) -> bool:
	return isinstance(seq, (list, tuple)) and len(seq) >= min_len


def _normalize_odor_name(value):
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	if text.startswith("Odor"):
		text = text.replace("Odor", "", 1)
	text = text.strip()
	return f"Odor{text}" if text else None


def _last_position_entry(pos_dict):
	if not isinstance(pos_dict, dict):
		return None
	candidates = []
	for key, entry in pos_dict.items():
		if not isinstance(entry, dict):
			continue
		position = entry.get("position")
		if position is None:
			try:
				position = int(key)
			except Exception:
				position = None
		candidates.append((position, entry))
	if not candidates:
		return None
	candidates.sort(key=lambda x: (x[0] is None, x[0]))
	return candidates[-1][1]


def _extract_position_entry(pos_dict, position: int):
	if not isinstance(pos_dict, dict):
		return None
	for key, entry in pos_dict.items():
		if not isinstance(entry, dict):
			continue
		entry_pos = entry.get("position")
		if entry_pos is None:
			try:
				entry_pos = int(key)
			except Exception:
				entry_pos = None
		if entry_pos == position:
			return entry
	return None


def _plot_box_with_points(ax, groups, y_label, x_label):
	labels = list(groups.keys())
	data = [groups[label] for label in labels]
	mean_half_width = 0.18
	cap_half_width = 0.10
	mean_lw = 2.2
	sem_lw = 1.4
	jitter_half_width = 0.15
	rng = np.random.default_rng(0)
	for i, values in enumerate(data, start=1):
		if not values:
			continue
		x = i + rng.uniform(-jitter_half_width, jitter_half_width, size=len(values))
		ax.scatter(x, values, s=18, color="blue", alpha=0.4, zorder=2)
		mean_val = float(np.mean(values))
		sem_val = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
		ax.hlines(
			mean_val,
			i - mean_half_width,
			i + mean_half_width,
			colors="black",
			linewidth=mean_lw,
			zorder=3,
		)
		ax.vlines(
			i,
			mean_val - sem_val,
			mean_val + sem_val,
			colors="black",
			linewidth=sem_lw,
			zorder=3,
		)
		if sem_val > 0:
			ax.hlines(
				mean_val - sem_val,
				i - cap_half_width,
				i + cap_half_width,
				colors="black",
				linewidth=sem_lw,
				zorder=3,
			)
			ax.hlines(
				mean_val + sem_val,
				i - cap_half_width,
				i + cap_half_width,
				colors="black",
				linewidth=sem_lw,
				zorder=3,
			)
	ax.set_ylabel(y_label)
	ax.set_xlabel(x_label)
	ax.set_xticks(range(1, len(labels) + 1))
	ax.set_xticklabels(labels, rotation=45, ha="right")


def _order_sequence_labels(groups):
	preferred = ["F-G-A", "E-D-A", "E-D-B", "C-G-B"]
	labels = []
	for name in preferred:
		if name in groups:
			labels.append(name)
	for name in groups:
		if name not in labels:
			labels.append(name)
	return labels


def _order_odor_labels(groups):
	def _key(name):
		text = str(name)
		if text.startswith("Odor"):
			text = text.replace("Odor", "", 1)
		return text
	return sorted(groups.keys(), key=_key)


SEQUENCE_COLORS = {
	"F-G-A": "#2ca02c",
	"C-G-B": "#d62728",
	"E-D-A": "#bfbfbf",
	"E-D-B": "#7f7f7f",
}

ODOR_COLORS = {
	"OdorA": "#2ca02c",
	"OdorB": "#d62728",
	"OdorF": "#7fbf7f",
	"OdorC": "#f08080",
	"OdorG": "#1f77b4",
	"OdorE": "#a0a0a0",
	"OdorD": "#606060",
}

SEQUENCE_ORDER = ["F-G-A", "E-D-A", "E-D-B", "C-G-B"]
ODOR_ORDER = ["OdorA", "OdorB", "OdorC", "OdorD", "OdorE", "OdorF", "OdorG"]


def _resolve_color(label, color_map, default="#444444"):
	return color_map.get(label, default)


def _ordered_groups(group_keys, preferred):
	result = []
	for name in preferred:
		if name in group_keys:
			result.append(name)
	for name in group_keys:
		if name not in result:
			result.append(name)
	return result


def _load_sorted_session(results_dir):
	df = _load_trial_data(results_dir)
	if df.empty:
		return df
	df = df.copy()
	if "sequence_start" in df.columns:
		df["_order_key"] = pd.to_datetime(df["sequence_start"], errors="coerce")
		df = df.sort_values("_order_key", na_position="last")
	elif "timestamp" in df.columns:
		df["_order_key"] = pd.to_datetime(df["timestamp"], errors="coerce")
		df = df.sort_values("_order_key", na_position="last")
	df = df.reset_index(drop=True)
	df["_trial_idx"] = np.arange(1, len(df) + 1)
	return df


def _plot_summary_daily(session_data, *, color_map, group_order, ylabel, title, ylim_bottom=None):
	"""
	Plot per-session mean for each group, connected by lines.

	session_data : list of dicts, one per session, each with keys:
		- "n_trials": int (full session length)
		- "groups": {group_label: [(trial_idx, value), ...]}
	X axis is Session (1, 2, ...).
	"""
	n_sessions = len(session_data)
	if n_sessions == 0:
		return None

	all_groups = set()
	for s in session_data:
		all_groups.update(s["groups"].keys())
	if not all_groups:
		return None
	groups = _ordered_groups(all_groups, group_order)

	fig, ax = plt.subplots(figsize=(10, 5))

	for group in groups:
		xs, ys = [], []
		for i, s in enumerate(session_data):
			vals = [v for _, v in s["groups"].get(group, [])]
			if vals:
				xs.append(i + 1)
				ys.append(float(np.mean(vals)))
		if not xs:
			continue
		color = _resolve_color(group, color_map)
		ax.plot(xs, ys, color=color, marker="o", linewidth=2, markersize=6, label=group)

	ax.set_xlabel("Session")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.set_xticks(range(1, n_sessions + 1))
	ax.set_xticklabels([str(i) for i in range(1, n_sessions + 1)])
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	if ylim_bottom is not None:
		ax.set_ylim(bottom=ylim_bottom)
	ax.legend(loc="best")
	fig.tight_layout()
	return fig


def _plot_summary_rolling(session_data, *, color_map, group_order, ylabel, title, window_size, step_size, ylim_bottom=None):
	"""
	Plot rolling mean within each session per group; X axis is continuous global trial id.
	Lines do not bridge session boundaries (gaps at day lines).
	"""
	window_n = max(1, int(window_size))
	step_n = max(1, int(step_size))

	if not session_data:
		return None

	all_groups = set()
	for s in session_data:
		all_groups.update(s["groups"].keys())
	if not all_groups:
		return None
	groups = _ordered_groups(all_groups, group_order)

	fig, ax = plt.subplots(figsize=(12, 6))

	# Width reserved on the X axis for a session that has no plottable rolling-window
	# points (e.g. fewer than window_size trials of any group). Keeps multi-day plots
	# from different categories visually aligned in day count.
	empty_session_span = 20

	global_offset = 0
	boundary_lines = []
	legend_done = set()

	for s in session_data:
		# Compute rolling-window points and per-session local bounds in one pass.
		session_points = {}
		min_x = None
		max_x = None
		for group in groups:
			entries = s["groups"].get(group, [])
			if not entries:
				continue
			entries_sorted = sorted(entries, key=lambda x: x[0])
			idxs = np.array([e[0] for e in entries_sorted])
			vals = np.array([e[1] for e in entries_sorted], dtype=float)
			n = len(vals)
			pts = []
			for end in range(window_n, n + 1, step_n):
				start = end - window_n
				rate = float(np.nanmean(vals[start:end]))
				local_x = int(idxs[end - 1])
				pts.append((local_x, rate))
				min_x = local_x if min_x is None else min(min_x, local_x)
				max_x = local_x if max_x is None else max(max_x, local_x)
			if pts:
				session_points[group] = pts

		if global_offset > 0:
			boundary_lines.append(global_offset - 0.5)

		if min_x is None:
			global_offset += empty_session_span
			continue

		shift = global_offset - min_x
		for group, pts in session_points.items():
			xs = [lx + shift for lx, _ in pts]
			ys = [y for _, y in pts]
			color = _resolve_color(group, color_map)
			label = group if group not in legend_done else None
			legend_done.add(group)
			ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, label=label)

		global_offset += max_x - min_x + 1

	for x in boundary_lines:
		ax.axvline(
			x=x,
			color="#1f77b4",
			linestyle=":",
			linewidth=1.2,
			alpha=0.7,
			zorder=1,
		)

	ax.set_xlabel("Trials (adjusted)")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	if global_offset > 0:
		ax.set_xlim(left=0, right=global_offset - 1)
	else:
		ax.set_xlim(left=0)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	if ylim_bottom is not None:
		ax.set_ylim(bottom=ylim_bottom)
	if legend_done:
		ax.legend(loc="best")
	fig.tight_layout()
	return fig


def _plot_summary(
	session_data,
	*,
	color_map,
	group_order,
	ylabel,
	title,
	moving_avg,
	window_size,
	step_size,
	ylim_bottom=None,
):
	"""Dispatcher: daily means (moving_avg=False) or rolling-within-session (moving_avg=True)."""
	if moving_avg:
		return _plot_summary_rolling(
			session_data,
			color_map=color_map,
			group_order=group_order,
			ylabel=ylabel,
			title=title,
			window_size=window_size,
			step_size=step_size,
			ylim_bottom=ylim_bottom,
		)
	return _plot_summary_daily(
		session_data,
		color_map=color_map,
		group_order=group_order,
		ylabel=ylabel,
		title=title,
		ylim_bottom=ylim_bottom,
	)


def _is_multi_session(date_vals):
	"""Summary plots are only meaningful when more than one session is loaded."""
	return date_vals is not None and len(date_vals) > 1


def _apply_shared_ylim(figs_to_share, *, bottom_zero=False):
	"""Set a common ylim across multiple figures so plots from a single call line up.

	The shared bottom is min of each figure's current bottom (or 0 if ``bottom_zero``),
	and the shared top is max of each figure's current top.
	"""
	if not figs_to_share:
		return
	axes = [fig.axes[0] for fig in figs_to_share if fig.axes]
	if not axes:
		return
	bottoms = [a.get_ylim()[0] for a in axes]
	tops = [a.get_ylim()[1] for a in axes]
	common_bottom = 0.0 if bottom_zero else min(bottoms)
	common_top = max(tops)
	for a in axes:
		a.set_ylim(common_bottom, common_top)


def _summary_save_suffix(moving_avg, window_size, step_size):
	if moving_avg:
		return f"rolling_w{int(window_size)}_s{int(step_size)}"
	return "daily"


def last_odor_poke_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of last-odor poke time by odor sequence, separated by response category.

	When multiple sessions are loaded, additional summary plots are produced for
	rewarded and unrewarded categories (one each). With ``moving_avg=False`` a per-
	session mean is plotted (X axis: Session); with ``moving_avg=True`` a rolling
	mean of size ``window_size`` (step ``step_size``) is plotted within each session
	with X axis = continuous global trial id (no line continuation across days).
	"""
	figs = []
	categories = ["rewarded", "unrewarded", "timeout_delayed"]
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		per_cat_pooled = {cat: {} for cat in categories}
		per_cat_sessions = {cat: [] for cat in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in categories:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			for cat in categories:
				session_groups = {}
				cat_df = completed[completed.get("response_time_category") == cat]
				for _, row in cat_df.iterrows():
					last_odor = row.get("last_odor")
					if last_odor not in {"OdorA", "OdorB"}:
						continue
					seq = _parse_json_value(row.get("odor_sequence"))
					if not _sequence_len_ok(seq):
						continue
					seq_label = _sequence_label(seq)
					if not seq_label:
						continue
					pos_dict = _parse_json_value(row.get("position_poke_times"))
					last_entry = _last_position_entry(pos_dict)
					if not last_entry:
						continue
					poke_ms = last_entry.get("poke_time_ms")
					if poke_ms is None:
						continue
					poke_val = float(poke_ms)
					trial_idx = int(row["_trial_idx"])
					session_groups.setdefault(seq_label, []).append((trial_idx, poke_val))
					per_cat_pooled[cat].setdefault(seq_label, []).append(poke_val)
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": session_groups})

		for cat in categories:
			pooled = per_cat_pooled[cat]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
				_plot_box_with_points(ax, ordered, "Last Odor Poke Time (ms)", "Odor Sequence")
				ax.set_title(f"Subjid {subjid} {cat} last-odor poke time")
				ax.set_ylim(bottom=0)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"last_odor_poke_time_{cat}", subjids=[subjid], dates=date_vals)

		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Last Odor Poke Time (ms)",
					title=f"Subjid {subjid} {cat} last-odor poke time ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
					ylim_bottom=0,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						save_figure(
							summary_fig,
							f"last_odor_poke_time_{cat}_summary_{suffix}",
							subjids=[subjid],
							dates=date_vals,
						)

	return figs


def trial_poke_duration(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of trial poke duration by odor sequence, separated by response category.

	Cross-date summary plots (one for rewarded, one for unrewarded) are produced
	in addition to the per-session pooled boxplots. See ``last_odor_poke_time`` for
	the meaning of ``moving_avg``, ``window_size``, ``step_size``.
	"""
	figs = []
	categories = ["rewarded", "unrewarded", "timeout_delayed"]
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		per_cat_pooled = {cat: {} for cat in categories}
		per_cat_sessions = {cat: [] for cat in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in categories:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			for cat in categories:
				session_groups = {}
				cat_df = completed[completed.get("response_time_category") == cat]
				for _, row in cat_df.iterrows():
					seq = _parse_json_value(row.get("odor_sequence"))
					if not _sequence_len_ok(seq):
						continue
					seq_label = _sequence_label(seq)
					if not seq_label:
						continue
					pos_dict = _parse_json_value(row.get("position_poke_times"))
					first_entry = _extract_position_entry(pos_dict, 1)
					last_entry = _last_position_entry(pos_dict)
					if not first_entry or not last_entry:
						continue
					start_ts = first_entry.get("poke_odor_start")
					end_ts = last_entry.get("poke_odor_end")
					if not start_ts or not end_ts:
						continue
					start_dt = pd.to_datetime(start_ts, errors="coerce")
					end_dt = pd.to_datetime(end_ts, errors="coerce")
					if pd.isna(start_dt) or pd.isna(end_dt):
						continue
					duration_ms = (end_dt - start_dt).total_seconds() * 1000.0
					dur_val = float(duration_ms)
					trial_idx = int(row["_trial_idx"])
					session_groups.setdefault(seq_label, []).append((trial_idx, dur_val))
					per_cat_pooled[cat].setdefault(seq_label, []).append(dur_val)
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": session_groups})

		for cat in categories:
			pooled = per_cat_pooled[cat]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
				_plot_box_with_points(ax, ordered, "Trial Poke Duration (ms)", "Odor Sequence")
				ax.set_title(f"Subjid {subjid} {cat} trial poke duration")
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"trial_poke_duration_{cat}", subjids=[subjid], dates=date_vals)

		summary_figs = []
		summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Trial Poke Duration (ms)",
					title=f"Subjid {subjid} {cat} trial poke duration ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					summary_figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						summary_save_specs.append(
							(summary_fig, f"trial_poke_duration_{cat}_summary_{suffix}")
						)

		_apply_shared_ylim(summary_figs)
		for fig, name in summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs


def first_odor_poke_duration(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplot of first-odor poke duration grouped by odor name (completed trials only).

	A cross-date summary plot is added when multiple sessions are loaded
	(daily mean or rolling within session — see ``last_odor_poke_time``).
	"""
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		pooled = {}
		session_records = []
		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				session_records.append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			session_groups = {}
			for _, row in completed.iterrows():
				pos_dict = _parse_json_value(row.get("position_poke_times"))
				first_entry = _extract_position_entry(pos_dict, 1)
				if not first_entry:
					continue
				odor_name = first_entry.get("odor_name")
				poke_ms = first_entry.get("poke_time_ms")
				if odor_name is None or poke_ms is None:
					continue
				key = str(odor_name)
				poke_val = float(poke_ms)
				trial_idx = int(row["_trial_idx"])
				session_groups.setdefault(key, []).append((trial_idx, poke_val))
				pooled.setdefault(key, []).append(poke_val)
			session_records.append({"n_trials": n_trials, "groups": session_groups})

		if pooled:
			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: pooled[k] for k in _order_odor_labels(pooled)}
			_plot_box_with_points(ax, ordered, "Poke Duration", "Odor")
			ax.set_title(f"Subjid {subjid} first-odor poke duration")
			ax.set_ylim(bottom=0)
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, "first_odor_poke_duration", subjids=[subjid], dates=date_vals)

		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			summary_fig = _plot_summary(
				session_records,
				color_map=ODOR_COLORS,
				group_order=ODOR_ORDER,
				ylabel="Poke Duration (ms)",
				title=f"Subjid {subjid} first-odor poke duration ({mode_label})",
				moving_avg=moving_avg,
				window_size=window_size,
				step_size=step_size,
				ylim_bottom=0,
			)
			if summary_fig is not None:
				figs.append(summary_fig)
				if save:
					suffix = _summary_save_suffix(moving_avg, window_size, step_size)
					save_figure(
						summary_fig,
						f"first_odor_poke_duration_summary_{suffix}",
						subjids=[subjid],
						dates=date_vals,
					)

	return figs


def poke_time_all_pos(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of poke duration pooled across positions, grouped by odor name.

	Two plots are produced per subject: one for completed trials (is_aborted=False)
	and one for aborted trials (is_aborted=True), so poke durations can be compared
	between the two. Cross-date summary plots are added for each (daily mean per
	session, or rolling within session if ``moving_avg=True``).
	"""
	figs = []
	categories = [("completed", False), ("aborted", True)]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		per_cat_pooled = {name: {} for name, _ in categories}
		per_cat_sessions = {name: [] for name, _ in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for name, _ in categories:
					per_cat_sessions[name].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			for name, aborted_flag in categories:
				cat_df = df[df.get("is_aborted") == aborted_flag]
				session_groups = {}
				for _, row in cat_df.iterrows():
					pos_dict = _parse_json_value(row.get("position_poke_times"))
					if not isinstance(pos_dict, dict):
						continue
					trial_idx = int(row["_trial_idx"])
					for entry in pos_dict.values():
						if not isinstance(entry, dict):
							continue
						odor_name = entry.get("odor_name")
						poke_ms = entry.get("poke_time_ms")
						if odor_name is None or poke_ms is None:
							continue
						key = str(odor_name)
						poke_val = float(poke_ms)
						session_groups.setdefault(key, []).append((trial_idx, poke_val))
						per_cat_pooled[name].setdefault(key, []).append(poke_val)
				per_cat_sessions[name].append({"n_trials": n_trials, "groups": session_groups})

		for name, _ in categories:
			pooled = per_cat_pooled[name]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_odor_labels(pooled)}
				_plot_box_with_points(ax, ordered, "Poke Duration (ms)", "Odor")
				ax.set_title(f"Subjid {subjid} poke duration (all positions, {name})")
				ax.set_ylim(bottom=0)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"poke_time_all_pos_{name}", subjids=[subjid], dates=date_vals)

		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for name, _ in categories:
				summary_fig = _plot_summary(
					per_cat_sessions[name],
					color_map=ODOR_COLORS,
					group_order=ODOR_ORDER,
					ylabel="Poke Duration (ms)",
					title=f"Subjid {subjid} poke duration (all positions, {name}) ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
					ylim_bottom=0,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						save_figure(
							summary_fig,
							f"poke_time_all_pos_{name}_summary_{suffix}",
							subjids=[subjid],
							dates=date_vals,
						)

	return figs


def response_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of response time by odor sequence (completed, non-timeout trials).

	Cross-date summary plots are produced for rewarded and unrewarded trials
	(daily mean per session, or rolling within session if ``moving_avg=True``).
	"""
	figs = []
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		pooled = {}
		per_cat_sessions = {cat: [] for cat in summary_cats}
		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in summary_cats:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			completed = completed[completed.get("response_time_category") != "timeout_delayed"]

			per_cat_session_groups = {cat: {} for cat in summary_cats}
			for _, row in completed.iterrows():
				seq = _parse_json_value(row.get("odor_sequence"))
				if not _sequence_len_ok(seq):
					continue
				seq_label = _sequence_label(seq)
				if not seq_label:
					continue
				pos_dict = _parse_json_value(row.get("position_poke_times"))
				last_entry = _last_position_entry(pos_dict)
				if not last_entry:
					continue
				poke_end = last_entry.get("poke_odor_end")
				supply_time = row.get("first_supply_time")
				if poke_end is None or supply_time is None:
					continue
				poke_end_dt = pd.to_datetime(poke_end, errors="coerce")
				supply_dt = pd.to_datetime(supply_time, errors="coerce")
				if pd.isna(poke_end_dt) or pd.isna(supply_dt):
					continue
				rt_ms = float((supply_dt - poke_end_dt).total_seconds() * 1000.0)
				trial_idx = int(row["_trial_idx"])
				pooled.setdefault(seq_label, []).append(rt_ms)
				cat = row.get("response_time_category")
				if cat in per_cat_session_groups:
					per_cat_session_groups[cat].setdefault(seq_label, []).append((trial_idx, rt_ms))

			for cat in summary_cats:
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": per_cat_session_groups[cat]})

		if pooled:
			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
			_plot_box_with_points(ax, ordered, "Response Time (ms)", "Odor Sequence")
			ax.set_title(f"Subjid {subjid} response time")
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, "response_time", subjids=[subjid], dates=date_vals)

		summary_figs = []
		summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Response Time (ms)",
					title=f"Subjid {subjid} {cat} response time ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					summary_figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						summary_save_specs.append(
							(summary_fig, f"response_time_{cat}_summary_{suffix}")
						)

		_apply_shared_ylim(summary_figs)
		for fig, name in summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs


def fa_analysis(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	FA trial analysis for aborted trials by last odor and FA port.

	Three cross-date summary plots are added (daily mean per session, or rolling
	within session if ``moving_avg=True``):
	  - FA poke time by odor.
	  - FA response time, FA→A, by odor.
	  - FA response time, FA→B, by odor.
	"""
	figs = []
	odor_whitelist = {"OdorC", "OdorD", "OdorE", "OdorF", "OdorG"}
	fa_labels = {"FA_time_in", "FA_time_out"}
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		poke_groups = {}
		resp_groups = {}
		poke_session_records = []
		resp_session_records = {"A": [], "B": []}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				poke_session_records.append({"n_trials": 0, "groups": {}})
				resp_session_records["A"].append({"n_trials": 0, "groups": {}})
				resp_session_records["B"].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			aborted = df[df.get("is_aborted") == True]
			fa_df = aborted[aborted.get("fa_label").isin(fa_labels)]

			session_poke = {}
			session_resp = {"A": {}, "B": {}}
			for _, row in fa_df.iterrows():
				last_odor = _normalize_odor_name(row.get("last_odor"))
				if last_odor not in odor_whitelist:
					continue

				pos_dict = _parse_json_value(row.get("position_poke_times"))
				if not isinstance(pos_dict, dict):
					continue
				odor_entry = None
				for entry in pos_dict.values():
					if not isinstance(entry, dict):
						continue
					entry_odor = _normalize_odor_name(entry.get("odor_name"))
					if entry_odor == last_odor:
						odor_entry = entry
						break
				if not odor_entry:
					continue

				poke_ms = odor_entry.get("poke_time_ms")
				if poke_ms is not None:
					poke_val = float(poke_ms)
					trial_idx = int(row["_trial_idx"])
					poke_groups.setdefault(last_odor, []).append(poke_val)
					session_poke.setdefault(last_odor, []).append((trial_idx, poke_val))

				fa_time = row.get("fa_time")
				poke_end = odor_entry.get("poke_odor_end")
				if fa_time is None or poke_end is None:
					continue
				fa_dt = pd.to_datetime(fa_time, errors="coerce")
				poke_end_dt = pd.to_datetime(poke_end, errors="coerce")
				if pd.isna(fa_dt) or pd.isna(poke_end_dt):
					continue
				rt_ms = (fa_dt - poke_end_dt).total_seconds() * 1000.0

				port = row.get("fa_port")
				try:
					port_val = int(port)
				except Exception:
					port_val = None
				if port_val == 1:
					port_label = "A"
				elif port_val == 2:
					port_label = "B"
				else:
					continue

				rt_val = float(rt_ms)
				resp_groups.setdefault(last_odor, {"A": [], "B": []})[port_label].append(rt_val)
				trial_idx = int(row["_trial_idx"])
				session_resp[port_label].setdefault(last_odor, []).append((trial_idx, rt_val))

			poke_session_records.append({"n_trials": n_trials, "groups": session_poke})
			resp_session_records["A"].append({"n_trials": n_trials, "groups": session_resp["A"]})
			resp_session_records["B"].append({"n_trials": n_trials, "groups": session_resp["B"]})

		if poke_groups:
			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: poke_groups[k] for k in _order_odor_labels(poke_groups)}
			_plot_box_with_points(ax, ordered, "Poke Time (ms)", "Odor")
			ax.set_title(f"Subjid {subjid} FA poke time by odor")
			ax.set_ylim(bottom=0)
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, "fa_poke_time", subjids=[subjid], dates=date_vals)

		if resp_groups:
			ordered_odors = _order_odor_labels(resp_groups)
			labels = []
			has_any = False
			fig, ax = plt.subplots(figsize=(12, 5))
			mean_half_width = 0.14
			cap_half_width = 0.08
			mean_lw = 2.2
			sem_lw = 1.4
			point_offset = 0.18
			jitter_half_width = 0.12
			rng = np.random.default_rng(0)

			for i, odor in enumerate(ordered_odors, start=1):
				odor_groups = resp_groups[odor]
				count_a = len(odor_groups.get("A", []))
				count_b = len(odor_groups.get("B", []))
				labels.append(f"{odor}\nRatio A/B: {count_a}/{count_b}")

				for port_label, color, offset in (
					("A", "red", -point_offset),
					("B", "green", point_offset),
				):
					values = odor_groups.get(port_label, [])
					if not values:
						continue
					has_any = True
					x_pos = i + offset
					xs = x_pos + rng.uniform(-jitter_half_width, jitter_half_width, size=len(values))
					ax.scatter(xs, values, s=18, color=color, alpha=0.4, zorder=2)
					mean_val = float(np.mean(values))
					sem_val = (
						float(np.std(values, ddof=1) / np.sqrt(len(values)))
						if len(values) > 1
						else 0.0
					)
					ax.hlines(
						mean_val,
						x_pos - mean_half_width,
						x_pos + mean_half_width,
						colors="black",
						linewidth=mean_lw,
						zorder=3,
					)
					ax.vlines(
						x_pos,
						mean_val - sem_val,
						mean_val + sem_val,
						colors="black",
						linewidth=sem_lw,
						zorder=3,
					)
					if sem_val > 0:
						ax.hlines(
							mean_val - sem_val,
							x_pos - cap_half_width,
							x_pos + cap_half_width,
							colors="black",
							linewidth=sem_lw,
							zorder=3,
						)
						ax.hlines(
							mean_val + sem_val,
							x_pos - cap_half_width,
							x_pos + cap_half_width,
							colors="black",
							linewidth=sem_lw,
							zorder=3,
						)

			if has_any:
				ax.set_xticks(range(1, len(labels) + 1))
				ax.set_xticklabels(labels, rotation=45, ha="right")
				ax.set_ylabel("FA Response Time (ms)")
				ax.set_xlabel("Odor")
				ax.set_title(f"Subjid {subjid} FA response time by port")
				ax.legend(
					handles=[
						Patch(facecolor="red", edgecolor="black", label="FA to port A"),
						Patch(facecolor="green", edgecolor="black", label="FA to port B"),
					],
					loc="upper right",
				)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, "fa_response_time", subjids=[subjid], dates=date_vals)
			else:
				plt.close(fig)

		resp_summary_figs = []
		resp_summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			summary_fig = _plot_summary(
				poke_session_records,
				color_map=ODOR_COLORS,
				group_order=ODOR_ORDER,
				ylabel="FA Poke Time (ms)",
				title=f"Subjid {subjid} FA poke time by odor ({mode_label})",
				moving_avg=moving_avg,
				window_size=window_size,
				step_size=step_size,
				ylim_bottom=0,
			)
			if summary_fig is not None:
				figs.append(summary_fig)
				if save:
					suffix = _summary_save_suffix(moving_avg, window_size, step_size)
					save_figure(
						summary_fig,
						f"fa_poke_time_summary_{suffix}",
						subjids=[subjid],
						dates=date_vals,
					)

			for port_label in ("A", "B"):
				resp_summary_fig = _plot_summary(
					resp_session_records[port_label],
					color_map=ODOR_COLORS,
					group_order=ODOR_ORDER,
					ylabel="FA Response Time (ms)",
					title=f"Subjid {subjid} FA response time (FA→{port_label}) by odor ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
				)
				if resp_summary_fig is not None:
					figs.append(resp_summary_fig)
					resp_summary_figs.append(resp_summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						resp_summary_save_specs.append(
							(resp_summary_fig, f"fa_response_time_port{port_label}_summary_{suffix}")
						)

		_apply_shared_ylim(resp_summary_figs)
		for fig, name in resp_summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs
