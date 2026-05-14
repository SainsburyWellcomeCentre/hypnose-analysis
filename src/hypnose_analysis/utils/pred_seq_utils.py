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
	mean_half_width = 0.12
	cap_half_width = 0.06
	mean_lw = 0.9
	sem_lw = 0.9
	for i, values in enumerate(data, start=1):
		if not values:
			continue
		x = np.full(len(values), i, dtype=float)
		ax.scatter(x, values, s=18, color="blue", alpha=0.7, zorder=2)
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


def last_odor_poke_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
):
	"""
	Boxplots of last-odor poke time by odor sequence, separated by response category.
	"""
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		frames = []
		for results_dir in results_dirs:
			frames.append(_load_trial_data(results_dir))
		df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
		if df.empty:
			continue

		completed = df[df.get("is_aborted") == False].copy()
		if completed.empty:
			continue

		categories = ["rewarded", "unrewarded", "timeout_delayed"]
		for cat in categories:
			cat_df = completed[completed.get("response_time_category") == cat]
			if cat_df.empty:
				continue

			groups = {}
			for _, row in cat_df.iterrows():
				last_odor = row.get("last_odor")
				if last_odor not in {"OdorA", "OdorB"}:
					continue
				seq = _parse_json_value(row.get("odor_sequence"))
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

				groups.setdefault(seq_label, []).append(float(poke_ms))

			if not groups:
				continue

			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: groups[k] for k in _order_sequence_labels(groups)}
			_plot_box_with_points(ax, ordered, "Last Odor Poke Time (ms)", "Odor Sequence")
			ax.set_title(f"Subjid {subjid} {cat} last-odor poke time")
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, f"last_odor_poke_time_{cat}", subjids=[subjid], dates=date_vals)

	return figs


def trial_poke_duration(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
):
	"""
	Boxplots of trial poke duration by odor sequence, separated by response category.
	"""
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		frames = []
		for results_dir in results_dirs:
			frames.append(_load_trial_data(results_dir))
		df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
		if df.empty:
			continue

		completed = df[df.get("is_aborted") == False].copy()
		if completed.empty:
			continue

		categories = ["rewarded", "unrewarded", "timeout_delayed"]
		for cat in categories:
			cat_df = completed[completed.get("response_time_category") == cat]
			if cat_df.empty:
				continue

			groups = {}
			for _, row in cat_df.iterrows():
				seq = _parse_json_value(row.get("odor_sequence"))
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
				groups.setdefault(seq_label, []).append(float(duration_ms))

			if not groups:
				continue

			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: groups[k] for k in _order_sequence_labels(groups)}
			_plot_box_with_points(ax, ordered, "Trial Poke Duration (ms)", "Odor Sequence")
			ax.set_title(f"Subjid {subjid} {cat} trial poke duration")
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, f"trial_poke_duration_{cat}", subjids=[subjid], dates=date_vals)

	return figs


def first_odor_poke_duration(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
):
	"""
	Boxplot of first-odor poke duration grouped by odor name (completed trials only).
	"""
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		frames = []
		for results_dir in results_dirs:
			frames.append(_load_trial_data(results_dir))
		df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
		if df.empty:
			continue

		completed = df[df.get("is_aborted") == False].copy()
		if completed.empty:
			continue

		groups = {}
		for _, row in completed.iterrows():
			pos_dict = _parse_json_value(row.get("position_poke_times"))
			first_entry = _extract_position_entry(pos_dict, 1)
			if not first_entry:
				continue
			odor_name = first_entry.get("odor_name")
			poke_ms = first_entry.get("poke_time_ms")
			if odor_name is None or poke_ms is None:
				continue
			groups.setdefault(str(odor_name), []).append(float(poke_ms))

		if not groups:
			continue

		fig, ax = plt.subplots(figsize=(10, 5))
		ordered = {k: groups[k] for k in _order_odor_labels(groups)}
		_plot_box_with_points(ax, ordered, "Poke Duration", "Odor")
		ax.set_title(f"Subjid {subjid} first-odor poke duration")
		fig.tight_layout()
		figs.append(fig)
		if save:
			save_figure(fig, "first_odor_poke_duration", subjids=[subjid], dates=date_vals)

	return figs


def poke_time_all_pos(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
):
	"""
	Boxplot of poke duration pooled across positions, grouped by odor name.
	"""
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		frames = []
		for results_dir in results_dirs:
			frames.append(_load_trial_data(results_dir))
		df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
		if df.empty:
			continue

		completed = df[df.get("is_aborted") == False].copy()
		if completed.empty:
			continue

		groups = {}
		for _, row in completed.iterrows():
			pos_dict = _parse_json_value(row.get("position_poke_times"))
			if not isinstance(pos_dict, dict):
				continue
			for entry in pos_dict.values():
				if not isinstance(entry, dict):
					continue
				odor_name = entry.get("odor_name")
				poke_ms = entry.get("poke_time_ms")
				if odor_name is None or poke_ms is None:
					continue
				groups.setdefault(str(odor_name), []).append(float(poke_ms))

		if not groups:
			continue

		fig, ax = plt.subplots(figsize=(10, 5))
		ordered = {k: groups[k] for k in _order_odor_labels(groups)}
		_plot_box_with_points(ax, ordered, "Poke Duration (ms)", "Odor")
		ax.set_title(f"Subjid {subjid} poke duration (all positions)")
		fig.tight_layout()
		figs.append(fig)
		if save:
			save_figure(fig, "poke_time_all_pos", subjids=[subjid], dates=date_vals)

	return figs


def response_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	save: bool = False,
):
	"""
	Boxplots of response time by odor sequence (completed, non-timeout trials).
	"""
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
		frames = []
		for results_dir in results_dirs:
			frames.append(_load_trial_data(results_dir))
		df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
		if df.empty:
			continue

		completed = df[df.get("is_aborted") == False].copy()
		if completed.empty:
			continue

		completed = completed[completed.get("response_time_category") != "timeout_delayed"]
		if completed.empty:
			continue

		groups = {}
		for _, row in completed.iterrows():
			seq = _parse_json_value(row.get("odor_sequence"))
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
			rt_ms = (supply_dt - poke_end_dt).total_seconds() * 1000.0
			groups.setdefault(seq_label, []).append(float(rt_ms))

		if not groups:
			continue

		fig, ax = plt.subplots(figsize=(10, 5))
		ordered = {k: groups[k] for k in _order_sequence_labels(groups)}
		_plot_box_with_points(ax, ordered, "Response Time (ms)", "Odor Sequence")
		ax.set_title(f"Subjid {subjid} response time")
		fig.tight_layout()
		figs.append(fig)
		if save:
			save_figure(fig, "response_time", subjids=[subjid], dates=date_vals)

	return figs
