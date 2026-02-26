import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import re

# ----------------------------------------------------
# Matplotlib styling
# ----------------------------------------------------
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True


# =====================================================
# Custom (hard-coded) x-shift overrides in GHz
# These are applied to BOTH data and theory x-values.
# If a curr is listed here, it overrides fitted delta_f.
# =====================================================
CUSTOM_DELTA_F_OVERRIDE_GHZ = {
	9: 0.15000,   # <-- set your custom delta_f for curr9 here (GHz)
}

# Optional: if you want "extra" shift added on top of fitted delta_f instead of overriding,
# set this to True. Otherwise (default False) it REPLACES fitted delta_f for that curr.
CUSTOM_DELTA_F_IS_ADDITIVE = True


# -----------------------------
# File finders
# -----------------------------
def find_curve_files_by_curr(folder: Path):
	"""
	Returns dict {curr_number : Path_to_curve_file}
	Matches: baseline_corrected_curr{curr}.csv
	"""
	files = list(folder.glob("baseline_corrected_curr*.csv"))
	found = {}

	pattern = re.compile(r"curr(\d+)")
	for f in files:
		m = pattern.search(f.name)
		if not m:
			continue
		curr = int(m.group(1))
		found[curr] = f

	return found


def find_param_files_by_curr(folder: Path):
	"""
	Returns dict {curr_number : Path_to_param_file}
	Matches: fit_params_curr{curr}.csv
	"""
	files = list(folder.glob("fit_params_curr*.csv"))
	found = {}

	pattern = re.compile(r"curr(\d+)")
	for f in files:
		m = pattern.search(f.name)
		if not m:
			continue
		curr = int(m.group(1))
		found[curr] = f

	return found


# -----------------------------
# LaTeX helpers
# -----------------------------
def latex_escape(s: str) -> str:
	# minimal escape set (enough for underscores etc.)
	return (s.replace("\\", r"\textbackslash{}")
			 .replace("_", r"\_")
			 .replace("%", r"\%")
			 .replace("&", r"\&")
			 .replace("#", r"\#")
			 .replace("{", r"\{")
			 .replace("}", r"\}")
			 .replace("^", r"\^{}")
			 .replace("~", r"\~{}"))


def fmt_sci(x, sig=4):
	"""Return LaTeX math-mode scientific notation like 1.23\\times 10^{4}."""
	if x is None:
		return None
	x = float(x)
	if not np.isfinite(x):
		return None
	if x == 0:
		return "0"
	exp = int(np.floor(np.log10(abs(x))))
	mant = x / (10 ** exp)
	# keep mantissa in [1,10)
	return f"{mant:.{sig}g}\\times 10^{{{exp}}}"

import numpy as np

import numpy as np


import numpy as np

def round_err_and_match_value(val, err):
	"""
	Physics-style rounding:
	  - Round error to 1 s.f.
	  - If leading digit is 1, round error to 2 s.f.
	  - Round value to the same decimal place as the rounded error

	Returns:
	  (v_rounded, e_rounded, decimals)
	where decimals is the number of decimal places used in round(..., decimals)
	(can be negative for rounding to 10s, 100s, etc.)
	"""
	v = float(val)
	e = float(err)

	if (not np.isfinite(v)) or (not np.isfinite(e)) or e <= 0:
		return v, None, None

	exp = int(np.floor(np.log10(abs(e))))     # e = mant * 10^exp
	mant = e / (10 ** exp)                    # 1 <= mant < 10

	sig = 2 if 1 <= mant < 2 else 1           # 2 s.f. only if leading digit is 1
	decimals = -exp + (sig - 1)

	e_rounded = round(e, decimals)
	v_rounded = round(v, decimals)

	return v_rounded, e_rounded, decimals


def fmt_num_latex(x, sci_sig=4):
	"""
	Format a number for LaTeX math mode.
	Uses scientific notation for very large/small values.
	"""
	x = float(x)
	if not np.isfinite(x):
		return r"--"
	if x == 0:
		return "0"

	ax = abs(x)
	if ax >= 1e4 or ax < 1e-3:
		exp = int(np.floor(np.log10(ax)))
		mant = x / (10 ** exp)
		return f"{mant:.{sci_sig}g}\\times 10^{{{exp}}}"
	else:
		return f"{x:g}"


def fmt_val_err_phys(val, err, sci_sig=4):
	"""
	Return a LaTeX-safe string for a table cell:
	  - value and error rounded with physics rule
	  - value rounded to match error decimal place
	  - both wrapped in \\( ... \\) so \\pm works
	"""
	if val is None:
		return r"--"

	try:
		v = float(val)
	except Exception:
		return str(val)

	# no error or invalid error -> just value
	try:
		e = float(err) if err is not None else None
	except Exception:
		e = None

	if e is None or (not np.isfinite(e)) or e <= 0:
		return rf"\({fmt_num_latex(v, sci_sig=sci_sig)}\)"

	v_r, e_r, decimals = round_err_and_match_value(v, e)

	# If we’re not in sci territory, force fixed decimals so the value matches the error
	ax = max(abs(v_r), abs(e_r))
	use_sci = (ax >= 1e4) or (0 < ax < 1e-3)

	if not use_sci and decimals is not None and decimals > 0:
		fmt = f"{{:.{decimals}f}}"
		v_str = fmt.format(v_r)
		e_str = fmt.format(e_r)
	else:
		# either decimals <= 0 or scientific formatting chosen
		v_str = fmt_num_latex(v_r, sci_sig=sci_sig)
		e_str = fmt_num_latex(e_r, sci_sig=sci_sig)

	return rf"\({v_str} \pm {e_str}\)"


def build_param_table(currs, param_files, param_list=None, fitted_only=False, sig=4):
	"""
	Returns a LaTeX table as a string.

	param_list:
	  - None => default: union of all parameters in the selected files
	  - otherwise: list of parameter names to include (e.g. ["Temp","a","delta_f","b0"])

	fitted_only:
	  - True => include only rows whose status == "FIT" (but still only within param_list if given)
	"""
	# Load params for each curr
	by_curr = {}
	all_params = set()

	for c in currs:
		if c not in param_files:
			continue
		dfp = pd.read_csv(param_files[c])
		dfp["parameter"] = dfp["parameter"].astype(str)
		by_curr[c] = dfp
		all_params.update(dfp["parameter"].tolist())

	if not by_curr:
		return None

	# choose params to print
	if param_list is None:
		params = sorted(all_params, key=lambda x: (x.startswith("b"), x))  # baseline b's last-ish
	else:
		params = [p.strip() for p in param_list if p.strip()]
		params = [p for p in params if p in all_params]

	# optionally filter to FIT only
	if fitted_only:
		keep = []
		for p in params:
			any_fit = False
			for c, dfp in by_curr.items():
				row = dfp.loc[dfp["parameter"] == p]
				if not row.empty and str(row.iloc[0]["status"]).upper() == "FIT":
					any_fit = True
					break
			if any_fit:
				keep.append(p)
		params = keep

	# header
	cols = ["Parameter"] + [f"curr{c}" for c in currs]
	colspec = "l" + "c" * len(currs)

	lines = []
	lines.append(r"\begin{table}[h]")
	lines.append(r"\centering")
	lines.append(r"\small")  # optional: helps fit more comfortably

	# --- key change: resize to page width ---
	lines.append(r"\resizebox{\linewidth}{!}{%")
	lines.append(rf"\begin{{tabular}}{{{colspec}}}")
	lines.append(r"\hline")
	lines.append(" & ".join(cols) + r" \\")
	lines.append(r"\hline")

	# body
	for p in params:
		row_cells = [latex_escape(p)]
		for c in currs:
			dfp = by_curr.get(c)
			if dfp is None:
				row_cells.append(r"--")
				continue
			r0 = dfp.loc[dfp["parameter"] == p]
			if r0.empty:
				row_cells.append(r"--")
				continue
			val = r0.iloc[0].get("value", np.nan)
			err = r0.iloc[0].get("error", np.nan)
			row_cells.append(fmt_val_err_phys(val, err))
		lines.append(" & ".join(row_cells) + r" \\")

	lines.append(r"\hline")
	lines.append(r"\end{tabular}%")
	lines.append(r"}")  # closes resizebox
	lines.append(r"\end{table}")

	return "\n".join(lines)

def load_curve_for_subtraction(
	curr: int,
	curve_files: dict,
	param_files: dict,
	apply_x_shift: bool,
	hide_bad_errors: bool,
	err_thresh: float = 0.03
):
	"""
	Load one baseline_corrected_curr{curr}.csv and apply the same cleaning and optional delta_f shift.

	Returns dict with keys: x, y, yerr, yfit
	where:
	  y    = Transmission_BaselineCorrected (data)
	  yfit = Theory_NoBaseline (theory)
	"""

	df = pd.read_csv(curve_files[curr])

	x = df["detuning_uv_GHz"].to_numpy(float)
	y = df["Transmission_BaselineCorrected"].to_numpy(float)
	yerr = df["TransmissionErr_BaselineCorrected"].to_numpy(float) if "TransmissionErr_BaselineCorrected" in df.columns else None
	yfit = df["Theory_NoBaseline"].to_numpy(float)

	mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yfit)
	if yerr is not None:
		mask &= np.isfinite(yerr)
		if hide_bad_errors:
			mask &= (np.abs(yerr) <= err_thresh)

	x, y, yfit = x[mask], y[mask], yfit[mask]
	if yerr is not None:
		yerr = yerr[mask]

	order = np.argsort(x)
	x, y, yfit = x[order], y[order], yfit[order]
	if yerr is not None:
		yerr = yerr[order]

	dshift = get_delta_f_shift(curr, apply_x_shift=apply_x_shift, param_files=param_files)
	if dshift != 0.0:
		print(f"curr{curr}: applying delta_f shift = {dshift:.6g} GHz (subtraction)")
		x = x + dshift

	return {"x": x, "y": y, "yerr": yerr, "yfit": yfit}


def plot_theory_minus_data(
	curve_files: dict,
	param_files: dict,
	apply_x_shift: bool,
	hide_bad_errors: bool,
	theory_curr: int = 8,
	data_curr: int = 9,
	err_thresh: float = 0.03
):
	"""
	Plot: (theory from theory_curr) - (data from data_curr)

	- Interpolates theory onto the data x-grid (after any delta_f shift).
	- Error bars come from the data only.
	- No residual subplots; separate figure.
	"""

	if theory_curr not in curve_files:
		raise ValueError(f"Missing curve file for theory curr{theory_curr}")
	if data_curr not in curve_files:
		raise ValueError(f"Missing curve file for data curr{data_curr}")

	th = load_curve_for_subtraction(
		curr=theory_curr,
		curve_files=curve_files,
		param_files=param_files,
		apply_x_shift=apply_x_shift,
		hide_bad_errors=hide_bad_errors,
		err_thresh=err_thresh
	)
	da = load_curve_for_subtraction(
		curr=data_curr,
		curve_files=curve_files,
		param_files=param_files,
		apply_x_shift=apply_x_shift,
		hide_bad_errors=hide_bad_errors,
		err_thresh=err_thresh
	)

	# interpolate theory (yfit) onto data x-grid
	# (assumes monotonic x, already sorted)
	x_data = da["x"]
	y_data = da["y"]
	yerr_data = da["yerr"]

	x_th = th["x"]
	y_th = th["yfit"]

	# guard: if theory x-range doesn't cover data x-range, np.interp clamps to endpoints;
	# instead mask to overlap region for cleaner physics.
	x_min = max(np.min(x_data), np.min(x_th))
	x_max = min(np.max(x_data), np.max(x_th))
	overlap = (x_data >= x_min) & (x_data <= x_max)

	if not np.any(overlap):
		raise ValueError("No overlap in x-range between chosen data and theory curves (after any delta_f shifts).")

	x_use = x_data[overlap]
	y_use = y_data[overlap]
	yerr_use = yerr_data[overlap] if yerr_data is not None else None

	yth_interp = np.interp(x_use, x_th, y_th)

	diff = yth_interp - y_use

	fig, ax = plt.subplots(figsize=(8, 4.8))
	ax.axhline(0, color="grey", lw=1)

	if yerr_use is not None:
		ax.errorbar(
			x_use, diff,
			yerr=np.abs(yerr_use),
			fmt=".", capsize=0,
			label=rf"curr{theory_curr} theory $-$ curr{data_curr} data"
		)
	else:
		ax.plot(x_use, diff, ".", label=rf"curr{theory_curr} theory $-$ curr{data_curr} data")

	ax.set_xlabel("Linear Detuning (GHz)")
	ax.set_ylabel("Theory − Data")
	ax.legend()
	ax.minorticks_on()
	fig.tight_layout()
	plt.show()

# -----------------------------
# Helper: fetch delta_f from fit_params csv
# -----------------------------
def get_delta_f_shift(curr: int, apply_x_shift: bool, param_files: dict) -> float:
	"""
	Returns the x-shift (GHz) to apply for this curr.

	Rules:
	- If apply_x_shift is False: return 0
	- Otherwise:
		* read fitted delta_f from fit_params_curr{curr}.csv (if present)
		* apply hard-coded override/additive shift for chosen currs
	"""
	if not apply_x_shift:
		return 0.0

	fitted = 0.0
	param_path = param_files.get(curr)

	if param_path and param_path.exists():
		dfp = pd.read_csv(param_path)
		row = dfp.loc[dfp["parameter"] == "delta_f"]
		if not row.empty:
			try:
				fitted = float(row.iloc[0]["value"])
			except Exception:
				fitted = 0.0

	# ---- hard-coded custom shift (curr9 etc.) ----
	if curr in CUSTOM_DELTA_F_OVERRIDE_GHZ:
		custom = float(CUSTOM_DELTA_F_OVERRIDE_GHZ[curr])
		if CUSTOM_DELTA_F_IS_ADDITIVE:
			return fitted + custom
		else:
			return custom

	return fitted

# -----------------------------
# Main
# -----------------------------
def main():
	folder = Path(__file__).resolve().parent
	print("Using folder:", folder.resolve())

	curve_files = find_curve_files_by_curr(folder)
	if not curve_files:
		print("No baseline_corrected_currX.csv files found.")
		return

	available = sorted(curve_files.keys())
	print("\nAvailable curr values (curves):", available)
	
	raw = input("Type curr numbers to plot (e.g. 4 7 9) or press Enter for ALL: ").strip()
	if raw == "":
		selected = available
	else:
		selected = [int(x) for x in raw.split()]
		missing = [c for c in selected if c not in curve_files]
		if missing:
			raise ValueError(f"Missing curr values (curves): {missing}. Available: {available}")

	hide_bad_errors = (
		input("Hide points with anomalous errors (> 0.03)? (Y/n, default Y): ")
		.strip().lower() != "n"
	)

	# ---- LaTeX table options ----
	make_table = (input("Print LaTeX table of fitted parameters? (y/N, default N): ").strip().lower() == "y")

	param_files = find_param_files_by_curr(folder)
	if make_table:
		missing_params = [c for c in selected if c not in param_files]
		if missing_params:
			print(f"\nWarning: no fit_params_curr*.csv found for curr: {missing_params}")
			print("Table will omit those curr columns or show '--' where missing.\n")

		fitted_only = (input("Only include parameters with status==FIT? (y/N, default N): ").strip().lower() == "y")

		raw_params = input(
			"Parameters to include (space/comma separated) or Enter for DEFAULT (all available): "
		).strip()

		if raw_params == "":
			param_list = None  # default union of all parameters
		else:
			param_list = [p for p in raw_params.replace(",", " ").split()]

		try:
			sig = int(input("Significant figures for numbers (default 4): ").strip() or "4")
		except ValueError:
			sig = 4

		latex = build_param_table(
			currs=selected,
			param_files=param_files,
			param_list=param_list,
			fitted_only=fitted_only,
			sig=sig
		)

		if latex is None:
			print("\nNo parameter files could be loaded, skipping LaTeX table.\n")
		else:
			print("\n===== COPY/PASTE LaTeX TABLE =====\n")
			print(latex)
			print("\n===== END LaTeX TABLE =====\n")

		# ---- Plot options ----
	normalise = (input("Normalise each trace to max=1? (y/N, default N): ").strip().lower() == "y")
	offset = float(input("Vertical offset between traces (default 0): ").strip() or "0")
	use_errorbars = (input("Use error bars? (Y/n, default Y): ").strip().lower() != "n")

	apply_x_shift = (input("Apply x-axis correction using fitted delta_f? (Y/n, default Y): ").strip().lower() != "n")

	# ---- Residual subplot options ----
	# default: curr8 only (if available); otherwise: none unless user chooses
	raw_res = input(
		"Residual panels: type curr numbers (e.g. '8 9'), 'all', or Enter for DEFAULT (curr8): "
	).strip().lower()

	if raw_res in ("all", "a"):
		residual_currs = list(selected)
	elif raw_res == "":
		residual_currs = [8] if 8 in selected else []
	else:
		residual_currs = [int(x) for x in raw_res.replace(",", " ").split() if x.strip()]
		missing_res = [c for c in residual_currs if c not in selected]
		if missing_res:
			raise ValueError(f"Residual curr values not in selected plot set: {missing_res}. Selected: {selected}")

	show_hist = (input("Show Gaussian histogram beside each residual panel? (Y/n, default Y): ").strip().lower() != "n")

	do_subtract = (input("Plot theory - data subtraction curve? (y/N, default N): ").strip().lower() == "y")

	if do_subtract:
		raw_sub = input("Subtraction: enter 'theory_curr data_curr' (default '8 9'): ").strip()
		if raw_sub == "":
			theory_curr, data_curr = 8, 9
		else:
			parts = [int(x) for x in raw_sub.replace(",", " ").split()]
			if len(parts) != 2:
				raise ValueError("Please enter exactly two integers: theory_curr data_curr")
			theory_curr, data_curr = parts

		plot_theory_minus_data(
			curve_files=curve_files,
			param_files=param_files,
			apply_x_shift=apply_x_shift,
			hide_bad_errors=hide_bad_errors,
			theory_curr=theory_curr,
			data_curr=data_curr,
			err_thresh=0.03
		)

	# -----------------------------
	# Load all curves once (so we can reuse for main + residuals)
	# -----------------------------
	curves = {}  # curr -> dict of arrays
	for curr in selected:
		df = pd.read_csv(curve_files[curr])

		x = df["detuning_uv_GHz"].to_numpy(float)
		y = df["Transmission_BaselineCorrected"].to_numpy(float)
		yerr = df["TransmissionErr_BaselineCorrected"].to_numpy(float) if "TransmissionErr_BaselineCorrected" in df.columns else None
		yfit = df["Theory_NoBaseline"].to_numpy(float)

		# residuals are optional (only if you saved them)
		resid = df["Residuals_Norm"].to_numpy(float) if "Residuals_Norm" in df.columns else None

		mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yfit)

		if yerr is not None:
			mask &= np.isfinite(yerr)

			# ---------------------------------
			# Optional anomalous error rejection
			# ---------------------------------
			if hide_bad_errors:
				bad_mask = np.abs(yerr) > 0.03
				n_bad = np.count_nonzero(bad_mask)
				if n_bad > 0:
					print(f"curr{curr}: removed {n_bad} points with error > 0.03")
				mask &= (np.abs(yerr) <= 0.03)

		if resid is not None:
			mask &= np.isfinite(resid)

		x, y, yfit = x[mask], y[mask], yfit[mask]
		if yerr is not None:
			yerr = yerr[mask]
		if resid is not None:
			resid = resid[mask]

		# sort by x for clean lines
		order = np.argsort(x)
		x, y, yfit = x[order], y[order], yfit[order]
		if yerr is not None:
			yerr = yerr[order]
		if resid is not None:
			resid = resid[order]

		# x-shift correction (delta_f)
		dshift = get_delta_f_shift(curr, apply_x_shift=apply_x_shift, param_files=param_files)
		if dshift != 0.0:
			print(f"curr{curr}: applying delta_f shift = {dshift:.6g} GHz")
			x = x + dshift

		curves[curr] = {"x": x, "y": y, "yerr": yerr, "yfit": yfit, "resid": resid}

	# -----------------------------
	# Normalisation (based on data y, same scaling applied to fit and error)
	# -----------------------------
	if normalise:
		for curr in selected:
			y = curves[curr]["y"]
			m = np.nanmax(np.abs(y))
			if m > 0:
				curves[curr]["y"] = curves[curr]["y"] / m
				curves[curr]["yfit"] = curves[curr]["yfit"] / m
				if curves[curr]["yerr"] is not None:
					curves[curr]["yerr"] = curves[curr]["yerr"] / m

	# -----------------------------
	# Figure layout: main axis + N residual axes, each with optional histogram axis
	# -----------------------------
		# -----------------------------
	# Figure layout: main axis (LEFT ONLY) + N residual axes (LEFT) + hist axes (RIGHT)
	# Histograms extend past the main axis to the right.
	# -----------------------------
	n_res = len(residual_currs)

	if n_res == 0:
		fig, ax_main = plt.subplots(figsize=(8, 4.8))
		res_axes = []
		hist_axes = []
	else:
		height_ratios = [3.0] + [1.0] * n_res
		# column 0 = main/residual width; column 1 = extra histogram strip (extends beyond main)
		width_ratios = [1.0, 0.22]

		fig = plt.figure(figsize=(9, 3.2 + 1.8 * n_res))
		gs = fig.add_gridspec(
			nrows=1 + n_res,
			ncols=2,
			height_ratios=height_ratios,
			width_ratios=width_ratios,
			hspace=0.08,
			wspace=0.05
		)

		# IMPORTANT: main axis uses ONLY left column
		ax_main = fig.add_subplot(gs[0, 0])

		res_axes = []
		hist_axes = []
		for i in range(n_res):
			axr = fig.add_subplot(gs[1 + i, 0], sharex=ax_main)
			res_axes.append(axr)

			if show_hist:
				ah = fig.add_subplot(gs[1 + i, 1], sharey=axr)
				hist_axes.append(ah)
			else:
				hist_axes.append(None)

		# Optional: hide unused top-right cell (keeps layout clean)
		ax_blank = fig.add_subplot(gs[0, 1])
		ax_blank.axis("off")

	# -----------------------------
	# MAIN PLOT (uses shifted x automatically)
	# -----------------------------
	for i, curr in enumerate(selected):
		x = curves[curr]["x"]          # <- already includes delta_f shift if enabled
		y = curves[curr]["y"]
		yerr = curves[curr]["yerr"]
		yfit = curves[curr]["yfit"]

		y_off = y + i * offset
		yfit_off = yfit + i * offset

		if use_errorbars and (yerr is not None):
			ax_main.errorbar(x, y_off, yerr=np.abs(yerr), fmt=".", capsize=0, label=f"curr{curr} data")
		else:
			ax_main.plot(x, y_off, ".", label=f"curr{curr} data")

		ax_main.plot(x, yfit_off, "-", linewidth=2, label=f"curr{curr} theory")

	ax_main.set_ylabel("Transmission" + (" (normalised)" if normalise else ""))
	ax_main.legend()
	ax_main.minorticks_on()

	# -----------------------------
	# RESIDUAL PANELS (x uses the same shifted axis as main)
	# -----------------------------
	def gauss_pdf(z):
		return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

	for idx, curr in enumerate(residual_currs):
		axr = res_axes[idx]
		ah = hist_axes[idx]

		x = curves[curr]["x"]          # <- shifted x (delta_f applied if enabled)
		resid = curves[curr]["resid"]

		axr.axhline(0, linewidth=1)

		if resid is None:
			axr.text(0.02, 0.7, f"curr{curr}: no 'Residuals_Norm' column", transform=axr.transAxes)
			axr.set_ylabel(f"Res (curr{curr})")
			axr.minorticks_on()
			continue

		axr.errorbar(
			x, resid,
			yerr=np.ones_like(resid, dtype=float),
			fmt=".", capsize=0, markersize=4
		)
		axr.set_ylabel(f"Res (curr{curr})")
		axr.minorticks_on()

		if ah is not None:
			r = resid[np.isfinite(resid)]
			if r.size > 0:
				rmin = min(np.floor(r.min()), -4)
				rmax = max(np.ceil(r.max()),  4)
				edges = np.arange(rmin - 0.5, rmax + 0.5 + 1e-9, 1.0)

				ah.hist(
					r, bins=edges, density=True, orientation="horizontal",
					alpha=0.6, edgecolor="black", linewidth=0.8
				)

				ys = np.linspace(edges[0], edges[-1], 400)
				ah.plot(gauss_pdf(ys), ys, linewidth=2)
				ah.axhline(0, linewidth=1)
				ah.set_xlabel("PDF")
				ah.set_xlim(left=0)
				ah.tick_params(direction="in", top=True, right=True)
				plt.setp(ah.get_yticklabels(), visible=False)

	# X-label only on bottom-most left plot axis
	if n_res > 0:
		res_axes[-1].set_xlabel("Linear Detuning (GHz)")
		plt.setp(ax_main.get_xticklabels(), visible=False)
	else:
		ax_main.set_xlabel("Linear Detuning (GHz)")

	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()