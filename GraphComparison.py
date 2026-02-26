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

def fmt_val_err(val, err, sig=4):
	"""
	Always returns a string safe to place in a tabular cell.
	Numeric output is wrapped in \\( ... \\) so \\pm is valid.
	"""
	if val is None or (isinstance(val, float) and not np.isfinite(val)):
		return r"--"

	try:
		v = float(val)
	except Exception:
		return latex_escape(str(val))

	e = None
	if err is not None:
		try:
			e = float(err)
		except Exception:
			e = None

	# no/invalid error => just value in math mode
	if e is None or (not np.isfinite(e)) or e == 0:
		vs = fmt_sci(v, sig=sig) if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3) else f"{v:.{sig}g}"
		return rf"\({vs}\)"

	# value ± error in math mode
	vs = fmt_sci(v, sig=sig) if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3) else f"{v:.{sig}g}"
	es = fmt_sci(e, sig=sig) if abs(e) >= 1e4 or (abs(e) > 0 and abs(e) < 1e-3) else f"{e:.{sig}g}"
	return rf"\({vs} \pm {es}\)"


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
			row_cells.append(fmt_val_err(val, err, sig=sig))
		lines.append(" & ".join(row_cells) + r" \\")

	lines.append(r"\hline")
	lines.append(r"\end{tabular}%")
	lines.append(r"}")  # closes resizebox
	lines.append(r"\end{table}")

	return "\n".join(lines)


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

	# -----------------------------
	# Helper: fetch delta_f from fit_params csv
	# -----------------------------
	def get_delta_f_shift(curr: int) -> float:
		if not apply_x_shift:
			return 0.0
		param_path = param_files.get(curr)
		if not param_path or (not param_path.exists()):
			return 0.0
		dfp = pd.read_csv(param_path)
		row = dfp.loc[dfp["parameter"] == "delta_f"]
		if row.empty:
			return 0.0
		try:
			return float(row.iloc[0]["value"])
		except Exception:
			return 0.0

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
		dshift = get_delta_f_shift(curr)
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

	ax_main.set_ylabel("Transmission (baseline-corrected)" + (" (normalised)" if normalise else ""))
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
		res_axes[-1].set_xlabel("UV Detuning (GHz)")
		plt.setp(ax_main.get_xticklabels(), visible=False)
	else:
		ax_main.set_xlabel("UV Detuning (GHz)")

	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()