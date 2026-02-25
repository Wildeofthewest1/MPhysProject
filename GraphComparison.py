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


def find_latest_files_by_curr(folder: Path):
	"""
	Returns dict:
		{ curr_number : Path_to_latest_file }
	"""
	files = list(folder.glob("baseline_corrected_curr*.csv"))
	latest = {}

	pattern = re.compile(r"curr(\d+)")

	for f in files:
		m = pattern.search(f.name)
		if not m:
			continue
		curr = int(m.group(1))

		# filename contains sortable timestamp => lexicographic compare works
		if curr not in latest or f.name > latest[curr].name:
			latest[curr] = f

	return latest


def main():
	folder = Path(__file__).resolve().parent
	print("Using folder:", folder.resolve())

	latest_files = find_latest_files_by_curr(folder)
	if not latest_files:
		print("No baseline_corrected_currX_*.csv files found.")
		return

	available = sorted(latest_files.keys())
	print("\nAvailable curr values:", available)

	raw = input("Type curr numbers to plot (e.g. 4 7 9) or press Enter for ALL: ").strip()
	if raw == "":
		selected = available
	else:
		selected = [int(x) for x in raw.split()]
		missing = [c for c in selected if c not in latest_files]
		if missing:
			raise ValueError(f"Missing curr values: {missing}. Available: {available}")

	normalise = (input("Normalise each trace to max=1? (y/N): ").strip().lower() == "y")
	offset = float(input("Vertical offset between traces (default 0): ").strip() or "0")
	use_errorbars = (input("Use error bars? (Y/n, default Y): ").strip().lower() != "n")

	fig, ax = plt.subplots()

	for i, curr in enumerate(selected):
		path = latest_files[curr]
		df = pd.read_csv(path)

		x = df["detuning_uv_GHz"].to_numpy(float)
		y = df["Transmission_BaselineCorrected"].to_numpy(float)
		yerr = df["TransmissionErr_BaselineCorrected"].to_numpy(float) if "TransmissionErr_BaselineCorrected" in df.columns else None
		yfit = df["Theory_NoBaseline"].to_numpy(float)

		# remove NaN/Inf consistently (just in case)
		mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yfit)
		if yerr is not None:
			mask &= np.isfinite(yerr)

		x, y, yfit = x[mask], y[mask], yfit[mask]
		if yerr is not None:
			yerr = yerr[mask]

		# sort by x for nice lines
		order = np.argsort(x)
		x, y, yfit = x[order], y[order], yfit[order]
		if yerr is not None:
			yerr = yerr[order]

		if normalise:
			m = np.nanmax(np.abs(y))
			if m > 0:
				y = y / m
				yfit = yfit / m
				if yerr is not None:
					yerr = yerr / m

		y = y + i * offset
		yfit = yfit + i * offset

		if use_errorbars and (yerr is not None):
			ax.errorbar(x, y, yerr=np.abs(yerr), fmt=".", capsize=0, label=f"curr{curr} data")
		else:
			ax.plot(x, y, ".", label=f"curr{curr} data")

		ax.plot(x, yfit, "-", linewidth=2, label=f"curr{curr} theory")

	ax.set_xlabel("UV Detuning (GHz)")
	ax.set_ylabel("Transmission (baseline-corrected)" + (" (normalised)" if normalise else ""))
	ax.legend()
	ax.minorticks_on()
	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()