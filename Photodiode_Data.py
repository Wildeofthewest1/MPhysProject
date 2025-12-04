import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os
from scipy.integrate import cumulative_trapezoid as cumtrapz
from matplotlib import rcParams
import pandas as pd

from matplotlib.ticker import AutoMinorLocator

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['xtick.minor.size'] = 2
rcParams['ytick.minor.size'] = 2

# --- Configuration ---
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

base_path = "Photodiode_Data/SILVER/"

power = 1.2e-6 #microwatts

###Reading files

averages1 = []
averages2 = []
background1 = 0.0
background2 = 0.0

for i in range(0,31+1):

	print(i)

	path = "ALL" + str(i).zfill(4) + "/A" + str(i).zfill(4)

	CH1 = "CH1.CSV"
	CH2 = "CH1.CSV"

	#CH1 Data
	df1 = pd.read_csv(base_path + path + CH1, header=None)
	idx1 = df1[df1[0] == "Waveform Data"].index[0]
	arr1 = df1.iloc[idx1+1:, 0].dropna().astype(float).to_numpy()
	avg1 = np.abs(np.average(arr1))

	#CH2 Data
	df2 = pd.read_csv(base_path + path + CH2, header=None)
	idx2 = df2[df2[0] == "Waveform Data"].index[0]
	arr2 = df2.iloc[idx2+1:, 0].dropna().astype(float).to_numpy()
	avg2 = np.abs(np.average(arr2))

	if i > 0:
		averages1.append(avg1)
		averages2.append(avg2)
	else:
		background1 = avg1
		background2 = avg2

xs = np.linspace(1, 4, 31)
y1 = np.array(averages1) - background1
y2 = np.array(averages2) - background2

# ---------------------------
# OPTION 1: Exclude by x-range
# e.g. ignore 2.0 < x < 3.0
# ---------------------------
exclude = (xs > 2.0) & (xs < 3.5)   # change these bounds as needed
mask = ~exclude                      # True where we KEEP points

# Fit a line to CH1 only on allowed points
coeffs1 = np.polyfit(xs[mask], y1[mask], 1)  # degree 1 = straight line
m1, c1 = coeffs1
fit_line1 = np.polyval(coeffs1, xs)          # evaluate line at all xs (for plotting)

plt.plot(xs, fit_line1, '--', label=f"CH1 fit\n y = {m1:.3g}x + {c1:.3g}")
plt.plot(xs, y1, marker="o", label="CH1 data")

plt.legend()
plt.xlabel("Voltage (V)")
plt.ylabel("Signal")

plt.savefig("Photodiode_Plot", dpi=300, bbox_inches='tight')

plt.show()

unscaled = y1-fit_line1 + fit_line1[0]

scaled = unscaled * power/np.max(unscaled)

plt.plot(xs, scaled,marker="o")

plt.xlabel("Voltage (V)")
plt.ylabel("Power (W)")

plt.savefig("Photodiode_Plot_Adjusted", dpi=300, bbox_inches='tight')

plt.show()