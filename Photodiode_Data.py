import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator

# ----------------------------------------------------
# Matplotlib styling
# ----------------------------------------------------
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

# ----------------------------------------------------
# Helper: Load Tektronix CSV (TIME, CH1, CH2)
# ----------------------------------------------------
def load_tektronix_csv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Locate the "TIME,CH1,CH2" header row
    for i, line in enumerate(lines):
        if line.strip().startswith("TIME"):
            header_index = i
            break
    else:
        raise ValueError("Could not find TIME,CH1,CH2 header in " + filename)

    # Load numerical data
    data = np.loadtxt(filename, delimiter=",", skiprows=header_index+1)

    # Extract into arrays
    t = data[:, 0]
    ch1 = data[:, 1]
    ch2 = data[:, 2]

    return np.abs(t), np.abs(ch1), np.abs(ch2)


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

import glob

file = 

base_path = "Photodiode_Data/SILVER_NEW/"

files = sorted(glob.glob(base_path + "tek*ALL.csv"))

print("Found files:", len(files))
print(files)

power = 1.2e-6  # 1.2 µW in W

averages1 = []
averages2 = []
background1 = 0.0
background2 = 0.0

# ----------------------------------------------------
# Load all datasets
# ----------------------------------------------------
for i in range(0, len(files)):

    print("Loading index:", i)

    path = "ALL" + str(i).zfill(4) + "/A" + str(i).zfill(4) + "/"

    # Single Tektronix CSV containing TIME, CH1, CH2
    file_csv = base_path + "tek" + str(i).zfill(4) + "ALL.csv"

    # Load data
    t, ch1_arr, ch2_arr = load_tektronix_csv(file_csv)

    # Compute averages
    avg1 = np.mean(ch1_arr)
    avg2 = np.mean(ch2_arr)

    if i != len(files)-1:
        averages1.append(avg1)
        averages2.append(avg2)
    else:
        background1 = avg1
        background2 = avg2
        print("doneeee")

# ----------------------------------------------------
# Convert to arrays and subtract background
# ----------------------------------------------------
xs = np.linspace(1, 4, len(files)-1)
y1 = np.array(averages1)## - background1
y2 = np.array(averages2)# - background2

# ----------------------------------------------------
# Remove region for fitting
# ----------------------------------------------------
exclude = (xs > 2.0) & (xs < 3.5)
mask = ~exclude

# Linear fit to CH1
coeffs1 = np.polyfit(xs[mask], y1[mask], 1)
m1, c1 = coeffs1
fit_line1 = np.polyval(coeffs1, xs)

# ----------------------------------------------------
# Plot original + fit
# ----------------------------------------------------
plt.plot(xs, fit_line1, '--', label=f"CH1 fit\n y = {m1:.3g}x + {c1:.3g}")
plt.plot(xs, y1, marker="o", label="CH1 data")
plt.plot(xs, y2, marker="o", label="CH2 data")
plt.legend()
plt.xlabel("Voltage (V)")
plt.ylabel("Signal")
plt.savefig("Photodiode_Plot", dpi=300, bbox_inches='tight')
plt.show()

"""
# ----------------------------------------------------
# Scale to power
# ----------------------------------------------------
unscaled = y1 - fit_line1 + fit_line1[0]
scaled = unscaled * power / np.max(unscaled)

plt.plot(xs, scaled, marker="o")
plt.xlabel("Voltage (V)")
plt.ylabel("Power (W)")
plt.savefig("Photodiode_Plot_Adjusted", dpi=300, bbox_inches='tight')
plt.show()
"""