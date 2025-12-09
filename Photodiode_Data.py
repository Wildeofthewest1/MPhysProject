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

c = 3e8

frequencies = (456777.182,
			   456777.031,
			   456776.861,
			   456776.687,
			   456776.513,
			   456776.324,
			   456776.140,
			   456775.963,
			   456775.772,
			   456775.582,
			   456775.389,
			   456775.196,
			   456774.994,
			   456774.795,
			   456774.602,
			   456774.402,
			   456774.214,
			   456774.026,
			   456773.816,
			   456773.620,
			   456773.396,
			   456773.201,
			   456772.985,
			   456772.769,
			   456772.562,
			   456772.341,
			   456772.131,
			   456771.902,
			   456771.694,
			   456771.464,
			   456771.252)

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

	return t, ch1, ch2


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

import glob

folder = "SilverSpecFirst/"
#folder = "SilverSpecSecond/"

base_path = "Photodiode_Data/" + folder

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

	#print("Loading index:", i)

	path = "ALL" + str(i).zfill(4) + "/A" + str(i).zfill(4) + "/"

	# Single Tektronix CSV containing TIME, CH1, CH2
	file_csv = base_path + "tek" + str(i).zfill(4) + "ALL.csv"

	# Load data
	t, ch1_arr, ch2_arr = load_tektronix_csv(file_csv)

	# Compute averages
	avg1 = np.mean(ch1_arr)
	avg2 = np.mean(ch2_arr)
	
	avg1err = np.std(ch1_arr)/np.sqrt(len(ch1_arr))
	avg2err = np.std(ch2_arr)/np.sqrt(len(ch2_arr))

	if i != len(files)-1:
		averages1.append((avg1, avg1err))
		averages2.append((avg2, avg2err))
	else:
		background1 = (avg1, avg1err)
		background2 = (avg2, avg2err)
		print("doneeee")


averages1_means = np.array([m for (m, e) in averages1])
averages1_errs  = np.array([e for (m, e) in averages1])

averages2_means = np.array([m for (m, e) in averages2])
averages2_errs  = np.array([e for (m, e) in averages2])

background1_mean, background1_err = background1
background2_mean, background2_err = background2

# ----------------------------------------------------
# Convert to arrays and subtract background
# ----------------------------------------------------
xs1 = np.linspace(1, 4, len(files)-1)
xs = -np.array(frequencies)*2 + (c / (328.1625)) - 633

y1 = np.abs(np.abs(averages1_means) - np.abs(background1_mean))
y2 = np.abs(np.abs(averages2_means) - np.abs(background2_mean))

y1_err = np.sqrt(averages1_errs**2 + background1_err**2)
y2_err = np.sqrt(averages2_errs**2 + background2_err**2)

#print(y1,y2)

# ----------------------------------------------------
# Remove region for fitting
# ----------------------------------------------------
exclude = (xs1 > 2.0) & (xs1 < 3.5)
mask = ~exclude

# Linear fit to CH1
coeffs1 = np.polyfit(xs1[mask], y1[mask], 1)
m1, c1 = coeffs1
fit_line1 = np.polyval(coeffs1, xs1)

# ----------------------------------------------------
# Plot original + fit
# ----------------------------------------------------
plt.plot(xs1, fit_line1, '--', label=f"CH1 fit\n y = {m1:.3g}x + {c1:.3g}")
plt.errorbar(xs1, y1, yerr = y1_err, marker="o", label="CH1 data")
plt.errorbar(xs1, y2, yerr = y2_err, marker="o", label="CH2 data")
plt.legend()
plt.xlabel("Voltage (GHz)")
plt.ylabel("Signal")
plt.savefig("Photodiode_Plot", dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# Transmission and uncertainty propagation
# ----------------------------------------------------
transmission = y1 / y2

transmission_err = np.abs(transmission) * np.sqrt(
    (y1_err / y1)**2 +
    (y2_err / y2)**2
)

plt.errorbar(xs, transmission/np.max(transmission),
             yerr=np.abs(transmission_err/np.max(transmission)),
             marker='o')

plt.xlabel("Detuning (GHz)")
plt.ylabel("Transmission")

plt.ylim(0,1.1)
plt.yticks([0.0,0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7 ,0.8,0.9, 1.0])

plt.tight_layout()

plt.savefig("Photodiode_Transmission", dpi=300, bbox_inches='tight')
plt.show()
