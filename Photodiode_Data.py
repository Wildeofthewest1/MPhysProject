import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

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
# Configuration
# ----------------------------------------------------
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
#os.chdir(r"C:\\Users\\Matt\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

c = 2.99792458e8

"""

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

frequencies2 = (456775.401,
				456775.362,
				456775.318,
				456775.272,
				456775.223,
				456775.169,
				456775.115,
				456775.058,
				456775.006,
				456774.944,
				456774.884,
				456774.829,
				456774.773,
				456774.710,
				456774.652,
				456774.592,
				456774.534,
				456774.468,
				456774.399,
				456774.341,
				456774.286,
				456774.228,
				456774.152,
				456774.101,
				456774.044,
				456773.982,
				456773.921,
				456773.867,
				456773.809,
				456773.750,
				456773.692,
				456773.632,
				456773.565,
				456773.506,
				456773.442,
				456773.384,
				456773.317,
				456773.253,
				456773.191,
				456773.128,
				456773.064,
				456773.007,
				456772.947,
				456772.886,
				456772.810,
				456772.754,
				456772.686,
				456772.623,
				456772.559,
				456772.504,
				456772.446)



df = pd.DataFrame({
    "freq1": frequencies,
})

df.to_csv("frequencies1.csv", index=False)

df = pd.DataFrame({
    "freq2": frequencies2
})

df.to_csv("frequencies2.csv", index=False)

"""

"""
frequencies3 = (456778.966,
				456778.967,
				456778.884,
				456778.773,
				456778.638,
				456778.495,
				456778.346,
				456778.185,
				456778.025,
				456777.838,
				456777.668,
				456777.507,
				456777.308,
				456777.124,
				456776.932,
				456776.743,
				456776.544,
				456776.334,
				456776.148,
				456775.954,
				456775.864,
				456775.787,
				456775.740,
				456775.625,
				456775.569,
				456775.506,
				456775.446,
				456775.382,
				456775.319,
				456775.263,
				456775.203,
				456775.144,
				456775.079,
				456775.020,
				456774.961,
				456774.902,
				456774.835,
				456774.772,
				456774.715,
				456774.636,
				456774.578,
				456774.518,
				456774.448,
				456774.847,
				456774.828,
				456774.815,
				456774.793,
				456774.777,
				456774.757,
				456774.732,
				456774.714,
				456774.697,
				456774.684,
				456774.663,
				456774.647,
				456774.627,
				456774.610,
				456774.589,
				456774.570,
				456774.552,
				456774.531,
				456774.513,
				456774.495,
				456774.471,
				456774.452,
				456774.434,
				456774.415,
				456774.398,
				456774.378,
				456774.361,
				456774.341,
				456774.320,
				456774.301,
				456774.284,
				456774.266,
				456774.243,
				456774.233,
				456774.203,
				456774.185,
				456774.165,
				456774.143,
				456774.125,
				456774.101,
				456774.084,
				456774.061,
				456774.043,
				456774.024,
				456774.004,
				456773.980,
				456773.962,
				456773.939,
				456773.919,
				456773.902,
				456773.879,
				456773.858,
				456773.835,
				456773.815,
				456773.793,
				456773.778,
				456773.759,
				456773.734,
				456773.712,
				456773.690,
				456773.670,
				456773.648,
				456773.629,
				456773.611,
				456773.586,
				456773.565,
				456773.544,
				456773.527,
				456773.505,
				456773.485,
				456773.463,
				456773.442,
				456773.421,
				456773.399,
				456773.378,
				456773.357,
				456773.338,
				456773.280,
				456773.226,
				456773.167,
				456773.106,
				456773.046,
				456772.984,
				456772.917,
				456772.857,
				456772.664,
				456772.469,
				456772.256,
				456772.050,
				456771.792,
				456771.517,
				456771.377,
				456771.177,
				456770.972,
				456770.763,
				456770.508,
				456770.332,
				456770.134,
				456769.923,
				456769.711,
				456769.501,
				456769.286)

df = pd.DataFrame({
    "freq3": frequencies3
})

df.to_csv("frequencies3.csv", index=False)

times = (15,
		 60*1,
		 60*1+59,
		 60*2+30,
		 60*3+00,
		 60*3+30,
		 60*3+51,
		 60*4+00,
		 60*4+30,
		 60*5+00,
		 60*5+31,
		 60*6+00,
		 60*6+30,
		 60*7+00,
		 60*7+30,
		 60*8+2,
		 60*8+31,
		 60*9+00,
		 60*9+30,
		 60*10+00,
		 60*10+30,
		 60*11+7,
		 60*11+30,
		 60*12+00,
		 60*13+00,
		 60*13+35,
		 60*14+30,
		 60*15+00,
		 60*15+30,
		 60*16+00,
		 60*16+37,
		 60*17+00,
		 60*17+32,
		 60*18+00,
		 60*18+30,
		 60*19+00,
		 60*19+30,
		 60*20+00,
		 60*20+30,
		 60*22+38,
		 60*28+48,
		 60*30+00,
		 60*36+00)

frequencies4 = times#()

df = pd.DataFrame({
    "times": times,
	"freq4": frequencies4
})

df.to_csv("times.csv", index=False)
"""

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

import glob

first = 22

if first == 0:
	folder = "SilverSpecFirst/"
elif first == 1:
	folder = "SilverSpecSecond/"
elif first == 2:
	folder = "VoltageTime/"
elif first == 3:
	folder = "TEEMP/"
elif first == 4:
	folder = "SilverSpecThird/"
elif first == 5:
	folder = "WeakProbeFirst/"
elif first == 6:
	folder = "SILVERRWPQ/M1/"
elif first == 7:
	folder = "SILVERRWPQ/M2/"
elif first == 8:
	folder = "SILVERRWPQ/M3/"
elif first == 9:
	folder = "SILVERRWPQ/M4/"
elif first == 10:
	folder = "SILVERRWPQ/M5/"
elif first == 11:
	folder = "SILVERRWPQ/M6/"
elif first == 12:
	folder = "SILVERRWPQ/M7/"
elif first == 13:
	folder = "SILVERRWPQ/M8/"
elif first == 14:
	folder = "SILVERWEAKPROBENEW/M1/"
elif first == 15:
	folder = "SILVERWEAKPROBENEW/M2/"
elif first == 16:
	folder = "SILVERWEAKPROBENEW/M3/"
elif first == 17:
	folder = "SILVERWEAKPROBENEW/M4/"
elif first == 18:
	folder = "SILVERWEAKPROBENEW/M5/"
elif first == 19:
	folder = "SILVERWEAKPROBENEW/M6/"
elif first == 20:
	folder = "SILVERWEAKPROBENEW/M7/"
elif first == 21:
	folder = "SILVERWEAKPROBENEW/M8/"
elif first == 22:
	folder = "SILVERWEAKPROBENEW/M9/"

base_path = "Photodiode_Data/" + folder

files = sorted(glob.glob(base_path + "tek*ALL.csv"))

print("Found files:", len(files))
#print(files)

power = 1.2e-6  # 1.2 µW in W

averages1 = []
averages2 = []
background1 = 0.0
background2 = 0.0

# ----------------------------------------------------
# Load all datasets
# ----------------------------------------------------
for i in range(0, len(files)):

	print(i)
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

if first < 5:

	import pandas as pd
	frequencies = pd.read_csv("frequencies1.csv")
	frequencies2 = pd.read_csv("frequencies2.csv")
	frequencies3 = pd.read_csv("frequencies3.csv")
	frequencies4 = pd.read_csv("times.csv")["freq4"]
	times = np.array(pd.read_csv("times.csv")["times"])/60

	xs1 = np.linspace(1, 4, len(files)-1)
	xs2 = np.linspace(0, len(files)-2, len(files)-1)

	print(times)

	if first == 0:
		xs = -np.array(frequencies)*2 + (c / (328.1629601))# - 633
	elif first == 1:
		xs = -np.array(frequencies2)*2 + (c / (328.1629601))# - 633
	elif first == 2:
		xs = xs2
	elif first == 3:
		xs = times
	elif first == 4:
		xs = -np.array(frequencies3)*2 + (c / (328.1629601))# - 633

y1 = np.abs(np.abs(averages1_means) - np.abs(background1_mean))
y2 = np.abs(np.abs(averages2_means) - np.abs(background2_mean))

y1_err = np.sqrt(averages1_errs**2 + background1_err**2)# + )
y2_err = np.sqrt(averages2_errs**2 + background2_err**2)# + )

powers = ((238.1-0.179),
		  (522-0.237),
		  (119.8-0.231),
		  (26.01-0.232),
		  (1.273-0.225))

powers1 = ((238.1-0.179),
		   (238.1-0.179),
		  (522-0.237),
		  (522-0.237),
		  (119.8-0.231),
		  (119.8-0.231),
		  (26.01-0.232),
		  (26.01-0.232),
		  (1.273-0.225),
		  (1.273-0.225))

angle_unc = 0.165/100

y2_err = np.sqrt(y2_err**2 + (angle_unc * y2)**2)

# --------------------------------------------------------
# Add % uncertainty due to beam power fluctuations
# --------------------------------------------------------
power_frac = 0.00   # 0 percent

y1_err = np.sqrt(y1_err**2 + (power_frac * y1)**2)
y2_err = np.sqrt(y2_err**2 + (power_frac * y2)**2)

#print(y1,y2)

if first < 5:

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

	"""
	print(y1)

	if first == 2:
		plt.errorbar(xs2, y1, yerr = y1_err, marker="o", label="CH1 data")
		plt.xlabel("Time (Mins)")
		plt.ylabel("CH1 Signal (V)")
	else:
		plt.plot(xs1, fit_line1, '--', label=f"CH1 fit\n y = {m1:.3g}x + {c1:.3g}")
		plt.errorbar(xs1, y1, yerr = y1_err, marker="o", label="CH1 data")
		plt.errorbar(xs1, y2, yerr = y2_err, marker="o", label="CH2 data")
		plt.legend()
		plt.xlabel("Voltage (GHz)")
		plt.ylabel("Signal (V)")

	if first == 0:
		plt.savefig("Photodiode_Plot", dpi=300, bbox_inches='tight')
	elif first == 1:
		plt.savefig("Photodiode_Plot2", dpi=300, bbox_inches='tight')
	plt.show()
	"""

# ----------------------------------------------------
# Transmission and uncertainty propagation
# ----------------------------------------------------
transmission = y1 / y2

transmission_err = np.abs(transmission) * np.sqrt(
    (y1_err / y1)**2 +
    (y2_err / y2)**2
)

#plt.errorbar( powers1, transmission, transmission_err, marker = ".", linestyle = "", label = "wf{}, Ch1".format(i))
#plt.plot( powers1, y2, marker = ".", linestyle = "", label = "wf{}, Ch2".format(i))

#plt.legend()
#plt.show()

######## save to csv

if first == 0:
	df = pd.DataFrame({
		"Transmission1": transmission,
		"Transmission1err": transmission_err,
	})
	df.to_csv("transmission1.csv", index=False)
elif first == 1:
	df = pd.DataFrame({
		"Transmission2": transmission,
		"Transmission2err": transmission_err,
	})
	df.to_csv("transmission2.csv", index=False)
elif first == 4:
	df = pd.DataFrame({
	"Transmission3": transmission,
	"Transmission3err": transmission_err,
	})
	df.to_csv("transmission3.csv", index=False)
elif first == 5:
	df = pd.DataFrame({
	"Transmission": transmission,
	"Transmissionerr": transmission_err,
	})
	df.to_csv("WeakProbeTransmissions.csv", index=False)
elif first >= 6:
	df = pd.DataFrame({
	"Transmission": transmission,
	"Transmissionerr": transmission_err,
	})
	df.to_csv("WeakProbeTransmissions{}.csv".format(first-4), index=False)
######## save to csv

print("saved to csv")

print(len(transmission))#,len(xs))

if first == 1 or first == 4:
	print(np.max(transmission))
	print(np.min(transmission))
	plt.errorbar(xs, transmission/np.max(transmission),
				yerr=np.abs(transmission_err/np.max(transmission)),
				fmt='.')
	
	plt.ylim(0,1.1)
	plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	plt.xlabel("Detuning (GHz)")
	plt.ylabel("Transmission")
elif first == 3:
	print(np.max(transmission))
	print(np.min(transmission))
	plt.errorbar(xs, transmission/np.max(transmission),
				yerr=np.abs(transmission_err/np.max(transmission)),
				marker='o')
	
	plt.ylim(0,1.1)
	plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	plt.xlabel("Time (Minutes)")
	plt.ylabel("Transmission")
elif first >= 5:
	print(np.max(transmission))
	print(np.min(transmission))
	#plt.errorbar(xs, transmission/np.max(transmission),yerr=np.abs(transmission_err/np.max(transmission)),marker='o')
	
	#plt.ylim(0,1.1)
	#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	#plt.xlabel("Time (Minutes)")
	#plt.ylabel("Transmission")
else:
	transmission = transmission/0.3301348605312241
	transmission_err = transmission_err/0.3301348605312241

	plt.errorbar(xs, transmission - np.min(transmission) + (0.08973872980696299/0.3301348605312241),
				yerr=np.abs(transmission_err),
				marker='o')
	#plt.ylim(0,1.1)
	#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
	plt.xlabel("Time (Minutes)")
	plt.ylabel("On resonance Transmission")


if first < 5:
	plt.tight_layout()

	if first == 0:
		plt.savefig("Photodiode_Transmission", dpi=300, bbox_inches='tight')
	elif first == 1:
		plt.savefig("Photodiode_Transmission2", dpi=300, bbox_inches='tight')
	elif first == 2:
		plt.savefig("TransmissionTime", dpi=300, bbox_inches='tight')

	if first < 2:
		plt.ylim([0, 1.1])
		plt.xlim([-8.5,8.5])

	plt.show()