import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os
from scipy.integrate import cumulative_trapezoid as cumtrapz
from matplotlib import rcParams

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

mode = 2

if mode >= 2:

	import json

	# Toggle this to control calibration behaviour
	saveNewScaleFactorMean = False     # True  → recompute and save new k_mean
							# False → load existing k_mean from file

	kmean_file = "k_mean_calibration.json"

	def save_kmean(k_mean):
		with open(kmean_file, "w") as f:
			json.dump({"k_mean": k_mean}, f)

	def load_kmean():
		if not os.path.exists(kmean_file):
			return None
		with open(kmean_file, "r") as f:
			return json.load(f)["k_mean"]

	# Determine starting k_mean based on saveNewMean flag
	if saveNewScaleFactorMean:
		# We want to generate a new calibration
		k_mean_global = None
	else:
		# Use existing calibration
		k_mean_global = load_kmean()
		if k_mean_global is None:
			raise RuntimeError(
				"saveNewScaleFactorMean=False but no k_mean_calibration.json exists."
			)

else:
	# In mode 0/1 always do calibration
	k_mean_global = None

h = 6.62607015e-34  # Planck's constant (J·s)
c = 2.99792458e8    # speed of light (m/s)
wavelength = 328.1629601e-9 #wavelength of light
wavelength_error = 0.0000022e-9 
gamma_nat = 1.472e8
gamma_nat_error = 0.007e8
tau = 6.79e-9 #error 0.03e-9
tau_error = 0.03e-9 

I_sat = 867 #(np.pi * h * c)/(3 * tau * wavelength**3) #867(4)
I_sat_error = 4

print(I_sat)

fontsz = 16
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'

focus_distance = None # Only show a certain distance
plot_main = False
#save_all_plots = True
save_all_plots = False

pixel_size = 3.45e-6 #m
pixel_area = pixel_size**2 #3.45 x 3.45 micrometers squared
photon_energy = h * c / wavelength

p_total = 1.02e-6#1.21e-6
p_total_error = 0.05e-6#0.04e-6
print("TOTAL MEASURED POWER = " + str(p_total) + "~+-~" + str(p_total_error) + "W")

if mode == 0:
	beam_images = {
		0:   {"centre": (748, 532), "exposure": None},
		25:   {"centre": (747, 523), "exposure": None},
		50:  {"centre": (751, 588), "exposure": None},
		75:  {"centre": (751, 549), "exposure": None},    
		100: {"centre": (751, 541), "exposure": None},
		125:   {"centre": (750, 556), "exposure": None},
		150:  {"centre": (750, 522), "exposure": None},
		175:  {"centre": (748, 534), "exposure": None}, 
		200: {"centre": (748, 493), "exposure": None},
		225:   {"centre": (748, 459), "exposure": None},
		250:  {"centre": (748, 517), "exposure": None},
		275:  {"centre": (749, 546), "exposure": None}, 
		300: {"centre": (751, 534), "exposure": None},
		325:   {"centre": (753, 521), "exposure": None},
		350:  {"centre": (756, 465), "exposure": None},
		375:  {"centre": (761, 501), "exposure": None}, 
		400: {"centre": (768, 477), "exposure": None},
		425: {"centre": (778, 519), "exposure": None},
		450: {"centre": (790, 585), "exposure": None},
		475: {"centre": (805, 525), "exposure": None},
	}
	base_path = "Beam_Images_Old/"
	end_path = "_0_12_097ms.bmp"
elif mode == 1:
	beam_images = {
		0:   {"centre": (725, 570), "exposure": 8.488e-3},
		25:   {"centre": (726, 569), "exposure": 8.488e-3},
		50:  {"centre": (733, 568), "exposure": 8.488e-3},
		75:  {"centre": (844, 566), "exposure": 8.488e-3},    
		100: {"centre": (854, 561), "exposure": 8.488e-3},
		125:   {"centre": (790, 557), "exposure": 8.488e-3},
		150:  {"centre": (802, 555), "exposure": 8.488e-3},
		175:  {"centre": (703, 550), "exposure": 8.488e-3}, 
		200: {"centre": (772, 547), "exposure": 8.488e-3},
		225:   {"centre": (789, 544), "exposure": 8.488e-3},
		250:  {"centre": (768, 541), "exposure": 8.488e-3},
		275:  {"centre": (782, 539), "exposure": 8.488e-3}, 
		300: {"centre": (792, 532), "exposure": 8.488e-3},
		325:   {"centre": (785, 533), "exposure": 8.488e-3},
		350:  {"centre": (769, 533), "exposure": 8.488e-3},
		375:  {"centre": (803, 534), "exposure": 8.488e-3}, 
		400: {"centre": (799, 538), "exposure": 8.488e-3},
		425: {"centre": (712, 544), "exposure": 8.488e-3},
		450: {"centre": (720, 552), "exposure": 8.488e-3},
		475: {"centre": (643, 563), "exposure": 8.488e-3},
		500: {"centre": (799, 578), "exposure": 8.488e-3},
		525: {"centre": (724, 594), "exposure": 8.488e-3},
		550: {"centre": (709, 610), "exposure": 8.488e-3},
	}
	base_path = "Camera_images_new/"
	end_path = "_1_8.488.bmp"
else:
	beam_images = {
		0:   {"centre": (740, 543), "exposure": 8.488e-3},
		50:   {"centre": (740, 543), "exposure": 8.488e-3},
		100:  {"centre": (740, 543), "exposure": 8.488e-3},
		150:  {"centre": (740, 543), "exposure": 8.488e-3},    
		200: {"centre": (740, 543), "exposure": 8.488e-3},
		250:   {"centre": (740, 543), "exposure": 8.488e-3},
		300:  {"centre": (740, 543), "exposure": 8.488e-3},
		350:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		400: {"centre": (740, 543), "exposure": 8.488e-3},
		450:   {"centre": (740, 543), "exposure": 8.488e-3},
		500:  {"centre": (740, 543), "exposure": 8.488e-3},
		550:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		600: {"centre": (740, 543), "exposure": 8.488e-3},
		650:   {"centre": (740, 543), "exposure": 8.488e-3},
		700:  {"centre": (740, 543), "exposure": 8.488e-3},
		750:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		800: {"centre": (740, 543), "exposure": 8.488e-3},
		850: {"centre": (740, 543), "exposure": 8.488e-3},
		900: {"centre": (740, 543), "exposure": 8.488e-3},
		950: {"centre": (740, 543), "exposure": 8.488e-3},
		1000: {"centre": (740, 543), "exposure": 8.488e-3},
		1050:   {"centre": (740, 543), "exposure": 8.488e-3},
		1100:  {"centre": (740, 543), "exposure": 8.488e-3},
		1150:  {"centre": (740, 543), "exposure": 8.488e-3},    
		1200: {"centre": (740, 543), "exposure": 8.488e-3},
		1250:   {"centre": (740, 543), "exposure": 8.488e-3},
		1300:  {"centre": (740, 543), "exposure": 8.488e-3},
		1350:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		1400: {"centre": (740, 543), "exposure": 8.488e-3},
		1450:   {"centre": (740, 543), "exposure": 8.488e-3},
		1500:  {"centre": (740, 543), "exposure": 8.488e-3},
		1550:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		1600: {"centre": (740, 543), "exposure": 8.488e-3},
		1650:   {"centre": (740, 543), "exposure": 8.488e-3},
		1700:  {"centre": (740, 543), "exposure": 8.488e-3},
		1750:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		1800: {"centre": (740, 543), "exposure": 8.488e-3},
		1850: {"centre": (740, 543), "exposure": 8.488e-3},
		1900: {"centre": (740, 543), "exposure": 8.488e-3},
		1950: {"centre": (740, 543), "exposure": 8.488e-3},
		2000: {"centre": (740, 543), "exposure": 8.488e-3},
		2050:   {"centre": (740, 543), "exposure": 8.488e-3},
		2100:  {"centre": (740, 543), "exposure": 8.488e-3},
		2150:  {"centre": (740, 543), "exposure": 8.488e-3},    
		2200: {"centre": (740, 543), "exposure": 8.488e-3},
		2250:   {"centre": (740, 543), "exposure": 8.488e-3},
		2300:  {"centre": (740, 543), "exposure": 8.488e-3},
		2350:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		2400: {"centre": (740, 543), "exposure": 8.488e-3},
		2450:   {"centre": (740, 543), "exposure": 8.488e-3},
		2500:  {"centre": (740, 543), "exposure": 8.488e-3},
		2550:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		2600: {"centre": (740, 543), "exposure": 8.488e-3},
		2650:   {"centre": (740, 543), "exposure": 8.488e-3},
		2700:  {"centre": (740, 543), "exposure": 8.488e-3},
		2750:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		2800: {"centre": (740, 543), "exposure": 8.488e-3},
		2850: {"centre": (740, 543), "exposure": 8.488e-3},
		2900: {"centre": (740, 543), "exposure": 8.488e-3},
		2950: {"centre": (740, 543), "exposure": 8.488e-3},
		3000: {"centre": (740, 543), "exposure": 8.488e-3},
		3050:   {"centre": (740, 543), "exposure": 8.488e-3},
		3100:  {"centre": (740, 543), "exposure": 8.488e-3},
		3150:  {"centre": (740, 543), "exposure": 8.488e-3},    
		3200: {"centre": (740, 543), "exposure": 8.488e-3},
		3250:   {"centre": (740, 543), "exposure": 8.488e-3},
		3300:  {"centre": (740, 543), "exposure": 8.488e-3},
		3350:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		3400: {"centre": (740, 543), "exposure": 8.488e-3},
		3450:   {"centre": (740, 543), "exposure": 8.488e-3},
		3500:  {"centre": (740, 543), "exposure": 8.488e-3},
		3550:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		3600: {"centre": (740, 543), "exposure": 8.488e-3},
		3650:   {"centre": (740, 543), "exposure": 8.488e-3},
		3700:  {"centre": (740, 543), "exposure": 8.488e-3},
		3750:  {"centre": (740, 543), "exposure": 8.488e-3}, 
		3800: {"centre": (740, 543), "exposure": 8.488e-3},
		3850: {"centre": (740, 543), "exposure": 8.488e-3},
		3900: {"centre": (740, 543), "exposure": 8.488e-3},
		3950: {"centre": (740, 543), "exposure": 8.488e-3},
		4000: {"centre": (740, 543), "exposure": 8.488e-3},
		4050: {"centre": (740, 543), "exposure": 8.488e-3},
	}
	base_path = "Camera_Spec_Voltage/"
	end_path = "mV_8.488_450.bmp"

default_exposure = 12.097e-3  # s
exposure_error = 0.001
allNormal = False

def to3string(dist: int):
	"""Converts integers to 3 digit strings, i.e. 25 -> 025"""
	return str(dist).zfill(3)

def to4string(dist: int):
	"""Converts integers to 3 digit strings, i.e. 25 -> 025"""
	return str(dist).zfill(4)

def round_sig(x, sig=3):
	"""Round a number to a given number of significant figures."""
	if x == 0:
		return 0
	return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def process_image(distance, centre=None, exposure=None, normalise=False, input_scale_factor = None):
	"""Process a single beam image and return all derived quantities."""
	#path = f"{base_path}{to3string(distance)}_0_12_097ms.bmp"
	if mode <= 1:
		path = f"{base_path}{to3string(distance)}" + end_path
	else:
		path = f"{base_path}{to4string(distance)}" + end_path
	img = plt.imread(path)
	if img.ndim == 3:
		img = img.mean(axis=2)
	ny, nx = img.shape

	# exposure handling
	if exposure is None:
		exposure = default_exposure

	if mode >= 2:#subtract lamp light
		lampImage = plt.imread("Spec_Voltage_Norm_Images/2.1_Laser_Off_8.488.bmp")
		if lampImage.ndim == 3:
			lampImage = lampImage.mean(axis=2)
		img = img - lampImage

	img = img / (exposure * 255) #gives unscaled intensity values to each pixel

	sf = 1

	#print(np.sum(img))

	if normalise:
		img = img / img.max()
	# beam centre
	if centre is not None:
		cx, cy = centre
	else:
		cy, cx = center_of_mass(img)

	# coordinate grid
	corners = np.array([
		[0 - cx,     0 - cy],
		[nx-1 - cx,  0 - cy],
		[0 - cx,     ny-1 - cy],
		[nx-1 - cx,  ny-1 - cy],
	])
	r_max = np.sqrt((corners**2).sum(axis=1)).max()

	# polar grid high-res
	nr = int(np.ceil(r_max))
	nt = int(np.ceil(2 * np.pi * r_max))
	nt = min(nt, 6000)
	r = np.linspace(0, r_max, nr) #convert pixel lengths to real lengths in m
	theta = np.linspace(-np.pi, np.pi, nt) #theta in radians
	r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")
	x_p = r_grid * np.cos(theta_grid) + cx
	y_p = r_grid * np.sin(theta_grid) + cy
	polar_img = map_coordinates(img, [y_p, x_p], order=1)

	# integration
	P_r_unnorm = np.trapezoid(polar_img, theta, axis=1)#gives the power per unit radial length
	P_total = np.trapezoid(P_r_unnorm * r, r)#Integrates over r to get the power, need to apply a scale factor so it equals the total measured power
	P_encircled = cumtrapz(P_r_unnorm * r, r, initial=0)
	#print(distance, P_total)
	scale_factor = 0
	if mode <= 1:
		scale_factor = p_total / (P_total) #scale factor for just beam plots
	elif input_scale_factor is not None:
		scale_factor = input_scale_factor #input scale factor
	else:
		scale_factor = p_total/P_total #default
	#print(scale_factor)

	#Total measured power is known

	profile_x = r * pixel_size #radial size in m
	profile_y = (P_r_unnorm * (scale_factor / pixel_area)) / (np.pi * 2) # average intensity per radius
	r_safe = r.copy()
	r_safe[0] = r_safe[1]
	I_avg_area = P_encircled / (np.pi * r_safe**2)  # avoid divide-by-zero at r=0
	I_avg_area[0] = 0
	I_avg_area_scaled = I_avg_area * (scale_factor / pixel_area)

	I_Peak = np.max(profile_y) #peak intensity of radial average intensity distribution
	I_Ave_Peak = np.max(I_avg_area_scaled)
	profile_label = "I(r)"
	polar_extent = (theta.min(), theta.max(), r.min(), r.max())
	polar_xlabel = "θ (radians)"
	polar_ylabel = "r (pixels)"

	return img, polar_img, profile_x, profile_y, \
		P_total, (cx, cy), polar_extent, \
		polar_xlabel, polar_ylabel, profile_label, \
		I_Peak, I_avg_area_scaled, I_Ave_Peak, scale_factor


def detect_pre_absorption_scale_factor(results, rel_tol=0.02, baseline_count=10):
	"""
	Detects the 'flat' region of scale factors (no absorption),
	averages them, and returns indices/values to use.

	Parameters
	----------
	results : dict
		Dict keyed by distance/voltage; must contain "scale_factor".
	rel_tol : float
		Allowed relative deviation from baseline (e.g. 0.02 = 2%).
	baseline_count : int
		Number of initial points to use to estimate baseline.

	Returns
	-------
	clean_indices : list[int]
		Indices (in sorted order) of the pre-absorption region.
	clean_values : np.ndarray
		Scale factors in that region.
	k_mean : float
		Average scale factor over the pre-absorption region.
	distances_sorted : list
		Distances/voltages in sorted order (for mapping indices back).
	all_scale_factors : np.ndarray
		All scale factors in sorted order.
	"""
	distances_sorted = sorted(results.keys())
	all_scale_factors = np.array([results[d]["scale_factor"] for d in distances_sorted])

	if len(all_scale_factors) <= baseline_count:
		# Not enough points to detect absorption – just use all
		clean_indices = list(range(len(all_scale_factors)))
		clean_values = all_scale_factors.copy()
		k_mean = clean_values.mean()
		return clean_indices, clean_values, k_mean, distances_sorted, all_scale_factors

	# Baseline from the first 'baseline_count' points
	baseline_vals = all_scale_factors[:baseline_count]
	baseline = np.median(baseline_vals)

	rel_dev = np.abs(all_scale_factors - baseline) / baseline

	# Find first index where deviation exceeds tolerance
	# Everything BEFORE that is considered non-absorbing
	cutoff_idx = len(all_scale_factors)  # default: use all if nothing deviates
	for i in range(baseline_count, len(all_scale_factors)):
		if rel_dev[i] > rel_tol:
			cutoff_idx = i
			break

	clean_indices = list(range(cutoff_idx))
	clean_values = all_scale_factors[clean_indices]
	k_mean = clean_values.mean()

	return clean_indices, clean_values, k_mean, distances_sorted, all_scale_factors

# --- Process all images ---
results = {}
for d, info in beam_images.items():

	img, polar_img, x_prof, y_prof, P, centre, \
		polar_extent, polar_xlabel, polar_ylabel, \
		profile_label, I_max, I_ave_profile, I_ave_peak, scale_factor = process_image(
		d,
		centre=info.get("centre"),
		exposure = info.get("exposure") or default_exposure,
		normalise=allNormal, input_scale_factor=k_mean_global
	)

	results[d] = {
		"img": img,
		"polar_img": polar_img,
		"x_prof": x_prof,
		"y_prof": y_prof,
		"P_total": P,
		"scale_factor": scale_factor,
		"centre": centre,
		"polar_extent": polar_extent,
		"polar_xlabel": polar_xlabel,
		"polar_ylabel": polar_ylabel,
		"profile_label": profile_label,
		"I_max": I_max,
		"I_Ave_profile" : I_ave_profile,
		"I_Ave_max": I_ave_peak,
	}

if saveNewScaleFactorMean:
	clean_indices, clean_values, k_mean_global, distances_sorted, all_k = \
		detect_pre_absorption_scale_factor(results)
	
	print("New calibration k_mean =", k_mean_global)
	save_kmean(k_mean_global)

else:
	print("Using previously saved calibration k_mean =", k_mean_global)


if plot_main:
	# --- Decide what to plot ---
	if focus_distance is not None:
		# Only plot the chosen distance
		distances_to_plot = [focus_distance]
	else:
		# Plot all
		distances_to_plot = list(results.keys())

	n = len(distances_to_plot)
	fig, axs = plt.subplots(n, 3, figsize=(12, 3.2 * n))

	if n == 1:
		axs = np.expand_dims(axs, 0)

	#fig.suptitle(f"Beam analysis — integration: {integration_order.replace('_', ' ')}", fontsize=14)

	for i, d in enumerate(distances_to_plot):
		data = results[d]
		img = data["img"]
		polar_img = data["polar_img"]
		cx, cy = data["centre"]
		x_prof, y_prof = data["x_prof"], data["y_prof"]
		polar_extent = data["polar_extent"]
		polar_xlabel = data["polar_xlabel"]
		polar_ylabel = data["polar_ylabel"]
		profile_label = data["profile_label"]
		I_max = data["I_max"]
		I_ave_peak = data["I_Ave_max"]

		# --- Column 1: Original image ---
		axs[i, 0].imshow(img, cmap="CMRmap", origin="lower")
		axs[i, 0].plot(cx, cy, "bo", markersize=3, alpha=0.6)
		axs[i, 0].set_ylabel("y (pixels)")
		if i == 0:
			axs[i, 0].set_title("Original")
		if i < n - 1:
			axs[i, 0].set_xlabel("")
			axs[i, 0].set_xticklabels([])
		else:
			axs[i, 0].set_xlabel("x (pixels)")

		# --- Column 2: Polar-transformed image ---
		axs[i, 1].imshow(polar_img, extent=polar_extent, aspect="auto", cmap="CMRmap", origin="lower")
		axs[i, 1].set_ylabel(polar_ylabel)
		if i == 0:
			axs[i, 1].set_title("Polar coordinates")
		if i < n - 1:
			axs[i, 1].set_xlabel("")
			axs[i, 1].set_xticklabels([])
		else:
			axs[i, 1].set_xlabel(polar_xlabel)

		# --- Column 3: Integrated profile ---
		axs[i, 2].plot(x_prof * 1e3, y_prof / I_sat, label=r"$I(r)$ / $I_{sat}$", zorder = 1)
		axs[i, 2].plot(x_prof * 1e3, data["I_Ave_profile"] / I_sat, '--', label=r"$I_{avg}(r)$ / $I_{sat}$", zorder = 0)
		axs[i, 2].legend(loc="upper right", fontsize=8)

		axs[i, 2].set_ylabel(r"$I$ / $I_{sat}$")
		I_round = round_sig(I_max, 4)
		I_round2 = round_sig(I_ave_peak, 4)

		axs[i, 2].text(0.98,0.8,
			r"Peak $I(r) = $" + f"{I_round}" + " ($W~m^{-2}$)",
			ha='right', va='top',
			transform=axs[i, 2].transAxes,
			fontsize=10,)
			#bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3'))

		axs[i, 2].text(0.98,0.7,
			r"Peak $I_{avg} = $" + f"{I_round2}" + " ($W~m^{-2}$)",
			ha='right', va='top',
			transform=axs[i, 2].transAxes,
			fontsize=10,)

		if i == 0:
			axs[i, 2].set_title(f"{profile_label} profile")
		if i < n - 1:
			axs[i, 2].set_xlabel("")
			axs[i, 2].set_xticklabels([])
		else:
			axs[i, 2].set_xlabel("r (mm)")

		axs[i, 2].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

		# Label each row on the left with its distance
		axs[i, 0].text(-0.25, 0.5, f"{d} mm", transform=axs[i, 0].transAxes,
					rotation=90, va="center", ha="right", fontsize=10)

	plt.subplots_adjust(hspace=0.1, wspace=0.25)

	if focus_distance != None and save_all_plots:
		if mode == 0:
			plt.savefig(to3string(focus_distance)+"_Plots", dpi=300, bbox_inches='tight')
		elif mode == 1:
			plt.savefig(to3string(focus_distance)+"_Plots_new", dpi=300, bbox_inches='tight')
		else:
			plt.savefig(to3string(focus_distance)+"_Plots_new_spec", dpi=300, bbox_inches='tight')
		

	plt.show()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D plotting)

# --- Conditional summary plots ---
if focus_distance is None:

	# --- Create a heatmap of all I(r) profiles (radius vs distance) ---

	# Extract and align all I(r) profiles
	distances = sorted(results.keys())
	r_values = results[distances[0]]["x_prof"]

	# Interpolate all profiles to a common r grid (in case lengths differ)
	r_common = np.linspace(0, max(r_values), 500)
	I_profiles = []

	for d in distances:
		x_prof = results[d]["x_prof"]
		y_prof = results[d]["y_prof"]
		I_interp = np.interp(r_common, x_prof, y_prof)/I_sat
		I_profiles.append(I_interp)

	I_profiles = np.array(I_profiles)  # shape = (num_distances, num_r_points)

	# Transpose so that radius is vertical (y-axis)
	I_profiles = I_profiles.T  # shape = (num_r_points, num_distances)

	# Create the heatmap
	plt.figure(figsize=(10, 6))
	extent = [distances[0], distances[-1], r_common[0]*1e3, r_common[-1]*1e3]  # x=distance (mm), y=radius (mm)
	plt.imshow(
		I_profiles,
		extent=extent,
		aspect='auto',
		origin='lower',
		cmap='CMRmap',
		interpolation='bilinear'
	)
	cbar = plt.colorbar(label=r"$I(r)$ / $I_{sat}$")
	cbar.formatter.set_powerlimits((-3, 3))
	cbar.formatter.set_useMathText(True)
	cbar.update_ticks()
	plt.xlabel("Distance (mm)")
	plt.ylabel("Radius (mm)")
	plt.title("Radial Intensity Profiles Heatmap")

	ax = plt.gca()
	if mode == 2:
		ax.xaxis.set_minor_locator(AutoMinorLocator(10))
	else:
		ax.xaxis.set_minor_locator(AutoMinorLocator(4))
	
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))

	plt.tight_layout()

	plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))


	if save_all_plots:
		if mode == 0:
			plt.savefig("I_r_heatmap.png", dpi=300, bbox_inches='tight')
		elif mode == 1:
			plt.savefig("I_r_heatmap_new.png", dpi=300, bbox_inches='tight')
		else:
			plt.savefig("I_r_heatmap_new_spec.png", dpi=300, bbox_inches='tight')

	plt.show()

	# --- Create a heatmap of all ⟨I⟩(r) profiles (encircled average intensity vs distance) ---

	# Interpolate all ⟨I⟩(r) profiles to the same r grid
	I_Ave_profiles = []

	for d in distances:
		x_prof = results[d]["x_prof"]
		y_ave = results[d]["I_Ave_profile"]
		I_Ave_interp = np.interp(r_common, x_prof, y_ave)/I_sat
		I_Ave_profiles.append(I_Ave_interp)

	I_Ave_profiles = np.array(I_Ave_profiles)  # shape = (num_distances, num_r_points)
	I_Ave_profiles = I_Ave_profiles.T  # transpose so radius is on y-axis

	# Create the heatmap
	plt.figure(figsize=(10, 6))
	extent = [distances[0], distances[-1], r_common[0]*1e3, r_common[-1]*1e3]  # x=distance (mm), y=radius (mm)
	plt.imshow(
		I_Ave_profiles,
		extent=extent,
		aspect='auto',
		origin='lower',
		cmap='CMRmap',
		interpolation='bilinear'
	)
	cbar = plt.colorbar(label=r"$I_{avg}(r)$ / $I_{sat}$")
	cbar.formatter.set_powerlimits((-3, 3))
	cbar.formatter.set_useMathText(True)
	cbar.update_ticks()
	plt.xlabel("Distance (mm)")
	plt.ylabel("Radius (mm)")
	plt.title("Encircled-Average Intensity Profiles Heatmap")

	ax = plt.gca()
	if mode == 2:
		ax.xaxis.set_minor_locator(AutoMinorLocator(10))
	else:
		ax.xaxis.set_minor_locator(AutoMinorLocator(4))
	
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))

	plt.tight_layout()

	plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

	if save_all_plots:
		if mode == 0:
			plt.savefig("I_Ave_r_heatmap.png", dpi=300, bbox_inches='tight')
		elif mode == 1:
			plt.savefig("I_Ave_r_heatmap_new.png", dpi=300, bbox_inches='tight')
		else:
			plt.savefig("I_Ave_r_heatmap_new_spec.png", dpi=300, bbox_inches='tight')

	plt.show()

	distances = list(results.keys())
	I_max_values = np.array([results[d]["I_max"] for d in distances]) / I_sat
	I_Ave_max_values = np.array([results[d]["I_Ave_max"] for d in distances]) / I_sat

	plt.figure(figsize=(8, 5))
	plt.plot(distances, I_max_values, "o-", color="tab:red", zorder = 1, label = r"Peak $I(r)$")
	plt.plot(distances, I_Ave_max_values , "x-", color="tab:orange", zorder = 0, label = r"Peak $I_{avg}$")
	plt.xlabel("Distance from 0 point (mm)")
	plt.ylabel(r"$I_{max}$ / $I_{sat}$")
	plt.title("Peak intensity vs distance")

	ax = plt.gca()
	if mode == 2:
		ax.xaxis.set_minor_locator(AutoMinorLocator(10))
	else:
		ax.xaxis.set_minor_locator(AutoMinorLocator(4))
	
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))

	plt.tight_layout()
	plt.legend(loc="lower right")

	plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

	if save_all_plots:
		if mode == 0:
			plt.savefig("I_max_distance_graph", dpi=300, bbox_inches='tight')
		elif mode == 1:
			plt.savefig("I_max_distance_graph_new", dpi=300, bbox_inches='tight')
		else:
			plt.savefig("I_max_distance_graph_new_spec", dpi=300, bbox_inches='tight')
	
	plt.show()

else:
	print(f"Displayed only {focus_distance} mm (single-image mode).")



# Collect all unscaled total powers
unscaled_powers = np.array([results[d]["P_total"] for d in results])

# Compute statistics
mean_unscaled = np.mean(unscaled_powers)
std_unscaled = np.std(unscaled_powers, ddof=1)  # sample standard deviation
sem_unscaled = std_unscaled / np.sqrt(len(unscaled_powers))  # standard error of the mean

print(f"Mean unscaled = {mean_unscaled:.3e} W")
print(f"Standard deviation = {std_unscaled:.3e} W")
print(f"Standard error = {sem_unscaled:.3e} W")

# --- Propagate all known uncertainties ---

# (1) Scale factor from calibration
scale_factor = p_total / mean_unscaled
u_scale = scale_factor * np.sqrt(
	(sem_unscaled / mean_unscaled) ** 2 + (p_total_error / p_total) ** 2
)

print(f"\nScale factor k = {scale_factor:.6g} ± {u_scale:.6g}  (rel {u_scale/scale_factor:.3%})")

# (2) Relative uncertainties
rel_scale_unc = u_scale / scale_factor         # from calibration
rel_Isat_unc = I_sat_error / I_sat             # from saturation intensity (tau etc.)

# (3) Combine them for I/I_sat
rel_I_div_Isat_unc = np.sqrt(rel_scale_unc**2 + rel_Isat_unc**2)

print(f"Relative uncertainty in I/I_sat: {rel_I_div_Isat_unc:.3%}")

# --- Estimate detector noise from dark region of polar images ---
# Uses r = 700–900 pixels region (far outside beam centre)
# This works better for near-field images where the beam fills the frame.

print("\nEstimating per-image noise from polar dark regions...")

r_noise_min, r_noise_max = 700, 900  # radial range (pixels) for dark background

for d in results:
	polar = results[d]["polar_img"]
	nr, nt = polar.shape

	# Convert the target range to indices safely within bounds
	r_idx_min = int(max(0, min(nr - 1, r_noise_min)))
	r_idx_max = int(max(0, min(nr - 1, r_noise_max)))

	if r_idx_max <= r_idx_min:
		print(f"Warning: r range {r_noise_min}-{r_noise_max} px out of bounds for {d} mm (nr={nr}).")
		continue

	bg_region = polar[r_idx_min:r_idx_max, :].ravel()
	sigma_pix = np.std(bg_region)

	# crude de-correlation: assume ~10 pixels per independent sample
	N_eff = max(1, bg_region.size // 10)
	u_noise = (p_total / results[d]["P_total"]) * sigma_pix / pixel_area
	u_noise_reduced = u_noise / np.sqrt(N_eff)
	rel_noise = u_noise_reduced / results[d]["I_max"]
	results[d]["rel_noise"] = rel_noise

	# Optional: print summary for debugging
	print(f"{d:>4} mm → σ_pix = {sigma_pix:.2e}, rel_noise = {rel_noise:.2%}")


# --- Estimate uncertainty due to manual centre placement ---
centre_error_px = 2  # estimated uncertainty in chosen beam centre [pixels]
centre_error_m = centre_error_px * pixel_size

print(f"\nEstimating per-image uncertainty from ±{centre_error_px:.0f} px centre placement...")

for d in results:
	x_prof_m = results[d]["x_prof"]  # already in metres
	y_prof = results[d]["y_prof"]    # I(r) profile (W/m²)

	# Gradient of radial intensity profile
	dy_dr = np.gradient(y_prof, x_prof_m)
	mask = y_prof > 0.05 * np.max(y_prof)
	g_char = np.percentile(np.abs(dy_dr[mask]), 90) if np.any(mask) else 0.0
	rel_centre_unc = (g_char * centre_error_m) / np.max(y_prof)
	results[d]["rel_centre_unc"] = rel_centre_unc

	print(f"{d:>4} mm → rel_centre_unc = {rel_centre_unc:.2%}")

# (4) propagate to per-image peaks
for d in results:
	Imax = results[d]["I_max"]
	Iavg = results[d]["I_Ave_max"]

	# Include scale, Isat, and background noise uncertainties
	rel_noise = results[d].get("rel_noise", 0)
	rel_centre_unc = results[d].get("rel_centre_unc", 0)
	rel_total_unc = np.sqrt(rel_scale_unc**2 + rel_Isat_unc**2 + rel_noise**2 + rel_centre_unc**2)

	Imax_norm = Imax / I_sat
	Iavg_norm = Iavg / I_sat
	u_Imax_norm = Imax_norm * rel_total_unc
	u_Iavg_norm = Iavg_norm * rel_total_unc

	# Store for optional plotting with error bars
	results[d]["u_I_max_norm"] = u_Imax_norm
	results[d]["u_I_avg_norm"] = u_Iavg_norm

# --- Optional print summary ---
print("\nExample propagated uncertainties:")
example_d = sorted(results.keys())[0]
print(f"At distance {example_d} mm:")
print(f"  I_max/I_sat = {results[example_d]['I_max']/I_sat:.3f} ± {results[example_d]['u_I_max_norm']:.3f}")
print(f"  I_avg/I_sat = {results[example_d]['I_Ave_max']/I_sat:.3f} ± {results[example_d]['u_I_avg_norm']:.3f}")

# --- Peak Intensity vs Distance (with error bars) ---
distances = list(results.keys())
I_max_values = np.array([results[d]["I_max"] for d in distances]) / I_sat
I_Ave_max_values = np.array([results[d]["I_Ave_max"] for d in distances]) / I_sat

# Extract propagated uncertainties
u_I_max_norms = np.array([results[d]["u_I_max_norm"] for d in distances])
u_I_Ave_norms = np.array([results[d]["u_I_avg_norm"] for d in distances])

plt.figure(figsize=(8, 5))

# Plot with error bars
plt.errorbar(
	distances, I_max_values, yerr=u_I_max_norms,
	fmt="o-", color="tab:red", zorder = 1, capsize=3, label=r"Peak $I(r)$"
)
plt.errorbar(
	distances, I_Ave_max_values, yerr=u_I_Ave_norms,
	fmt="x-", color="tab:orange", zorder = 0, capsize=3, label=r"Peak $I_{avg}$"
)

plt.xlabel("Distance from 0 point (mm)")
plt.ylabel(r"$I_{max}$ / $I_{sat}$")
plt.title("Peak intensity vs distance")
plt.legend(loc="lower right")

ax = plt.gca()
if mode == 2:
	ax.xaxis.set_minor_locator(AutoMinorLocator(10))
else:
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))

ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.tight_layout()

plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

if save_all_plots:
	if mode == 0:
		plt.savefig("I_max_distance_graph_errors", dpi=300, bbox_inches='tight')
	elif mode == 1:
		plt.savefig("I_max_distance_graph_errors_new", dpi=300, bbox_inches='tight')
	else:
		plt.savefig("I_max_distance_graph_errors_new_spec", dpi=300, bbox_inches='tight')

plt.show()

# --- Print summary table of I/I_sat values with uncertainties (scientific format) ---
print("\n" + "-"*75)
print(" Summary of normalised intensities (I/I_sat) with uncertainties")
print("-"*75)
print(f"{'Distance (mm)':>13s} | {'I_max/I_sat':>25s} | {'I_avg/I_sat':>25s}")
print("-"*75)

for d in sorted(results.keys()):
	Imax_norm = results[d]["I_max"] / I_sat
	Iavg_norm = results[d]["I_Ave_max"] / I_sat
	u_Imax_norm = results[d]["u_I_max_norm"]
	u_Iavg_norm = results[d]["u_I_avg_norm"]

	print(f"{d:13.0f} | {Imax_norm:10.3e} ± {u_Imax_norm:8.1e} | {Iavg_norm:10.3e} ± {u_Iavg_norm:8.1e}")

print("-"*75)

# --- Generate LaTeX table output (scientific notation) ---
print("\nLaTeX table output:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Normalised peak and average intensities with propagated uncertainties.}")
print("\\begin{tabular}{ccc}")
print("\\hline")
print("Distance (mm) & $I_{\\max}/I_{\\mathrm{sat}}$ & $I_{\\mathrm{avg}}/I_{\\mathrm{sat}}$\\\\")
print("\\hline")

for d in sorted(results.keys()):
	Imax_norm = results[d]["I_max"] / I_sat
	Iavg_norm = results[d]["I_Ave_max"] / I_sat
	u_Imax_norm = results[d]["u_I_max_norm"]
	u_Iavg_norm = results[d]["u_I_avg_norm"]

	Imax_str = f"${Imax_norm:.3e} \\pm {u_Imax_norm:.1e}$"
	Iavg_str = f"${Iavg_norm:.3e} \\pm {u_Iavg_norm:.1e}$"
	print(f"{d:>4} & {Imax_str} & {Iavg_str} \\\\")

print("\\hline")
print("\\end{tabular}")
print("\\end{table}")


def compute_and_plot_pixel_sums(results):
	"""
	Computes the total (unscaled) pixel sum of each raw image
	and plots pixel-sum vs. distance.

	Parameters
	----------
	results : dict
		The results dictionary populated after process_image().

	Returns
	-------
	distances : np.ndarray
		Sorted array of distances.
	pixel_sums : np.ndarray
		Total pixel value for each distance.
	"""
	distances = np.array(sorted(results.keys()))
	pixel_sums = np.array([np.sum(results[d]["img"]) for d in distances])

	plt.figure(figsize=(8, 5))
	plt.plot(distances, pixel_sums, "o-", zorder=1)
	plt.xlabel("Voltage (mV)")
	plt.ylabel("Total pixel sum (arb.)")
	plt.title("Sum of all pixel values vs. distance")

	ax = plt.gca()

	if mode == 2:
		ax.xaxis.set_minor_locator(AutoMinorLocator(10))
	else:
		ax.xaxis.set_minor_locator(AutoMinorLocator(4))
	
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))

	plt.tight_layout()
	plt.show()

	return distances, pixel_sums


#distances, pixel_sums = compute_and_plot_pixel_sums(results)

# --- Integrated (scaled) total power vs voltage ---

# Distances/voltages sorted
voltages = np.array(sorted(results.keys()))

# Unscaled total integrated power from each image
P_unscaled = np.array([results[v]["P_total"] for v in voltages])

# Scaled powers: P_scaled = P_total × k_mean_global
P_scaled = P_unscaled * k_mean_global

plt.figure(figsize=(8, 5))
plt.plot(voltages, P_scaled, "o-", color="tab:blue", lw=2, markersize=6)

plt.xlabel("Voltage (mV)")
plt.ylabel("Integrated power (W)")
plt.title("Integrated, Scaled Power vs. Applied Voltage")

ax = plt.gca()
if mode == 2:
	ax.xaxis.set_minor_locator(AutoMinorLocator(10))
else:
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.grid(False)
plt.tight_layout()

# Optional: save

if save_all_plots:
	if mode == 0:
		plt.savefig("scaled_power_vs_distance.png", dpi=300, bbox_inches='tight')
	elif mode == 1:
		plt.savefig("scaled_power_vs_distance_new.png", dpi=300, bbox_inches='tight')
	else:
		plt.savefig("scaled_power_vs_voltage.png", dpi=300, bbox_inches='tight')

plt.show()


plt.figure(figsize=(8, 5))
plt.plot(voltages, P_scaled/np.max(P_scaled), "o-", color="tab:blue", lw=2, markersize=6)

plt.xlabel("Voltage (mV)")
plt.ylabel("Transmission")
plt.title("Transmission vs. Applied Voltage")

ax = plt.gca()
if mode == 2:
	ax.xaxis.set_minor_locator(AutoMinorLocator(10))
else:
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.grid(False)
plt.tight_layout()

# Optional: save

if save_all_plots:
	if mode == 0:
		plt.savefig("Transmission_vs_distance.png", dpi=300, bbox_inches='tight')
	elif mode == 1:
		plt.savefig("Transmission_vs_distance_new.png", dpi=300, bbox_inches='tight')
	else:
		plt.savefig("Transmission_vs_voltage.png", dpi=300, bbox_inches='tight')

plt.show()