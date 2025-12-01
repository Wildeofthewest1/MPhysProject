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
normalTransmission = True
normalTransmission = False

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

p_total = 1.21e-6 #1.02e-6#1.21e-6
p_total_error = 0.05e-6#0.04e-6
print("TOTAL MEASURED POWER = " + str(p_total) + "~+-~" + str(p_total_error) + "W")

beam_images = {	
	#4001: {"centre": (718, 556), "exposure": 6.107e-3},
	4001: {"centre": (740, 543), "exposure": 8.488e-3},
}

cutoff = 530

#base_path = "Voltage_Spec_2/"
base_path = "Spec_Voltage_Norm_Images/"
end_path = "_mV_6.107ms_450mm.bmp"

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

def process_image(centre=None, exposure=None, input_scale_factor = None):
	"""Process a single beam image and return all derived quantities."""

	path = base_path + "laser_lamp.bmp"
	img = plt.imread(path)

	if img.ndim == 3:
		img = img.mean(axis=2)
	ny, nx = img.shape

	# exposure handling
	if exposure is None:
		exposure = default_exposure

	lampImage = plt.imread(base_path + "lamp.bmp")
	if lampImage.ndim == 3:
		lampImage = lampImage.mean(axis=2)

	img = img - lampImage

	NormImage = plt.imread(base_path + "laser.bmp")
	if NormImage.ndim == 3:
		NormImage = NormImage.mean(axis=2)

	img = np.divide(img, NormImage, out=np.ones_like(img), where=NormImage != 0)

	#img = NormImage

	#img = img / (exposure * 255) #gives unscaled intensity values to each pixel

	plt.imshow(img)
	plt.show()

	sf = 1

	#print(np.sum(img))

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

	plt.imshow(polar_img, aspect="auto", cmap="CMRmap", origin="lower")
	plt.axhline(cutoff)
	plt.show()

	# integration
	P_r_unnorm = np.trapezoid(polar_img, theta, axis=1)#gives the power per unit radial length
	P_total = np.trapezoid(P_r_unnorm * r, r)#Integrates over r to get the power, need to apply a scale factor so it equals the total measured power
	P_encircled = cumtrapz(P_r_unnorm * r, r, initial=0)

	if mode == 2 and input_scale_factor is not None:
		scale_factor = input_scale_factor #input scale factor
	else:
		scale_factor = p_total / (P_total) #scale factor for just beam plots

	profile_x = r * pixel_size #radial size in m
	if mode == 2:
		profile_y = P_r_unnorm / (np.pi * 2)
	else:
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

# --- Process all images ---
results = {}
for d, info in beam_images.items():

	img, polar_img, x_prof, y_prof, P, centre, \
		polar_extent, polar_xlabel, polar_ylabel, \
		profile_label, I_max, I_ave_profile, I_ave_peak, scale_factor = process_image(
		centre=info.get("centre"),
		exposure = info.get("exposure") or default_exposure, input_scale_factor=k_mean_global
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

plt.plot(x_prof * 1e3,1-y_prof)
plt.axvline(cutoff*pixel_size*1e3)
plt.show()