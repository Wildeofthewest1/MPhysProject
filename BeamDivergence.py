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

fontsz = 16
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'

pixel_size = 3.45e-6 #m
pixel_area = pixel_size**2 #3.45 x 3.45 micrometers squared

beam_images = {
	0:   {"centre": (570, 790), "exposure": 18.088-3},
	25:   {"centre": (526, 785), "exposure": 18.088e-3},
	50:  {"centre": (542, 786), "exposure": 18.088-3},
	75:  {"centre": (501, 769), "exposure": 18.088e-3},    
	100: {"centre": (560, 775), "exposure": 18.088e-3},
	125:   {"centre": (618, 772), "exposure": 18.088e-3},
	150:  {"centre": (591, 760), "exposure": 18.088e-3},
	175:  {"centre": (635, 775), "exposure": 18.088e-3}, 
	200: {"centre": (565, 765), "exposure": 18.088e-3},
	225:   {"centre": (600, 755), "exposure": 18.088e-3},
	250:  {"centre": (598, 765), "exposure": 18.088e-3},
	275:  {"centre": (573, 767), "exposure": 18.088e-3}, 
	300: {"centre": (575, 758), "exposure": 18.088e-3},
	325:   {"centre": (562, 766), "exposure": 18.088e-3},
	350:  {"centre": (572, 768), "exposure": 18.088e-3},
	375:  {"centre": (507, 753), "exposure": 18.088e-3}, 
	400: {"centre": (565, 751), "exposure": 18.088e-3},
	425: {"centre": (521, 753), "exposure": 18.088e-3},
	450: {"centre": (509, 743), "exposure": 18.088e-3},
	475: {"centre": (515, 743), "exposure": 18.088e-3},
	500: {"centre": (558, 752), "exposure": 18.088e-3},
	525: {"centre": (558, 763), "exposure": 18.088e-3},
	550: {"centre": (532, 761), "exposure": 18.088e-3},
}
base_path = "NEWSHAPEBEAMPICS/"
end_path = "_18.088ms.bmp"
default_exposure = 12.097e-3  # s
exposure_error = 0.001

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import os

ARC_HALF_ANGLE_DEG = 20
ARC_N_ANGLES = 21
ARC_DR = 1.0

show_debug = False#True        # turn on/off
DEBUG_RAY_LEN_PX = 250         # how far to draw rays (pixels)
DEBUG_ARC_COLOR_X = "w"        # x-arc rays colour
DEBUG_ARC_COLOR_Y = "c"        # y-arc rays colour

# --- Waist definition: threshold = I0 / e**WAIST_E_POWER ---
WAIST_E_POWER = 2   # set to 1 for 1/e, 2 for 1/e^2 (Gaussian standard)

# --- Centre brightness averaging ---
CENTRE_AVG_RADIUS_PX = 20   # radius of circle (pixels). e.g. 3 px, 5 px, etc.

SAVE_DIVERGENCE_PLOT = False#True
DIVERGENCE_PLOT_FILENAME = f"BeamDivergence_{WAIST_E_POWER}p_arc{ARC_HALF_ANGLE_DEG}deg_r{CENTRE_AVG_RADIUS_PX}px.png"
DIVERGENCE_PLOT_DPI = 300

# ---------------- helpers ----------------
def to3string(dist: int):
	return str(dist).zfill(3)

def bilinear_sample(img, x, y):
	"""
	Bilinear sample at floating (x,y). x is column, y is row.
	"""
	h, w = img.shape
	x = np.clip(x, 0, w - 1)
	y = np.clip(y, 0, h - 1)

	x0 = int(np.floor(x)); x1 = min(x0 + 1, w - 1)
	y0 = int(np.floor(y)); y1 = min(y0 + 1, h - 1)

	dx = x - x0
	dy = y - y0

	Ia = img[y0, x0]
	Ib = img[y0, x1]
	Ic = img[y1, x0]
	Id = img[y1, x1]

	return (Ia * (1 - dx) * (1 - dy) +
			Ib * dx * (1 - dy) +
			Ic * (1 - dx) * dy +
			Id * dx * dy)

import numpy as np
from scipy.ndimage import map_coordinates

def mean_and_sem(values):
    """
    Return (mean, SEM) with ddof=1. If N<2, SEM=nan.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return np.nan, np.nan
    mu = float(np.mean(v))
    if n < 2:
        return mu, np.nan
    sem = float(np.std(v, ddof=1) / np.sqrt(n))
    return mu, sem

def centre_amplitude_circle_mean_sem(img, cx, cy, r_px):
    """
    Mean+SEM of pixels inside a circle of radius r_px.
    """
    if r_px <= 0:
        raise ValueError("r_px must be > 0")
    h, w = img.shape

    x_min = int(np.floor(cx - r_px))
    x_max = int(np.ceil (cx + r_px))
    y_min = int(np.floor(cy - r_px))
    y_max = int(np.ceil (cy + r_px))

    x_min = max(0, x_min); x_max = min(w - 1, x_max)
    y_min = max(0, y_min); y_max = min(h - 1, y_max)

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    X, Y = np.meshgrid(xs, ys)

    r2 = (X - cx)**2 + (Y - cy)**2
    mask = r2 <= (r_px**2)

    vals = img[y_min:y_max + 1, x_min:x_max + 1][mask]
    return mean_and_sem(vals)

def axis_arc_averaged_profile_mean_sem(img, cx, cy, axis="x", delta_deg=4.0, n_angles=9, dr=1.0):
    """
    Like axis_arc_averaged_profile, but returns:
      s (signed pixels), I_mean(s), I_sem(s) where SEM is across rays.
    """
    h, w = img.shape
    r_max = float(np.hypot(w, h))

    delta = np.deg2rad(delta_deg)
    angles = np.linspace(-delta, +delta, n_angles)

    if axis == "x":
        base_dirs = [0.0, np.pi]
    elif axis == "y":
        base_dirs = [0.5*np.pi, 1.5*np.pi]
    else:
        raise ValueError("axis must be 'x' or 'y'")

    profiles_neg = []
    profiles_pos = []

    for a in angles:
        r, Ipos = radial_profile(img, cx, cy, base_dirs[0] + a, r_max, dr=dr)
        _, Ineg = radial_profile(img, cx, cy, base_dirs[1] + a, r_max, dr=dr)
        profiles_pos.append(Ipos)
        profiles_neg.append(Ineg)

    Ppos = np.vstack(profiles_pos)  # shape (n_angles, n_r)
    Pneg = np.vstack(profiles_neg)

    Ipos_mean = np.mean(Ppos, axis=0)
    Ineg_mean = np.mean(Pneg, axis=0)

    # SEM across rays
    if n_angles >= 2:
        Ipos_sem = np.std(Ppos, axis=0, ddof=1) / np.sqrt(n_angles)
        Ineg_sem = np.std(Pneg, axis=0, ddof=1) / np.sqrt(n_angles)
    else:
        Ipos_sem = np.full_like(Ipos_mean, np.nan, dtype=float)
        Ineg_sem = np.full_like(Ineg_mean, np.nan, dtype=float)

    # Stitch into signed profile
    s_neg = -r[1:][::-1]
    s_pos = r
    s = np.concatenate([s_neg, s_pos])

    I_mean = np.concatenate([Ineg_mean[1:][::-1], Ipos_mean])
    I_sem  = np.concatenate([Ineg_sem [1:][::-1], Ipos_sem ])

    return s, I_mean, I_sem

def centre_amplitude_circle(img, cx, cy, r_px):
    """
    Mean intensity inside a circle of radius r_px (pixels) centred at (cx,cy).
    Uses pixel-centre geometry; cx,cy can be floats.
    """
    if r_px <= 0:
        raise ValueError("r_px must be > 0")

    h, w = img.shape

    # Bounding box around circle
    x_min = int(np.floor(cx - r_px))
    x_max = int(np.ceil (cx + r_px))
    y_min = int(np.floor(cy - r_px))
    y_max = int(np.ceil (cy + r_px))

    # Clip to image
    x_min = max(0, x_min); x_max = min(w - 1, x_max)
    y_min = max(0, y_min); y_max = min(h - 1, y_max)

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    X, Y = np.meshgrid(xs, ys)

    # Distance from centre (in pixels)
    r2 = (X - cx)**2 + (Y - cy)**2
    mask = r2 <= (r_px**2)

    vals = img[y_min:y_max + 1, x_min:x_max + 1][mask]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))

def centre_amplitude_25px(img, cx, cy):
	"""
	Mean of the 5x5 = 25 pixels centred on (cx,cy) (rounded to nearest pixel).
	"""
	x0 = int(round(cx))
	y0 = int(round(cy))
	r = 2  # 5x5

	y_min = max(0, y0 - r); y_max = min(img.shape[0], y0 + r + 1)
	x_min = max(0, x0 - r); x_max = min(img.shape[1], x0 + r + 1)

	patch = img[y_min:y_max, x_min:x_max]
	return float(np.mean(patch))

def crossing_with_uncertainty(s, I, Isem, thr, thr_sem):
    """
    Find left/right crossings AND estimate their uncertainties using local slope.
    Returns:
      (sL, sR, sL_sem, sR_sem)
    """
    sL, sR = threshold_crossings(s, I, thr)

    def local_sigma_at_crossing(s_cross):
        if s_cross is None:
            return np.nan
        # find nearest segment index
        idx = np.searchsorted(s, s_cross) - 1
        idx = int(np.clip(idx, 0, len(s) - 2))

        s0, s1 = s[idx], s[idx+1]
        I0, I1 = I[idx], I[idx+1]
        sem0, sem1 = Isem[idx], Isem[idx+1]

        slope = (I1 - I0) / (s1 - s0) if (s1 != s0) else np.nan
        if not np.isfinite(slope) or slope == 0:
            return np.nan

        # SEM of intensity at crossing: take average of adjacent SEMs
        Icross_sem = np.nanmean([sem0, sem1])
        return float(np.sqrt(Icross_sem**2 + thr_sem**2) / abs(slope))

    sL_sem = local_sigma_at_crossing(sL)
    sR_sem = local_sigma_at_crossing(sR)

    return sL, sR, sL_sem, sR_sem


def radial_profile(img, cx, cy, theta, r_max, dr=1.0):
	"""
	Sample intensity along a ray from the centre at angle theta (radians).
	Returns r (pixels) and I(r).
	Uses bilinear interpolation via map_coordinates.
	"""
	r = np.arange(0.0, r_max + dr, dr)
	xs = cx + r * np.cos(theta)
	ys = cy + r * np.sin(theta)

	coords = np.vstack([ys, xs])  # map_coordinates wants [row, col]
	I = map_coordinates(img, coords, order=1, mode="nearest")
	return r, I

def axis_arc_averaged_profile(img, cx, cy, axis="x", delta_deg=4.0, n_angles=9, dr=1.0):
	"""
	Average radial profiles over a small angular arc around an axis direction.
	axis='x' averages around 0 and pi; axis='y' averages around pi/2 and 3pi/2.
	Returns signed coordinate s (pixels) and averaged intensity I(s).

	The output profile is 1D like a lineout through the centre, but denoised
	by averaging nearby angles.
	"""
	h, w = img.shape
	r_max = float(np.hypot(w, h))  # safe upper bound

	delta = np.deg2rad(delta_deg)
	angles = np.linspace(-delta, +delta, n_angles)

	if axis == "x":
		base_dirs = [0.0, np.pi]  # +x and -x
	elif axis == "y":
		base_dirs = [0.5*np.pi, 1.5*np.pi]  # +y and -y
	else:
		raise ValueError("axis must be 'x' or 'y'")

	# Sample rays and build two half-profiles, then stitch into signed profile.
	# Negative side: base_dirs[1], Positive side: base_dirs[0]
	profiles_neg = []
	profiles_pos = []

	for a in angles:
		r, Ipos = radial_profile(img, cx, cy, base_dirs[0] + a, r_max, dr=dr)
		_, Ineg = radial_profile(img, cx, cy, base_dirs[1] + a, r_max, dr=dr)
		profiles_pos.append(Ipos)
		profiles_neg.append(Ineg)

	Ipos_mean = np.mean(np.vstack(profiles_pos), axis=0)
	Ineg_mean = np.mean(np.vstack(profiles_neg), axis=0)

	# Build signed coordinate s: negative side reversed (excluding r=0 to avoid duplicate centre)
	s_neg = -r[1:][::-1]
	s_pos = r
	s = np.concatenate([s_neg, s_pos])

	I = np.concatenate([Ineg_mean[1:][::-1], Ipos_mean])
	return s, I

def threshold_crossings(profile_s, profile_I, threshold):
	"""
	Find the two crossings of I(s)=threshold around s=0, using linear interpolation.
	Returns (s_left, s_right) in pixels, or (None, None).
	"""
	s = profile_s
	I = profile_I

	# Split into left (s<0) and right (s>0)
	left_mask = s < 0
	right_mask = s > 0

	sL = None
	if np.any(left_mask):
		sl = s[left_mask]
		Il = I[left_mask]
		# Search from centre outward: nearest to 0 is last element in sl
		for k in range(len(sl)-1, 0, -1):
			y0, y1 = Il[k], Il[k-1]
			if (y0 >= threshold and y1 < threshold) or (y0 <= threshold and y1 > threshold):
				# interpolate between sl[k] and sl[k-1]
				if y1 == y0:
					sL = float(sl[k])
				else:
					t = (threshold - y0) / (y1 - y0)
					sL = float(sl[k] + t*(sl[k-1] - sl[k]))
				break

	sR = None
	if np.any(right_mask):
		sr = s[right_mask]
		Ir = I[right_mask]
		for k in range(0, len(sr)-1):
			y0, y1 = Ir[k], Ir[k+1]
			if (y0 >= threshold and y1 < threshold) or (y0 <= threshold and y1 > threshold):
				if y1 == y0:
					sR = float(sr[k])
				else:
					t = (threshold - y0) / (y1 - y0)
					sR = float(sr[k] + t*(sr[k+1] - sr[k]))
				break

	return sL, sR

def centre_amplitude(img, cx, cy, method="mean3"):
	"""
	Return centre intensity I0.
	method:
	  - "pixel": single-pixel (bilinear if cx,cy not ints)
	  - "mean3": mean over 3x3 around nearest pixel
	  - "mean5": mean over 5x5 around nearest pixel
	"""
	if method == "pixel":
		return float(bilinear_sample(img, cx, cy))

	x0 = int(round(cx))
	y0 = int(round(cy))

	if method == "mean3":
		r = 1
	elif method == "mean5":
		r = 2
	else:
		raise ValueError("Unknown method")

	y_min = max(0, y0 - r); y_max = min(img.shape[0], y0 + r + 1)
	x_min = max(0, x0 - r); x_max = min(img.shape[1], x0 + r + 1)
	return float(np.mean(img[y_min:y_max, x_min:x_max]))

def crossing_positions_1d(profile, centre_idx, threshold):
	"""
	Given a 1D profile and a centre index, find left and right positions
	where the profile crosses 'threshold', using linear interpolation.

	Returns (x_left, x_right) in index units (floats). Returns (None, None)
	if crossings not found.
	"""
	n = len(profile)

	# --- search left ---
	i = int(np.clip(centre_idx, 0, n - 1))
	left = None
	for k in range(i, 0, -1):
		y1 = profile[k]
		y0 = profile[k - 1]
		if (y1 >= threshold and y0 < threshold) or (y1 <= threshold and y0 > threshold):
			# interpolate between k-1 and k
			if y1 == y0:
				left = float(k)
			else:
				t = (threshold - y0) / (y1 - y0)  # fraction from k-1 -> k
				left = (k - 1) + t
			break

	# --- search right ---
	right = None
	for k in range(i, n - 1):
		y0 = profile[k]
		y1 = profile[k + 1]
		if (y0 >= threshold and y1 < threshold) or (y0 <= threshold and y1 > threshold):
			if y1 == y0:
				right = float(k)
			else:
				t = (threshold - y0) / (y1 - y0)  # fraction from k -> k+1
				right = k + t
			break

	return left, right

def process_image(distance, centre=None, exposure=None,
				  base_path="NEWSHAPEBEAMPICS/", end_path="_18.088ms.bmp",
				  default_exposure=12.097e-3):
	path = f"{base_path}{to3string(distance)}{end_path}"
	img = plt.imread(path)

	if img.ndim == 3:
		img = img[:, :, 0]

	if exposure is None:
		exposure = default_exposure

	# normalise to "counts per second"
	if img.dtype.kind == "f":
		img = img / exposure
	else:
		img = img / (exposure * 255)

	# centre
	if centre is not None:
		cx, cy = centre
	else:
		cy, cx = center_of_mass(img)

	return img, float(cx), float(cy)

# ---------------- main analysis ----------------
pixel_size = 3.45e-6  # m

results = {}  # stores per-distance waists etc.

if WAIST_E_POWER == 1:
    waist_label = r"1/e"
elif WAIST_E_POWER == 2:
    waist_label = r"1/e$^2$"
else:
    waist_label = rf"1/e$^{{{WAIST_E_POWER}}}$"

for d, info in beam_images.items():
	img, cx, cy = process_image(
		d,
		centre=info.get("centre"),
		exposure=info.get("exposure") or default_exposure,
		base_path=base_path,
		end_path=end_path,
		default_exposure=default_exposure
	)

	I0, I0_sem = centre_amplitude_circle_mean_sem(img, cx, cy, CENTRE_AVG_RADIUS_PX)


	thr = I0 / (np.e**WAIST_E_POWER)
	thr_sem = I0_sem / (np.e**WAIST_E_POWER)


	# Arc-averaged “axis profiles”
	sx, Ix, Ix_sem = axis_arc_averaged_profile_mean_sem(img, cx, cy, "x",
														delta_deg=ARC_HALF_ANGLE_DEG,
														n_angles=ARC_N_ANGLES,
														dr=ARC_DR)

	sy, Iy, Iy_sem = axis_arc_averaged_profile_mean_sem(img, cx, cy, "y",
														delta_deg=ARC_HALF_ANGLE_DEG,
														n_angles=ARC_N_ANGLES,
														dr=ARC_DR)

	# Crossings at 1/e^2
	xL, xR, xL_sem, xR_sem = crossing_with_uncertainty(sx, Ix, Ix_sem, thr, thr_sem)
	yT, yB, yT_sem, yB_sem = crossing_with_uncertainty(sy, Iy, Iy_sem, thr, thr_sem)

	wx = wy = None
	wx_sem = wy_sem = None

	if xL is not None and xR is not None and xR > xL:
		wx = 0.5 * (xR - xL) * pixel_size
		wx_sem = 0.5 * pixel_size * np.sqrt(xL_sem**2 + xR_sem**2)

	if yT is not None and yB is not None and yB > yT:
		wy = 0.5 * (yB - yT) * pixel_size
		wy_sem = 0.5 * pixel_size * np.sqrt(yT_sem**2 + yB_sem**2)

	results[d] = {
		"img": img,
		"cx": cx, "cy": cy,
		"I0": I0,
		"threshold": thr,
		"wx_m": wx,
		"wy_m": wy,
		"x_cross": (xL, xR),
		"y_cross": (yT, yB),
		"wx_sem_m": wx_sem,
		"wy_sem_m": wy_sem,
		"I0_sem": I0_sem,
		"thr_sem": thr_sem,
		"x_cross_sem": (xL_sem, xR_sem),
		"y_cross_sem": (yT_sem, yB_sem),

	}

	# --- optional: show per-image overlay for debugging ---
	# Mark the pixels above threshold (1/e region) and draw crosshair + crossings.

	if show_debug:
		mask = img >= thr  # this is 1/e^2 because thr = I0 / e^2

		plt.figure(figsize=(7, 6))
		plt.imshow(img, origin="upper")
		plt.contour(mask.astype(float), levels=[0.5])  # outlines I >= I0/e^2
		plt.scatter([cx], [cy], s=50, marker="x")

		# --- draw arc rays ---
		delta = np.deg2rad(ARC_HALF_ANGLE_DEG)
		angles = np.linspace(-delta, +delta, ARC_N_ANGLES)

		R = DEBUG_RAY_LEN_PX

		def draw_rays(base_angle, colour, alpha=0.35, lw=1.0):
			for a in angles:
				th = base_angle + a
				x1 = cx + R * np.cos(th)
				y1 = cy + R * np.sin(th)
				plt.plot([cx, x1], [cy, y1], colour, alpha=alpha, lw=lw)

		# x-axis arc rays (0 and pi)
		draw_rays(0.0,       DEBUG_ARC_COLOR_X)
		draw_rays(np.pi,     DEBUG_ARC_COLOR_X)

		# y-axis arc rays (pi/2 and 3pi/2)
		draw_rays(0.5*np.pi, DEBUG_ARC_COLOR_Y)
		draw_rays(1.5*np.pi, DEBUG_ARC_COLOR_Y)

		# --- plot threshold crossing points (convert signed offsets -> pixel coords) ---
		if xL is not None and xR is not None:
			plt.scatter([cx + xL, cx + xR], [cy, cy], s=40)

		if yT is not None and yB is not None:
			plt.scatter([cx, cx], [cy + yT, cy + yB], s=40)

		title = f"d={d} mm, {waist_label}"

		if wx is not None and wy is not None:
			title += f", wx={wx*1e6:.2f} µm, wy={wy*1e6:.2f} µm"

		from matplotlib.patches import Circle

		# centre-averaging circle
		circ = Circle((cx, cy), CENTRE_AVG_RADIUS_PX, fill=False, lw=1.5)
		plt.gca().add_patch(circ)

		plt.title(title)
		plt.tight_layout()

		#plt.savefig(DIVERGENCE_PLOT_FILENAME+"_BEAM.png", dpi=DIVERGENCE_PLOT_DPI, bbox_inches="tight")

		plt.show()

# ---------------- plot waist vs distance ----------------
ds = np.array(sorted(results.keys()), dtype=float)
wx = np.array([results[d]["wx_m"] for d in ds], dtype=float)
wy = np.array([results[d]["wy_m"] for d in ds], dtype=float)
wx_sem = np.array([results[d]["wx_sem_m"] for d in ds], dtype=float)
wy_sem = np.array([results[d]["wy_sem_m"] for d in ds], dtype=float)

plt.figure()
plt.errorbar(ds, wx*1e6, yerr=wx_sem*1e6, fmt="o-", capsize=3, label=rf"$w_x$ ({waist_label})")
plt.errorbar(ds, wy*1e6, yerr=wy_sem*1e6, fmt="o-", capsize=3, label=rf"$w_y$ ({waist_label})")
plt.xlabel("Distance (mm)")
plt.ylabel(r"Waist radius ($\mu$m)")
plt.legend()
plt.tight_layout()
if SAVE_DIVERGENCE_PLOT:
    plt.savefig(DIVERGENCE_PLOT_FILENAME, dpi=DIVERGENCE_PLOT_DPI, bbox_inches="tight")
plt.show()


# ---------------- optional: crude divergence estimate (linear fit) ----------------
# If you want divergence, choose a region where w grows ~ linearly with z.
# Example: fit all points (you can restrict to far-field indices)

maskx = np.isfinite(wx) & np.isfinite(wx_sem) & (wx_sem > 0)
if maskx.sum() >= 2:
    z = ds[maskx] * 1e-3
    w = wx[maskx]
    sigma = wx_sem[maskx]

    weights = 1.0 / sigma
    p, cov = np.polyfit(z, w, 1, w=weights, cov=True)
    div_x = p[0]
    div_x_sem = np.sqrt(cov[0,0])
    print(f"Weighted divergence (x): {div_x:.3e} ± {div_x_sem:.3e} rad")

masky = np.isfinite(wy) & np.isfinite(wy_sem) & (wy_sem > 0)
if masky.sum() >= 2:
    z = ds[masky] * 1e-3
    w = wy[masky]
    sigma = wy_sem[masky]

    weights = 1.0 / sigma
    p, cov = np.polyfit(z, w, 1, w=weights, cov=True)
    div_y = p[0]
    div_y_sem = np.sqrt(cov[0,0])
    print(f"Weighted divergence (y): {div_y:.3e} ± {div_y_sem:.3e} rad")