import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from matplotlib import rcParams
from libs import main_functions as mf

# =========================
# Working directory
# =========================
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

# =========================
# Plot style
# =========================
fontsz = 16
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif'

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

# =========================
# Constants
# =========================
Detuning = np.linspace(-10, 10, 2000) * 1e3   # MHz
E_in = np.array([1, 0, 0])

choice = 1
AgCustomGroundPopulation = True

Dline = 'D2'
lcell = 25e-3
Bfield = 0
Btheta = 0

colours = ['deepskyblue','firebrick', 'purple', 'darkkhaki', 'orange', 'pink']

# Base values
BASE_TEMP = 130.23
BASE_NUMDEN = 1.679e16
BASE_CUSTOMA = None
BASE_ISOSHIFT = (229.24, -246.76)

# Output folders
FRAME_ROOT = "animation_frames"
GIF_ROOT = "gifs"

os.makedirs(FRAME_ROOT, exist_ok=True)
os.makedirs(GIF_ROOT, exist_ok=True)


def format_sci_tex(num):
	exp = int(np.floor(np.log10(num)))
	coeff = num / 10**exp
	return rf"${coeff:.2f} \times 10^{{{exp}}}$"


def get_element(choice, Dline):
	if choice == 0:
		return 'Rb'
	elif choice == 1:
		return 'Ag'
	elif choice == 2:
		return 'K'
	elif choice == 3:
		return 'Na'
	else:
		return 'Cs'

def build_custom_population(customa, use_custom=True):
	if (not use_custom) or (customa is None):
		return None

	a = customa
	b = (1 - a) / 3
	return [a, b, b, b]


def format_seconds(seconds):
	seconds = int(round(seconds))
	h = seconds // 3600
	m = (seconds % 3600) // 60
	s = seconds % 60

	if h > 0:
		return f"{h:d}h {m:02d}m {s:02d}s"
	elif m > 0:
		return f"{m:d}m {s:02d}s"
	else:
		return f"{s:d}s"

def compute_spectra(Temp, AgNumberDensity, customa, AgIsotopeShift):
	element = get_element(choice, Dline)
	custpop = build_custom_population(customa, AgCustomGroundPopulation)

	p_dict = {
		'Elem': element,
		'Dline': Dline,
		'T': Temp,
		'lcell': lcell,
		'Bfield': Bfield,
		'Btheta': Btheta,
		'AgNumden': AgNumberDensity,
		'Isotope_Combination': 1,
		'CustomPop': custpop,
		'AgIsotope_shift': AgIsotopeShift
	}

	p_dict2 = {
		'Elem': element,
		'Dline': Dline,
		'T': Temp,
		'lcell': lcell,
		'Bfield': Bfield,
		'Btheta': Btheta,
		'AgNumden': AgNumberDensity,
		'Isotope_Combination': 2,
		'CustomPop': custpop,
		'AgIsotope_shift': AgIsotopeShift
	}

	p_dict3 = {
		'Elem': element,
		'Dline': Dline,
		'T': Temp,
		'lcell': lcell,
		'Bfield': Bfield,
		'Btheta': Btheta,
		'AgNumden': AgNumberDensity,
		'Isotope_Combination': 0,
		'CustomPop': custpop,
		'AgIsotope_shift': AgIsotopeShift
	}

	[S0, S1, S2, S3, E_out, Ix, Iy] = mf.get_spectra(
		Detuning, E_in, p_dict,
		outputs=['S0', 'S1', 'S2', 'S3', 'E_out', 'Ix', 'Iy']
	)

	[S0_1] = mf.get_spectra(Detuning, E_in, p_dict2, outputs=['S0'])
	[S0_2] = mf.get_spectra(Detuning, E_in, p_dict3, outputs=['S0'])

	return S0, S0_1, S0_2


def plot_frame(Temp, AgNumberDensity, customa, AgIsotopeShift, save_path, varied_param_name):
	element = get_element(choice, Dline)
	line = int(Dline[-1])
	total_isotope_shift = AgIsotopeShift[0] - AgIsotopeShift[1]

	S0, S0_1, S0_2 = compute_spectra(Temp, AgNumberDensity, customa, AgIsotopeShift)

	theory_detuning = Detuning / 1e3
	theory_curve = S0_2[0].real

	fig, ax = plt.subplots(figsize=(8, 5))

	# First isotope combination
	for i in range(len(S0) - 1):
		if len(S0) >= 7:
			color = colours[1] if i <= 2 else colours[0]
		else:
			color = colours[1] if i <= 1 else colours[0]

		trans = S0[i].real
		ax.plot(theory_detuning, trans, color=color, linewidth=1.5, alpha=0.8, linestyle="--")

		tmin = np.min(trans)
		idx = np.argmin(trans)
		ax.vlines(theory_detuning[idx], tmin, 2, color=color, linewidth=1.2, alpha=0.8, linestyle="--")

	# Second isotope combination
	for i in range(len(S0_1) - 1):
		if len(S0_1) >= 7:
			color = colours[3] if i <= 2 else colours[2]
		else:
			color = colours[3] if i <= 1 else colours[2]

		trans = S0_1[i].real
		ax.plot(theory_detuning, trans, color=color, linewidth=1.5, alpha=0.8, linestyle="--")

		tmin = np.min(trans)
		idx = np.argmin(trans)
		ax.vlines(theory_detuning[idx], tmin, 2, color=color, linewidth=1.2, alpha=0.8, linestyle="--")

	# Total transmission
	ax.plot(theory_detuning, theory_curve, color="grey", linewidth=1.7)
	ax.fill_between(theory_detuning, theory_curve, 1, color="lightgrey", alpha=0.5)
	ax.axhline(1, color='grey', lw=1)

	# Lower-right labels
	x_text = 4.8

	ax.text(x_text, 0.075+0.08*4, f"{element}-D$_{line}$", ha="right", va="top", fontsize=fontsz+1)
	ax.text(x_text, 0.075+0.08*3, rf"$T$ = {Temp:.2f} $\degree$C", ha="right", va="top")
	ax.text(x_text, 0.075+0.08*2, rf"$N_D$ = {format_sci_tex(AgNumberDensity)}", ha="right", va="top")
	ax.text(x_text, 0.075+0.08, rf"$\Delta_{{iso}}$ = {total_isotope_shift:.2f} MHz", ha="right", va="top")

	if customa is not None:
		ax.text(x_text, 0.077, rf"$F=0$ Ground Population = {customa*100:.1f}%", ha="right", va="top")

	#ax.text(-3.8, 0.075, f"Varied: {varied_param_name}", ha="left", va="top", fontsize=fontsz-1)

	ax.set_xlabel("Linear Detuning (GHz)")
	ax.set_ylabel("Transmission")
	ax.set_xlim([-4, 5])
	ax.set_ylim([0.0, 1.15])

	plt.tight_layout()
	plt.savefig(save_path, dpi=200, bbox_inches='tight')
	plt.close(fig)

def ease_in_out_values(start, stop, n):
	u = np.linspace(0, 1, n)
	eased = 0.5 * (1 - np.cos(np.pi * u))
	return start + (stop - start) * eased

def make_loop_values(start, middle, total_frames):
	# forward has n_half frames, backward contributes n_half-1
	# total = 2*n_half - 1
	n_half = int(np.ceil((total_frames + 1) / 2))

	forward = ease_in_out_values(start, middle, n_half)
	backward = ease_in_out_values(middle, start, n_half)

	values = np.concatenate([forward, backward[1:]])

	# trim in case rounding gave one extra frame
	return values[:total_frames]


def make_multi_slow_loop(start, middle, total_frames):
	# 4 eased segments:
	# [start -> q1], [q1 -> middle], [middle -> q2], [q2 -> start]
	# with overlaps removed, total = 4*n_seg - 3
	n_seg = int(np.ceil((total_frames + 3) / 4))

	q1 = start + 0.5 * (middle - start)
	q2 = middle + 0.5 * (start - middle)

	keypoints = [start, q1, middle, q2, start]

	values = []
	for i in range(len(keypoints) - 1):
		seg = ease_in_out_values(keypoints[i], keypoints[i + 1], n_seg)
		if i > 0:
			seg = seg[1:]
		values.append(seg)

	values = np.concatenate(values)

	# trim in case rounding gave one extra frame
	return values[:total_frames]

def make_gif(frame_folder, gif_path, duration=0.25):
	frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
	images = [imageio.imread(f) for f in frame_files]

	imageio.mimsave(
		gif_path,
		images,
		duration=duration,   # seconds per frame
		loop=0               # infinite looping
	)


def make_animation(param_name, values, fps):
	frame_folder = os.path.join(FRAME_ROOT, param_name)
	os.makedirs(frame_folder, exist_ok=True)

	# Optional: clear old frames
	for f in glob.glob(os.path.join(frame_folder, "*.png")):
		os.remove(f)

	nframes = len(values)
	t0 = time.perf_counter()

	print(f"\nStarting animation: {param_name}")
	print(f"Total frames: {nframes}")

	for i, val in enumerate(values):
		Temp = BASE_TEMP
		AgNumberDensity = BASE_NUMDEN
		customa = BASE_CUSTOMA
		AgIsotopeShift = BASE_ISOSHIFT

		if param_name == "temperature":
			Temp = val
		elif param_name == "number_density":
			AgNumberDensity = val
		elif param_name == "customa":
			customa = val
		elif param_name == "isotope_shift":
			midpoint = 0.5 * (BASE_ISOSHIFT[0] + BASE_ISOSHIFT[1])
			half_sep = 0.5 * val
			AgIsotopeShift = (midpoint + half_sep, midpoint - half_sep)

		save_path = os.path.join(frame_folder, f"frame_{i:04d}.png")

		frame_start = time.perf_counter()

		plot_frame(
			Temp=Temp,
			AgNumberDensity=AgNumberDensity,
			customa=customa,
			AgIsotopeShift=AgIsotopeShift,
			save_path=save_path,
			varied_param_name=param_name
		)

		frame_end = time.perf_counter()

		frames_done = i + 1
		elapsed = frame_end - t0
		avg_per_frame = elapsed / frames_done
		remaining = avg_per_frame * (nframes - frames_done)
		total_est = avg_per_frame * nframes
		this_frame = frame_end - frame_start

		print(
			f"[{frames_done:>3d}/{nframes}] "
			f"{param_name}: value = {val:.5g} | "
			f"frame = {this_frame:.2f}s | "
			f"elapsed = {format_seconds(elapsed)} | "
			f"remaining ~ {format_seconds(remaining)} | "
			f"total ~ {format_seconds(total_est)}"
		)

	gif_start = time.perf_counter()
	gif_path = os.path.join(GIF_ROOT, f"{param_name}.gif")
	make_gif(frame_folder, gif_path, duration=1/fps)
	gif_end = time.perf_counter()

	total_time = gif_end - t0
	print(f"Finished animation: {param_name}")
	print(f"GIF creation time: {format_seconds(gif_end - gif_start)}")
	print(f"Total time: {format_seconds(total_time)}\n")


fps = 30
total_seconds = 16/4
total_frames = int(total_seconds * fps)

temp_values = make_loop_values(500, -272, total_frames)
numden_values = make_loop_values(1.60e14, 1.60e17, total_frames)
customa_values = make_loop_values(0, 1, total_frames)
shift_values = make_loop_values(0, 5000, total_frames)

make_animation("temperature", temp_values, fps)
make_animation("number_density", numden_values, fps)
make_animation("customa", customa_values, fps)
make_animation("isotope_shift", shift_values, fps)