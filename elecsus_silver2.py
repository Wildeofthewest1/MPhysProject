import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from libs import main_functions as mf
from matplotlib import rcParams

import os
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
#os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())


fontsz = 16
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'

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

Detuning=np.linspace(-10,10,2000)*1e3 # MHz
E_in=np.array([1,0,0]) #Horizontal Linear Light input. We define E_in = [Ex,Ey,Ez]

choice = 1 #0 = Rb, 1 = Ag, 2 = K, 3 = Na, 4 = Cs

# ---------------------------------------------------------
# UPDATED: include b0, b1 at the end
# fitresults = (Temp, N, a, shift107, shift109, b0, b1)
# Baseline: B(x) = b0 + b1*(x - x0)
# ---------------------------------------------------------

fitresults = (149.982, 1.69656e+16, 0.450258, 229.24, -246.76, 1.11171, 0.314955, 0.00146153)

fitresults = (149.982, 1.69656e+16, None, 0, -500, 1.11171, 0.314955, 0.00146153)

fitresultsErrors = (3.59048, 2.69753e-12, 0.00155848, 0, 0, 0.00222233, 0.000414683, 0.000101566)

fitresults = (233.009, 1.65191e+16, 0.427007, 0, -0, 1.09942, 0.31512, 0.00140447)    
fitresultsErrors = (11.8845, 9.4676e-12, 0.00520905, 0, 0, 0.00846909, 0.00103746, 0.000214019)
#fitresults = (149.982, 1.69656e+16, None, 229.24, -246.76, 1.11171, 0.314955, 0.00146153)
#fitresultsErrors = (3.59048, 2.69753e-12, 0.00155848, 0, 0, 0.00222233, 0.000414683, 0.000101566)

#fitresults = (402.255, 1.46035e+16, None, 229.24, -246.76, 1.04713, 0.314955, 0.00146153)
#fitresultsErrors = (3.7081, 6.15048e-13, 0, 0, 0, 0.00257602, 0, 0)

fitresults = (85.0875, 1e+15, 0.483269, 229.24, -246.76, 0.974239, 1.00788, -0.000142033, -0.000118411)
fitresultsErrors = (20.6645, 6.77552e-11, 0.00999288, 0, 0, 0.014525, 0.000502216, 0.000113887, 2.05673e-05)

first = 8
AgCustomGroundPopulation = True


import pandas as pd
frequencies = pd.read_csv("frequencies1.csv")
frequencies2 = pd.read_csv("frequencies2.csv")
frequencies3 = pd.read_csv("frequencies3.csv")
transmissions1 = pd.read_csv("transmission1.csv")
transmissions2 = pd.read_csv("transmission2.csv")
transmissions3 = pd.read_csv("transmission3.csv")
if first == 0:
	freq = np.array(frequencies["freq1"])
	trans = np.array(transmissions1["Transmission1"])
	transerr = np.array(transmissions1["Transmission1err"])
	adj = 0.011
elif first == 1:
	freq = np.array(frequencies2["freq2"])
	trans = np.array(transmissions2["Transmission2"])
	transerr = np.array(transmissions2["Transmission2err"])
	adj = 0
elif first == 2:
	freq = np.array(frequencies3["freq3"])
	trans = np.array(transmissions3["Transmission3"])
	transerr = np.array(transmissions3["Transmission3err"])
	adj = 0
elif first == 8:
	fitresults = (155.173, 1.44153e+16, 0.433209, 229.24, -246.76, 0.957844, 1.01574, 0.0068862, -0.000409223, -4.88557e-05, 5.02038e-06)
	fitresultsErrors = (2.5533, 4.57799e-10, 0.00117417, 0, 0, 0.00167431, 0.000673272, 0.00012505, 3.59029e-05, 1.90248e-06, 3.71449e-07)
	frequencies    = pd.read_csv("frequencies8A.csv")
	transmissions  = pd.read_csv("Spec15MicW8A.csv")
	freq = np.array(frequencies["freq"])
	trans = np.array(transmissions["Transmission"])
	transerr = np.array(transmissions["Transmissionerr"])
elif first == 7:
	fitresults = (146.508, 9.68541e+15, 0.427828, 229.24, -246.76, 0.971351, 1.02897, 0.00153783, -0.000107466)
	fitresultsErrors = (3.02813, 4.54146e-12, 0.00140673, 0, 0, 0.00198064, 0.000556059, 0.000115817, 2.08349e-05)
	frequencies    = pd.read_csv("frequencies7A.csv")
	transmissions  = pd.read_csv("Spec15MicW7A.csv")
	freq = np.array(frequencies["freq"])
	trans = np.array(transmissions["Transmission"])
	transerr = np.array(transmissions["Transmissionerr"])
elif first == 6:
	fitresults = (132.15, 4.23146e+15, 0.428362, 229.24, -246.76, 0.955212, 1.0232, 0.00231507, -0.000168166)
	fitresultsErrors = (5.81956, 1.5608e-11, 0.00281734, 0, 0, 0.00383613, 0.000536506, 0.000115592, 2.06861e-05)
	frequencies    = pd.read_csv("frequencies6A.csv")
	transmissions  = pd.read_csv("Spec15MicW6A.csv")
	freq = np.array(frequencies["freq"])
	trans = np.array(transmissions["Transmission"])
	transerr = np.array(transmissions["Transmissionerr"])
elif first == 4:
	fitresults = (85.0875, 1e+15, 0.483269, 229.24, -246.76, 0.974239, 1.00788, -0.000142033, -0.000118411)
	fitresultsErrors = (20.6645, 6.77552e-11, 0.00999288, 0, 0, 0.014525, 0.000502216, 0.000113887, 2.05673e-05)
	frequencies    = pd.read_csv("frequencies4A.csv")
	transmissions  = pd.read_csv("Spec15MicW4A.csv")
	freq = np.array(frequencies["freq"])
	trans = np.array(transmissions["Transmission"])
	transerr = np.array(transmissions["Transmissionerr"])

Temp = fitresults[0]
AgNumberDensity = fitresults[1]
customa = fitresults[2]
AgIsotopeShift = (fitresults[3],fitresults[4])
deltaf = fitresults[5]

Dline = 'D2'
lcell = 25e-3
Bfield = 0
Btheta = 0
ShowTransPlot = False

if choice == 0:
	element = 'Rb'
	if Dline == 'D2':
		Temp = 19
	else:
		Temp = 28
elif choice == 1:
	element = 'Ag'
elif choice == 2:
	element = 'K'
	if Dline == 'D2':
		Temp = 45
	else:
		Temp = 53
elif choice == 3:
	element = 'Na'
	if Dline == 'D2':
		Temp = 115
	else:
		Temp = 125
else:
	element = 'Cs'
	if Dline == 'D2':
		Temp = 3
	else:
		Temp = 11

if AgCustomGroundPopulation:
	a = customa
	if a != None:
		b = (1-a)/3
		custpop = [a, b, b, b]
	else:
		custpop = None
else:
	custpop = None

Zoom = False

p_dict={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta,
		'AgNumden': AgNumberDensity, 'Isotope_Combination': 1, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift}
p_dict2={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta,
		 'AgNumden': AgNumberDensity, 'Isotope_Combination': 2, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift}
p_dict3={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta,
		 'AgNumden': AgNumberDensity, 'Isotope_Combination': 0, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift}

[S0,S1,S2,S3,E_out,Ix,Iy]=mf.get_spectra(Detuning,E_in,p_dict,outputs=['S0','S1','S2','S3','E_out','Ix','Iy'])
[S0_1] = mf.get_spectra(Detuning,E_in,p_dict2,outputs=['S0'])
[S0_2] = mf.get_spectra(Detuning,E_in,p_dict3,outputs=['S0'])

line = int(Dline[-1])

def format_sci_tex(num):
	exp = int(np.floor(np.log10(num)))
	coeff = num / 10**exp
	return rf"${coeff:.1f} \times 10^{{{exp}}}$"

##########################################
#Add data to figure
##########################################





c = 2.99792458e8
lambd = 328.1629601



freqerr = 0.01 #0.01 GHz
lambderr = 0.0000022

freq = -np.array(freq)*2 + (c / (lambd)) + deltaf

df_dx = -2
df_dl = -c / (lambd**2)

freq_total_err = np.sqrt(
	(df_dx * freqerr)**2 +
	(df_dl * lambderr)**2
)

freqerr_array = np.full_like(freq, freq_total_err)

###############################################
# THEORETICAL CURVE (already computed above)
###############################################
theory_curve = S0_2[0].real
theory_detuning = Detuning / 1e3   # GHz

###############################################
# EXPERIMENTAL DATA PROCESSING (UPDATED)
###############################################
exp_detuning = freq
x0 = np.mean(exp_detuning)

# Baseline from supplied (b0,b1)

param = exp_detuning - x0

b = fitresults[6:]

# numpy expects highest power first
baseline = np.polyval(b[::-1], param)

# Safety: avoid dividing by zero / negative baseline
if np.any(baseline <= 0):
	raise ValueError("Baseline became non-positive for some detuning points. Check b0/b1 values.")

# Normalised transmission + errors
exp_transmission = trans / baseline
exp_error = np.abs(transerr / baseline)

###############################################
# INTERPOLATE THEORY ONTO EXPERIMENTAL POINTS
###############################################
theory_interp = np.interp(exp_detuning, theory_detuning, theory_curve)

###############################################
# COMPUTE NORMALISED RESIDUALS
###############################################
residuals = (exp_transmission - theory_interp) / exp_error

###############################################
# 2-SUBPLOT FIGURE
###############################################

fig, (ax_main, ax_res) = plt.subplots(
	2, 1,
	figsize=(8, 6),
	sharex=True,
	gridspec_kw={"height_ratios": [3, 1]}
)

###########################################################
# MAIN TRANSMISSION PLOT
###########################################################
colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki', 'orange', 'pink']

for i in range(len(S0)-1):
	if len(S0) >= 7:
		color = colours[1] if i <= 2 else colours[0]
	else:
		color = colours[1] if i <= 1 else colours[0]

	ax_main.plot(theory_detuning, S0[i].real, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
	idx = np.argmin(S0[i].real)
	#ax_main.axvline(theory_detuning[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")

for i in range(len(S0_1)-1):
	if len(S0_1) >= 7:
		color = colours[3] if i <= 2 else colours[2]
	else:
		color = colours[3] if i <= 1 else colours[2]

	ax_main.plot(theory_detuning, S0_1[i].real, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
	idx = np.argmin(S0_1[i].real)
	#ax_main.axvline(theory_detuning[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")


ax_main.fill_between(theory_detuning, theory_curve, 1, color="lightgrey", alpha=0.5)

#plot theory curve and fill

#############################################################################################################

#np.savez("theory_spectrum3.npz",theory_detuning_saved=theory_detuning,theory_curve_saved=theory_curve)

# Load saved theory
dataload = np.load("theory_spectrum.npz")
#dataload = np.load("theory_spectrum2.npz")
x2 = dataload["theory_detuning_saved"]
y2 = dataload["theory_curve_saved"]

# Assume your other curve is already in memory:
# x1 = theory_detuning
# y1 = theory_curve
x1 = theory_detuning
y1 = theory_curve

# Ensure x arrays are increasing (np.interp requires ascending x)
if x1[0] > x1[-1]:
    x1 = x1[::-1]
    y1 = y1[::-1]

if x2[0] > x2[-1]:
    x2 = x2[::-1]
    y2 = y2[::-1]

# Define a common x grid over the overlapping region
xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())

mask1 = (x1 >= xmin) & (x1 <= xmax)
x_common = x1[mask1]  # use x1 spacing in overlap

# Interpolate y2 onto x_common
y2_interp = np.interp(x_common, x2, y2)

# Also get y1 on x_common (already aligned because we used x1[mask1])
y1_common = y1[mask1]

# Plot curves
#ax_main.plot(x1, y1, color="grey", linewidth=1.5, label="Theory (Total)")
#ax_main.plot(x2, y2, color="#b22222", linestyle = "--", linewidth=1, label="Theory (Total) loaded")

# Fill between them
ax_main.fill_between(
    x_common,
    y1_common,
    y2_interp,
    color="#b22222",
    alpha=0.25,
    label="Difference band"
)

##################################################################################################

ax_main.plot(theory_detuning, theory_curve, color="#1f4ed8"   # deep royal blue
, linewidth=2, label="Theory (Total)")

# Experimental data (NOW baseline-normalised using b0,b1)
ax_main.errorbar(
	exp_detuning,
	exp_transmission,
	yerr=exp_error,
	xerr=freqerr_array,
	fmt='x',
	color='black',
	label='Experiment (baseline-normalised)',
	capsize=2
)

ax_main.axhline(1, color='grey', lw=1)

ax_main.text(x=3.9, y=0.4-0.05,
			 s=element+"-D$_{}$".format(line)+r" @{}$\degree$C".format(Temp),
			 fontsize=fontsz+1, ha="right", va="center")
ax_main.text(x=3.9, y=0.32-0.05,
			 s="$N_D/L_{cell}$ = "+format_sci_tex(AgNumberDensity/lcell),
			 fontsize=fontsz-3, ha="right", va="center")

# Optional: show baseline parameters on plot
#ax_main.text(x=3.9, y=0.24-0.05,
#			 s=rf"Baseline: $b_0={b0:.3g}$, $b_1={b1:.3g}$ (per GHz)",
#			 fontsize=fontsz-5, ha="right", va="center")

ax_main.set_ylabel("Transmission")
ax_main.set_ylim([0.15, 1.1])
ax_main.set_xlim([-3.5, 4.5])

###########################################################
# RESIDUAL SUBPLOT
###########################################################
ax_res.axhline(0, color='grey', linewidth=1)
ax_res.errorbar(
	exp_detuning,
	residuals,
	yerr=np.ones_like(residuals),
	xerr=freqerr_array,
	fmt='x',
	color='black',
	markersize=4,
	capsize=2
)

print("residual mean = {}".format(np.mean(residuals)))

ax_res.set_ylabel("Residuals\n(normalised)")
ax_res.set_xlabel("Linear Detuning (GHz)")

plt.subplots_adjust(hspace=0.05)

# -----------------------------
# Add side histogram (Gaussian mini plot) without moving existing axes
# -----------------------------
# Freeze current figure/axes geometry AFTER subplots_adjust
orig_fig_w, orig_fig_h = fig.get_size_inches()
pos_main = ax_main.get_position().frozen()
pos_res  = ax_res.get_position().frozen()

# Extend canvas to the right (inches)
extra_width_in = 2.2
new_fig_w = orig_fig_w + extra_width_in
fig.set_size_inches(new_fig_w, orig_fig_h, forward=True)

def _keep_physical_bbox(pos, old_w, old_h, new_w, new_h):
    """
    Keep an axes the same physical size (inches) after resizing the figure.
    """
    x0_in = pos.x0 * old_w
    y0_in = pos.y0 * old_h
    w_in  = pos.width  * old_w
    h_in  = pos.height * old_h
    return [x0_in / new_w, y0_in / new_h, w_in / new_w, h_in / new_h]

# Re-apply so ax_main and ax_res stay fixed in physical size/location
ax_main.set_position(_keep_physical_bbox(pos_main, orig_fig_w, orig_fig_h, new_fig_w, orig_fig_h))
ax_res .set_position(_keep_physical_bbox(pos_res,  orig_fig_w, orig_fig_h, new_fig_w, orig_fig_h))

# Now place histogram flush to the right of the residuals axis
res_pos = ax_res.get_position().frozen()
pad = 0.01          # small gap
hist_width = 0.10   # fraction of total NEW figure width

ax_hist = fig.add_axes(
    [res_pos.x1 + pad, res_pos.y0, hist_width, res_pos.height],
    sharey=ax_res
)

# Data (residuals should already be ~N(0,1) if normalised correctly)
res = np.asarray(residuals)
res = res[np.isfinite(res)]

# Bin width = 1 sigma
rmin = min(np.floor(res.min()), -4)
rmax = max(np.ceil(res.max()),  4)
edges = np.arange(rmin - 0.5, rmax + 0.5 + 1e-9, 1.0)

# Horizontal histogram
ax_hist.hist(
    res, bins=edges, density=True,
    orientation='horizontal',
    alpha=0.6, edgecolor='black', linewidth=0.8
)

# Ideal N(0,1) PDF overlay
ys = np.linspace(edges[0], edges[-1], 400)
pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ys**2)
ax_hist.plot(pdf, ys, lw=2, color='red')

# Cosmetics
ax_hist.axhline(0, color='grey', lw=1)
ax_hist.set_xlabel("PDF")
ax_hist.set_xlim(left=0)
ax_hist.tick_params(direction='in', top=True, right=True)
plt.setp(ax_hist.get_yticklabels(), visible=False)

#plt.savefig("FinalFig111.png", dpi=600, bbox_inches='tight')

plt.show()