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
fitresultsErrors = (3.59048, 2.69753e-12, 0.00155848, 0, 0, 0.00222233, 0.000414683, 0.000101566)

#fitresults = (402.255, 1.46035e+16, None, 229.24, -246.76, 1.04713, 0.314955, 0.00146153)
#fitresultsErrors = (3.7081, 6.15048e-13, 0, 0, 0, 0.00257602, 0, 0)

first = 2
AgCustomGroundPopulation = True

Temp = fitresults[0]
AgNumberDensity = fitresults[1]
customa = fitresults[2]
AgIsotopeShift = (fitresults[3],fitresults[4])

# Baseline params (manual inputs, used to normalise plotted data)
b0 = fitresults[6]
b1 = fitresults[7]

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

import pandas as pd
frequencies = pd.read_csv("frequencies1.csv")
frequencies2 = pd.read_csv("frequencies2.csv")
frequencies3 = pd.read_csv("frequencies3.csv")
transmissions1 = pd.read_csv("transmission1.csv")
transmissions2 = pd.read_csv("transmission2.csv")
transmissions3 = pd.read_csv("transmission3.csv")

c = 2.99792458e8
lambd = 328.1629601

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
baseline = b0 + b1*(exp_detuning - x0)

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
	ax_main.axvline(theory_detuning[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")

for i in range(len(S0_1)-1):
	if len(S0_1) >= 7:
		color = colours[3] if i <= 2 else colours[2]
	else:
		color = colours[3] if i <= 1 else colours[2]

	ax_main.plot(theory_detuning, S0_1[i].real, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
	idx = np.argmin(S0_1[i].real)
	ax_main.axvline(theory_detuning[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")

ax_main.plot(theory_detuning, theory_curve, color="grey", linewidth=1.5, label="Theory (Total)")
ax_main.fill_between(theory_detuning, theory_curve, 1, color="lightgrey", alpha=0.5)

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
ax_main.set_ylim([0.2, 1.1])
ax_main.set_xlim([-3, 4])

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

#plt.savefig("FitGood.png", dpi=600, bbox_inches='tight')

plt.show()