import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from libs import main_functions as mf
from matplotlib import rcParams

import os
#os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
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

Detuning=np.linspace(-10,10,2000)*1e3 #Detuning range between -10 and 10 GHz. Needs to be input in MHz
E_in=np.array([1,0,0]) #Horizontal Linear Light input. We define E_in = [Ex,Ey,Ez]

choice = 1 #0 = Rb, 1 = Ag, 2 = K, 3 = Na, 4 = Cs

fitresults = (130, 1.679e+16, 0.45, 267.99, 246.76)

#Temp = 200.00#147.53
#AgNumberDensity = 1.671e+16#1.678e+16
#AgIsotopeShift = (229.24,-246.76)#476 #MHz
first = 2
AgCustomGroundPopulation = True

Temp = fitresults[0]
AgNumberDensity = fitresults[1]
customa = fitresults[2]
AgIsotopeShift = (fitresults[3],fitresults[4])

deltaf = 1.09

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
	a = customa#0.444#0.44
	b = (1-a)/3
	custpop = [a, b, b, b]
else:
	custpop = None

Zoom = True
Zoom = False

p_dict={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 1, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift}#, 'Ag107frac':100}
p_dict2={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 2, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift}
p_dict3={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 0, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift}

#A 75 mm cell of natural abundance Rb at 20C. No bfield and hence no angle Btheta between the k-vector and the mag field. 
[S0,S1,S2,S3,E_out,Ix,Iy]=mf.get_spectra(Detuning,E_in,p_dict,outputs=['S0','S1','S2','S3','E_out','Ix','Iy'])

[S0_1] = mf.get_spectra(Detuning,E_in,p_dict2,outputs=['S0'])

[S0_2] = mf.get_spectra(Detuning,E_in,p_dict3,outputs=['S0'])

line = int(Dline[-1])

def format_sci_tex(num):#format long numbers in standard form
	#Return LaTeX-style scientific notation, e.g. 3x10¹⁵.
	exp = int(np.floor(np.log10(num)))
	coeff = num / 10**exp
	return rf"${coeff:.1f} \times 10^{{{exp}}}$"

"""
#plt.figure(figsize=(5, 3.5))
plt.figure(figsize=(8, 5))

colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki', 'orange', 'pink']


for i in range(len(S0)-1):

	if len(S0) >= 7:

		if i <= 2:
			color = colours[1]
		else:
			color = colours[0]
	else:
		
		if i <= 1:
			color = colours[1]
		else:
			color = colours[0]

	label = f'{i}'
	lw = 1.5
	alpha = 0.8

	plt.plot(Detuning / 1e3, S0[i].real, alpha=alpha, color=color, linewidth=lw, label=label, linestyle = "--")

if choice <= 2:
	for i in range(len(S0_1)-1):

		if len(S0_1) >= 7:
			if i <= 2:
				color = colours[3]
			else:
				color = colours[2]
		else:
			if i <= 1:
				color = colours[3]
			else:
				color = colours[2]

		label = f'{i}'
		lw = 1.5
		alpha = 0.8

		plt.plot(Detuning / 1e3, S0_1[i].real, alpha=alpha, color=color, linewidth=lw, label=label, linestyle = "--")

if choice == 2:#Extra for potassium as it has 3 isotopes
	p_dict4={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 3}
	[S0_3] = mf.get_spectra(Detuning,E_in,p_dict4,outputs=['S0'])	
	for i in range(len(S0_3)-1):

		if len(S0_3) >= 7:
			if i <= 2:
				color = colours[4]
			else:
				color = colours[5]
		else:
			if i <= 1:
				color = colours[4]
			else:
				color = colours[5]

		label = f'{i}'
		lw = 1.5
		alpha = 0.8

		plt.plot(Detuning / 1e3, S0_3[i].real, alpha=alpha, color=color, linewidth=lw, label=label, linestyle = "--")

plt.plot(Detuning / 1e3, S0_2[0].real, alpha = 0.8, color='grey', linewidth = 1, label='Total Transmission')
plt.fill_between(Detuning / 1e3, S0_2[0].real, 1, color='lightgrey', alpha=0.5)

plt.axhline(1, color='grey', lw=1)

plt.ylabel("Transmission")
plt.xlabel("Linear Detuning (GHz)")

## Labels (Adding labels to go with the transition level diagram)

adjust = 0.17



Text_y = 1.04 #1.09

plt.text(x=-8, y=Text_y, s=element+"-D$_{}$".format(line), fontsize=fontsz+2, ha = "left", va = "center") ##Ag-D2
#plt.text(x=8, y=Text_y, s=r"{}$degree$C".format(Temp), fontsize=fontsz+2, ha = "right", va = "center") ##Temperature


if choice == 1:
	plt.text(x=-8, y=0.95, s="$N_D$ = "+format_sci_tex(AgNumberDensity), fontsize=fontsz-2, ha = "left", va = "center") ##Temperature

"""

#if ShowTransPlot:
#	plt.text(x=-8, y=0.12, s="$5^2$S$_{1/2}$", fontsize=fontsz, ha = "left", va = "center")#5s2S1/2
#	plt.text(x=-8, y=0.44, s="$5^2$P$_{3/2}$", fontsize=fontsz, ha = "left", va = "center")#5p2P3/2
#	plt.text(x=-3, y=0.28, s="D$_2$", fontsize=fontsz, ha = "left", va = "center")#D2
#	plt.text(x=5.5+adjust, y=0.05, s="0", fontsize=fontsz, ha = "left", va = "center")#F=0
#	plt.text(x=5.5+adjust, y=0.18, s="1", fontsize=fontsz, ha = "left", va = "center")#F=1
#	plt.text(x=5.5+adjust, y=0.37, s="1", fontsize=fontsz, ha = "left", va = "center")#F'=1
#	plt.text(x=5.5+adjust, y=0.49, s="2", fontsize=fontsz, ha = "left", va = "center")#F'=2
#	plt.text(x=6.5+adjust, y=0.12, s="$F$", fontsize=fontsz, ha = "left", va = "center")#F
#	plt.text(x=6.5+adjust, y=0.44, s="$F^'$", fontsize=fontsz, ha = "left", va = "center")#F'
#	# --- Overlay the image ---
	#img = mpimg.imread(r"C:\Users\Matt\Desktop\Lvl_4\Project\SilverD2Diagram109.png")
	#plt.imshow(img, extent=[-5, 5.2+adjust, 0.05, 0.5], aspect='auto', alpha=0.7)


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
lambd = 328.1629601#(22)

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

freqerr = 0.01 #0.01GHz
lambderr = 0.0000022

freq = -np.array(freq)*2 + (c / (lambd)) + deltaf#1.09

df_dx = -2
df_dl = -c / (lambd**2)

freq_total_err = np.sqrt(
	(df_dx * freqerr)**2 +
	(df_dl * lambderr)**2
)

# Make an array matching freq
freqerr_array = np.full_like(freq, freq_total_err)

#print(freq, trans, transerr)
"""
plt.errorbar(freq, trans/(np.max(trans)-adj),
			 yerr=np.abs(transerr/(np.max(trans)-adj)),
			 xerr=freqerr_array,
			 marker='o',label = "data")

if Zoom:
	plt.ylim([0.2, 1.1])
	plt.xlim([-2.5,3])
else:
	plt.ylim([0, 1.1])
	plt.xlim([-8.5,8.5])
"""

#plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
#plt.yticks([0.4, 0.5, 0.6, 0.7 ,0.8,0.9, 1.0])
#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#plt.xticks([ -8, -6, -4, -2, 0, 2, 4, 6, 8])

#plt.savefig(r"TheoryExperiment1.png", dpi=600, bbox_inches='tight')

#plt.legend()

#plt.show()


###############################################
# THEORETICAL CURVE (already computed above)
###############################################

theory_curve = S0_2[0].real  # theoretical total transmission
theory_detuning = Detuning / 1e3   # in GHz

###############################################
# EXPERIMENTAL DATA PROCESSING
###############################################
if first != 2:
	exp_detuning = freq
	exp_transmission = trans / (np.max(trans) - adj)
	exp_error = np.abs(transerr / (np.max(trans) - adj))
else:
	# ----------------------------------------------------
	# Remove region for fitting
	# ----------------------------------------------------
	exclude = (freq > -5) & (freq < 2.5)
	mask = ~exclude

	# Linear fit to CH1
	coeffs1 = np.polyfit(freq[mask], trans[mask], 1)
	m1, c1 = coeffs1
	fit_line1 = np.polyval(coeffs1, freq)

	exp_detuning = freq
	exp_transmission = trans / fit_line1
	exp_error = np.abs(transerr / fit_line1)

###############################################
# INTERPOLATE THEORY ONTO EXPERIMENTAL POINTS
###############################################

# Ensure frequencies match
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
# Plot theoretical curves
colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki', 'orange', 'pink']

for i in range(len(S0)-1):
	if len(S0) >= 7:
		color = colours[1] if i <= 2 else colours[0]
	else:
		color = colours[1] if i <= 1 else colours[0]

	ax_main.plot(theory_detuning, S0[i].real, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
	idx = np.argmin(S0[i].real)
	ax_main.axvline(theory_detuning[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")

# Isotopes
for i in range(len(S0_1)-1):
	if len(S0_1) >= 7:
		color = colours[3] if i <= 2 else colours[2]
	else:
		color = colours[3] if i <= 1 else colours[2]

	ax_main.plot(theory_detuning, S0_1[i].real, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
	idx = np.argmin(S0_1[i].real)
	ax_main.axvline(theory_detuning[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")

# Total theoretical transmission
ax_main.plot(theory_detuning, theory_curve, color="grey", linewidth=1.5, label="Theory (Total)")
ax_main.fill_between(theory_detuning, theory_curve, 1, color="lightgrey", alpha=0.5)

# Experimental data
ax_main.errorbar(
	exp_detuning,
	exp_transmission,
	yerr=exp_error,
	xerr=freqerr_array,
	fmt='x',
	color='black',
	label='Experiment',
	capsize = 2
)

if first == 2:
	ax_main.plot(
		exp_detuning,
		fit_line1/fit_line1,
		label=f"CH1 fit\n y = {m1:.3g}x + {c1:.3g}",
		color='red'
	)

ax_main.axhline(1, color='grey', lw=1)

#ax_main.text(x=3.05, y=0.49-0.05, s=element+"-D$_{}$".format(line), fontsize=fontsz+2, ha = "right", va = "center") ##Ag-D2
ax_main.text(x=3.9, y=0.4-0.05, s=element+"-D$_{}$".format(line)+r" @{}$\degree$C".format(Temp), fontsize=fontsz+1, ha = "right", va = "center") ##Temperature
#ax_main.text(x=3.9, y=0.4-0.05, s="lcell = {} mm".format(lcell*1000), fontsize=fontsz+2, ha = "right", va = "center") ##Temperature
ax_main.text(x=3.9, y=0.32-0.05, s="$N_D/L_{cell}$ = "+format_sci_tex(AgNumberDensity/lcell), fontsize=fontsz-3, ha = "right", va = "center") ##Temperature

ax_main.set_ylabel("Transmission")
ax_main.set_ylim([0.2, 1.1])
ax_main.set_xlim([-3, 4])

#ax_main.legend()

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
	capsize = 2
)

print("residual mean = {}".format(np.mean(residuals)))

ax_res.set_ylabel("Residuals\n(normalised)")
ax_res.set_xlabel("Linear Detuning (GHz)")

#ax_res.set_ylim([-3, 3])  # Adjust as needed

###########################################################
# SAVE & SHOW
###########################################################

plt.subplots_adjust(hspace=0.05)

#plt.savefig("TheoryExperiment_WithResiduals.png", dpi=600, bbox_inches='tight')
plt.show()
