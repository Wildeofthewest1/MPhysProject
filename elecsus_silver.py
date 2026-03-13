import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from libs import main_functions as mf


import os
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
#os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

from matplotlib import rcParams
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

#Temp = 200.00#147.53
#AgNumberDensity = 1.671e+16#1.678e+16
#AgIsotopeShift = (229.24,-246.76)#476 #MHz
first = 2
AgCustomGroundPopulation = True
SubDoppler = True

pump_params = {
    'pol': 'Left',
    'probe_pol': 'Left',
    'I_pump': 2030,   # W/m^2
    'I_probe': 13.2,  # W/m^2
    'I_sat': 867,    # W/m^2
	'eta_pump': 0.019
}
subdop_params = {
	'vcc_kernel': 'thermal_reset',
    'Nv': 101,
    'vmax_sigma': 4.0,
    'gamma_transit_Hz': 2.0e4,
    'gamma_vcc_Hz': 1.0e7,
    'vcc_width': 20.0,
}

Temp = 130.23
AgNumberDensity = 1.679e+16
customa = 0.45
AgIsotopeShift = (229.24, -246.76)
TotalIsotopeShift = AgIsotopeShift[0]-AgIsotopeShift[1]

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

p_dict={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 1, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift, "SubDoppler": SubDoppler, 'pump_params': pump_params, 'subdop_params': subdop_params}#, 'Ag107frac':100}
p_dict2={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 2, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift, "SubDoppler": SubDoppler, 'pump_params': pump_params, 'subdop_params': subdop_params}
p_dict3={'Elem':element,'Dline':Dline,'T':Temp,'lcell':lcell,'Bfield':Bfield,'Btheta':Btheta, 'AgNumden': AgNumberDensity, 'Isotope_Combination': 0, 'CustomPop': custpop, 'AgIsotope_shift': AgIsotopeShift, "SubDoppler": SubDoppler, 'pump_params': pump_params, 'subdop_params': subdop_params}

# ---------------------------------------------------------
# Always compute weak-probe spectra
# ---------------------------------------------------------
#p_dict_wp  = dict(p_dict)
#p_dict2_wp = dict(p_dict2)
p_dict3_wp = dict(p_dict3)

#p_dict_wp['SubDoppler'] = False
#p_dict2_wp['SubDoppler'] = False
p_dict3_wp['SubDoppler'] = False

#[S0_wp, S1_wp, S2_wp, S3_wp, E_out_wp, Ix_wp, Iy_wp] = mf.get_spectra(
#	Detuning, E_in, p_dict_wp, outputs=['S0','S1','S2','S3','E_out','Ix','Iy']
#)

#[S0_1_wp] = mf.get_spectra(Detuning, E_in, p_dict2_wp, outputs=['S0'])
[S0_2_wp] = mf.get_spectra(Detuning, E_in, p_dict3_wp, outputs=['S0'])

# ---------------------------------------------------------
# Optionally compute sub-Doppler spectra as well
# ---------------------------------------------------------
if SubDoppler:
	p_dict_sd  = dict(p_dict)
	p_dict2_sd = dict(p_dict2)
	p_dict3_sd = dict(p_dict3)

	p_dict_sd['SubDoppler'] = True
	p_dict2_sd['SubDoppler'] = True
	p_dict3_sd['SubDoppler'] = True

	#[S0_sd] = mf.get_spectra(Detuning, E_in, p_dict_sd, outputs=['S0'])
	#[S0_1_sd] = mf.get_spectra(Detuning, E_in, p_dict2_sd, outputs=['S0'])
	[S0_2_sd] = mf.get_spectra(Detuning, E_in, p_dict3_sd, outputs=['S0'])

	#ChiPlus_sub, ChiMinus_sub, ChiZ_sub, comps = mf.calc_chi_subdoppler_agd2(
	#	Detuning, p_dict3_sd, pump_params, subdop_params, return_components=True
	#)

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


df_dx = -2
df_dl = -c / (lambd**2)


#plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
#plt.yticks([0.4, 0.5, 0.6, 0.7 ,0.8,0.9, 1.0])
#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#plt.xticks([ -8, -6, -4, -2, 0, 2, 4, 6, 8])

#plt.legend()

#plt.show()


###############################################
# THEORETICAL CURVES
###############################################

theory_detuning = Detuning / 1e3   # GHz

theory_curve_wp = S0_2_wp[0].real

if SubDoppler:
	theory_curve_sd = S0_2_sd[0].real

###############################################
# FIGURE
###############################################

fig, ax_main = plt.subplots(figsize=(8, 5))

colours = ['deepskyblue', 'firebrick', 'purple', 'darkkhaki', 'orange', 'pink']

# =========================================================
# WEAK-PROBE COMPONENTS
# =========================================================

"""
for i in range(len(S0_wp) - 1):
	if len(S0_wp) >= 7:
		color = colours[1] if i <= 2 else colours[0]
	else:
		color = colours[1] if i <= 1 else colours[0]

	trans = S0_wp[i].real
	ax_main.plot(
		theory_detuning, trans,
		color=color, linewidth=1.5, alpha=0.8, linestyle="--"
	)

	tmin = np.min(trans)
	idx = np.argmin(trans)
	ax_main.vlines(
		theory_detuning[idx], tmin, 2,
		color=color, linewidth=1.5, alpha=0.8, linestyle="--"
	)

for i in range(len(S0_1_wp) - 1):
	if len(S0_1_wp) >= 7:
		color = colours[3] if i <= 2 else colours[2]
	else:
		color = colours[3] if i <= 1 else colours[2]

	trans = S0_1_wp[i].real
	ax_main.plot(
		theory_detuning, trans,
		color=color, linewidth=1.5, alpha=0.8, linestyle="--"
	)

	tmin = np.min(trans)
	idx = np.argmin(trans)
	ax_main.vlines(
		theory_detuning[idx], tmin, 2,
		color=color, linewidth=1.5, alpha=0.8, linestyle="--"
	)

"""
# Weak-probe total
ax_main.plot(
	theory_detuning, theory_curve_wp,
	color="black", linewidth=1.8, label="Weak probe total"
)


# =========================================================
# SUB-DOPPLER TOTAL + OPTIONAL DIFFERENTIAL CONTRIBUTIONS
# =========================================================
if SubDoppler:
	# Optional: sub-Doppler differential contributions
	colour_index = 0

	# Sub-Doppler total
	ax_main.plot(
		theory_detuning, theory_curve_sd,
		color="grey", linewidth=1.8, label="Sub-Doppler total"
	)

	ax_main.fill_between(
		theory_detuning, theory_curve_sd, 1,
		color="lightgrey", alpha=0.4
	)

# Reference line
ax_main.axhline(1, color='grey', lw=1)

ax_main.set_ylabel("Transmission")
ax_main.set_xlabel("Linear Detuning (GHz)")
ax_main.set_ylim([0, 1.1])
ax_main.set_xlim([-3, 4])

ax_main.legend()

plt.show()