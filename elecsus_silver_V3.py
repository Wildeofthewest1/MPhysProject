import os

# Set BLAS/OpenMP thread limits BEFORE importing numpy/scipy/matplotlib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

print("Logical CPUs available:", os.cpu_count())

# =========================================================
# USER-SET PARALLELISM
# =========================================================
N_JOBS = 12   # try 8 first, then 12, then 16 if you want to benchmark

# =========================================================
# WORKING DIRECTORY
# =========================================================
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

from libs import main_functions as mf_old
from libs import main_functions_V3 as mf_new

# =========================================================
# Text Style
# =========================================================
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator

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

# =========================================================
# SETTINGS
# =========================================================
E_in = np.array([1, 0, 0]) #np.array([1/np.sqrt(2), -1j/np.sqrt(2), 0], dtype=complex)#
DETUNING_SCAN_MHZ = np.linspace(-4000, 4000, 401)

AG107_SHIFT = 229.24
AG109_SHIFT = -246.76

base_p_dict = {
	'Elem': 'Ag',
	'Dline': 'D2',
	'T': 130.23,
	'lcell': 25e-3,
	'Bfield': 0.0,
	'Btheta': 0.0,
	'AgNumden': 1.0e16,
	'Ag107frac': 51.839,
	'Isotope_Combination': 0,
	'AgIsotope_shift': (AG107_SHIFT, AG109_SHIFT),
	'GammaBuf': 0.0,
	'Constrain': True,
	'SubDoppler': False,
	'pump_params': {
		'pol': 'Left',
		'probe_pol': 'Left',
		'eta_pump': 1.0,
		'eta_probe': 1.0,
		'I_pump': 200,#2030.0,
		'I_probe': 13.2,
		'I_sat': 867.0,
	},
	'subdop_params': {
		'Nv': 201,
		'vmax_sigma': 4.0,
		'gamma_transit_Hz': 2.0e4,
		'n_jobs': N_JOBS,
		'gamma_rep_Hz': 1.0e3,
		'gamma_vcc_Hz': 1.0e4
	}
}

# Old model: use CustomPop
p_dict_old = {
	**base_p_dict,
	'CustomPop': [0.45, 0.55/3, 0.55/3, 0.55/3],
	'BoltzmannFactor': False,
	'SubDoppler': False,
}

# New V3 model: do NOT use CustomPop
p_dict_new = {
	**base_p_dict,
	'CustomPop': None,
	'BoltzmannFactor': True,
	'SubDoppler': True,
}

if __name__ == "__main__":
	# =========================================================
	# RUN OLD MODEL
	# =========================================================
	chi_old_plus, chi_old_minus, chi_old_z = mf_old.calc_chi(DETUNING_SCAN_MHZ, p_dict_old)

	S0_old = mf_old.chi_to_S0(
		DETUNING_SCAN_MHZ,
		E_in,
		[chi_old_plus, chi_old_minus, chi_old_z],
		p_dict_old
	)

	# =========================================================
	# RUN NEW V3 MODEL
	# =========================================================
	chi_new_plus, chi_new_minus, chi_new_z, details = mf_new.calc_chi_subdoppler_agd2_dm(
		DETUNING_SCAN_MHZ,
		p_dict_new,
		return_details=True
	)

	S0_new = mf_old.chi_to_S0(
		DETUNING_SCAN_MHZ,
		E_in,
		[chi_new_plus, chi_new_minus, chi_new_z],
		p_dict_new
	)

	# =========================================================
	# AXIS
	# =========================================================
	xGHz = DETUNING_SCAN_MHZ / 1e3

	# =========================================================
	# STYLE
	# =========================================================
	plt.rcParams.update({
		"font.size": 12,
		"axes.labelsize": 13,
		"axes.titlesize": 14,
		"legend.fontsize": 11,
		"xtick.direction": "in",
		"ytick.direction": "in",
		"xtick.top": True,
		"ytick.right": True,
	})

	# =========================================================
	# FIGURE 1: TRANSMISSION
	# =========================================================
	fig, ax = plt.subplots(figsize=(9, 5.5))
	ax.plot(xGHz, S0_old, lw=2.2, label='Old weak-probe model')
	ax.plot(xGHz, S0_new, lw=2.2, ls='--', label='New V3 density-matrix model')

	ax.set_xlabel('Detuning (GHz)')
	ax.set_ylabel('Transmission $S_0$')
	ax.set_title(f'Silver D2 transmission: old vs V3 model (n_jobs={N_JOBS})')
	ax.grid(alpha=0.25)
	ax.legend(frameon=False)
	fig.tight_layout()

	# =========================================================
	# FIGURE 2: IMAGINARY SUSCEPTIBILITY
	# =========================================================
	fig, ax = plt.subplots(figsize=(9, 5.5))
	ax.plot(xGHz, chi_old_plus.imag, lw=2.0, label=r'Old Im$(\chi_+)$')
	ax.plot(xGHz, chi_new_plus.imag, lw=2.0, ls='--', label=r'New Im$(\chi_+)$')

	ax.plot(xGHz, chi_old_minus.imag, lw=2.0, label=r'Old Im$(\chi_-)$')
	ax.plot(xGHz, chi_new_minus.imag, lw=2.0, ls='--', label=r'New Im$(\chi_-)$')

	ax.plot(xGHz, chi_old_z.imag, lw=2.0, label=r'Old Im$(\chi_\pi)$')
	ax.plot(xGHz, chi_new_z.imag, lw=2.0, ls='--', label=r'New Im$(\chi_\pi)$')

	ax.set_xlabel('Detuning (GHz)')
	ax.set_ylabel('Imaginary susceptibility')
	ax.set_title('Imaginary susceptibility: old vs V3 model')
	ax.grid(alpha=0.25)
	ax.legend(frameon=False, ncol=2)
	fig.tight_layout()

	# =========================================================
	# FIGURE 3: REAL SUSCEPTIBILITY
	# =========================================================
	fig, ax = plt.subplots(figsize=(9, 5.5))
	ax.plot(xGHz, chi_old_plus.real, lw=2.0, label=r'Old Re$(\chi_+)$')
	ax.plot(xGHz, chi_new_plus.real, lw=2.0, ls='--', label=r'New Re$(\chi_+)$')

	ax.plot(xGHz, chi_old_minus.real, lw=2.0, label=r'Old Re$(\chi_-)$')
	ax.plot(xGHz, chi_new_minus.real, lw=2.0, ls='--', label=r'New Re$(\chi_-)$')

	ax.plot(xGHz, chi_old_z.real, lw=2.0, label=r'Old Re$(\chi_\pi)$')
	ax.plot(xGHz, chi_new_z.real, lw=2.0, ls='--', label=r'New Re$(\chi_\pi)$')

	ax.set_xlabel('Detuning (GHz)')
	ax.set_ylabel('Real susceptibility')
	ax.set_title('Real susceptibility: old vs V3 model')
	ax.grid(alpha=0.25)
	ax.legend(frameon=False, ncol=2)
	fig.tight_layout()

	plt.show()

# =========================================================
# BRANCH-RESOLVED SUSCEPTIBILITIES
# =========================================================
fig, axs = plt.subplots(3, 2, figsize=(12, 11), sharex=True)

branches = [
	(r'$\chi_+$', chi_old_plus,  chi_new_plus),
	(r'$\chi_-$', chi_old_minus, chi_new_minus),
	(r'$\chi_\pi$', chi_old_z,   chi_new_z),
]

for i, (label, chi_old_b, chi_new_b) in enumerate(branches):
	# Real part
	axs[i, 0].plot(xGHz, chi_old_b.real, lw=2.0, label=f'Old Re({label})')
	axs[i, 0].plot(xGHz, chi_new_b.real, lw=2.0, ls='--', label=f'New Re({label})')
	axs[i, 0].set_ylabel('Real susceptibility')
	axs[i, 0].set_title(f'{label} real part')
	axs[i, 0].grid(alpha=0.25)
	axs[i, 0].legend(frameon=False)

	# Imaginary part
	axs[i, 1].plot(xGHz, chi_old_b.imag, lw=2.0, label=f'Old Im({label})')
	axs[i, 1].plot(xGHz, chi_new_b.imag, lw=2.0, ls='--', label=f'New Im({label})')
	axs[i, 1].set_ylabel('Imaginary susceptibility')
	axs[i, 1].set_title(f'{label} imaginary part')
	axs[i, 1].grid(alpha=0.25)
	axs[i, 1].legend(frameon=False)

axs[-1, 0].set_xlabel('Detuning (GHz)')
axs[-1, 1].set_xlabel('Detuning (GHz)')
fig.suptitle('Branch-resolved susceptibilities: old vs V3 model', y=0.98)
fig.tight_layout()
plt.show()

# =========================================================
# PUMP-INDUCED CHANGE IN EACH BRANCH
# =========================================================
fig, axs = plt.subplots(3, 2, figsize=(12, 11), sharex=True)

for i, (label, chi_old_b, chi_new_b) in enumerate(branches):
	dchi = chi_new_b - chi_old_b

	axs[i, 0].plot(xGHz, dchi.real, lw=2.0)
	axs[i, 0].set_ylabel(r'$\Delta \mathrm{Re}(\chi)$')
	axs[i, 0].set_title(f'{label} pump-induced change: real part')
	axs[i, 0].grid(alpha=0.25)

	axs[i, 1].plot(xGHz, dchi.imag, lw=2.0)
	axs[i, 1].set_ylabel(r'$\Delta \mathrm{Im}(\chi)$')
	axs[i, 1].set_title(f'{label} pump-induced change: imaginary part')
	axs[i, 1].grid(alpha=0.25)

axs[-1, 0].set_xlabel('Detuning (GHz)')
axs[-1, 1].set_xlabel('Detuning (GHz)')
fig.suptitle('Branch-resolved pump-induced susceptibility change', y=0.98)
fig.tight_layout()
plt.show()

xGHz = DETUNING_SCAN_MHZ / 1e3

# -------------------------------------------------
# Build total diagnostic susceptibilities from details
# -------------------------------------------------
chi_pop_left = np.zeros_like(chi_new_plus)
chi_pop_right = np.zeros_like(chi_new_minus)
chi_pop_z = np.zeros_like(chi_new_z)

chi_bare_left = np.zeros_like(chi_new_plus)
chi_bare_right = np.zeros_like(chi_new_minus)
chi_bare_z = np.zeros_like(chi_new_z)

for iso in details['per_isotope']:
	chi_pop_left += details['per_isotope'][iso]['chi_pop_left']
	chi_pop_right += details['per_isotope'][iso]['chi_pop_right']
	chi_pop_z += details['per_isotope'][iso]['chi_pop_z']

	chi_bare_left += details['per_isotope'][iso]['chi_bare_left']
	chi_bare_right += details['per_isotope'][iso]['chi_bare_right']
	chi_bare_z += details['per_isotope'][iso]['chi_bare_z']

fig, axs = plt.subplots(3, 2, figsize=(12, 11), sharex=True)

branches = [
	(r'$\chi_+$', chi_old_plus,  chi_new_plus,  chi_pop_left),
	(r'$\chi_-$', chi_old_minus, chi_new_minus, chi_pop_right),
	(r'$\chi_\pi$', chi_old_z,   chi_new_z,     chi_pop_z),
]

for i, (label, chi_old_b, chi_new_b, chi_pop_b) in enumerate(branches):
	dchi_full = chi_new_b - chi_old_b
	dchi_pop = chi_pop_b - chi_old_b

	axs[i, 0].plot(xGHz, dchi_full.real, lw=2.0, label='Full DM')
	axs[i, 0].plot(xGHz, dchi_pop.real, lw=2.0, ls='--', label='Population diagnostic')
	axs[i, 0].set_ylabel(r'$\Delta \mathrm{Re}(\chi)$')
	axs[i, 0].set_title(f'{label} real part')
	axs[i, 0].grid(alpha=0.25)
	axs[i, 0].legend(frameon=False)

	axs[i, 1].plot(xGHz, dchi_full.imag, lw=2.0, label='Full DM')
	axs[i, 1].plot(xGHz, dchi_pop.imag, lw=2.0, ls='--', label='Population diagnostic')
	axs[i, 1].set_ylabel(r'$\Delta \mathrm{Im}(\chi)$')
	axs[i, 1].set_title(f'{label} imaginary part')
	axs[i, 1].grid(alpha=0.25)
	axs[i, 1].legend(frameon=False)

axs[-1, 0].set_xlabel('Detuning (GHz)')
axs[-1, 1].set_xlabel('Detuning (GHz)')
fig.suptitle('Full DM vs pumped-population diagnostic susceptibility change', y=0.98)
fig.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axs[0].plot(xGHz, chi_pop_left.imag, lw=2.0)
axs[0].set_ylabel(r'Im$(\chi_+)$')
axs[0].set_title(r'Population-based diagnostic: $\chi_+$')
axs[0].grid(alpha=0.25)

axs[1].plot(xGHz, chi_pop_right.imag, lw=2.0)
axs[1].set_ylabel(r'Im$(\chi_-)$')
axs[1].set_title(r'Population-based diagnostic: $\chi_-$')
axs[1].grid(alpha=0.25)

axs[2].plot(xGHz, chi_pop_z.imag, lw=2.0)
axs[2].set_ylabel(r'Im$(\chi_\pi)$')
axs[2].set_xlabel('Detuning (GHz)')
axs[2].set_title(r'Population-based diagnostic: $\chi_\pi$')
axs[2].grid(alpha=0.25)

fig.tight_layout()
plt.show()

mask = (xGHz > -1.2) & (xGHz < 0.6)

plt.figure(figsize=(10, 5))
plt.plot(xGHz[mask], (chi_new_plus.imag - chi_old_plus.imag)[mask], lw=2, label=r'Full $\Delta$Im$(\chi_+)$')
plt.plot(xGHz[mask], (chi_pop_left.imag - chi_old_plus.imag)[mask], lw=2, ls='--', label=r'Pop-based $\Delta$Im$(\chi_+)$')
plt.xlabel('Detuning (GHz)')
plt.ylabel(r'$\Delta$Im$(\chi_+)$')
plt.title(r'Zoomed comparison for $\chi_+$')
plt.grid(alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Build total diagnostic susceptibilities from details
# -------------------------------------------------
chi_pop_left = np.zeros_like(chi_new_plus)
chi_pop_right = np.zeros_like(chi_new_minus)
chi_pop_z = np.zeros_like(chi_new_z)

chi_bare_left = np.zeros_like(chi_new_plus)
chi_bare_right = np.zeros_like(chi_new_minus)
chi_bare_z = np.zeros_like(chi_new_z)

for iso in details['per_isotope']:
	chi_pop_left += details['per_isotope'][iso]['chi_pop_left']
	chi_pop_right += details['per_isotope'][iso]['chi_pop_right']
	chi_pop_z += details['per_isotope'][iso]['chi_pop_z']

	chi_bare_left += details['per_isotope'][iso]['chi_bare_left']
	chi_bare_right += details['per_isotope'][iso]['chi_bare_right']
	chi_bare_z += details['per_isotope'][iso]['chi_bare_z']

# -------------------------------------------------
# Imaginary parts only
# -------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axs[0].plot(xGHz, (chi_new_plus - chi_old_plus).imag, lw=2.0, label='Full DM')
axs[0].plot(xGHz, (chi_pop_left - chi_old_plus).imag, lw=2.0, ls='--', label='Population diagnostic')
axs[0].plot(xGHz, (chi_bare_left - chi_old_plus).imag, lw=2.0, ls=':', label='Bare-probe diagnostic')
axs[0].set_ylabel(r'$\Delta$Im$(\chi_+)$')
axs[0].set_title(r'$\chi_+$ branch')
axs[0].grid(alpha=0.25)
axs[0].legend(frameon=False)

axs[1].plot(xGHz, (chi_new_minus - chi_old_minus).imag, lw=2.0, label='Full DM')
axs[1].plot(xGHz, (chi_pop_right - chi_old_minus).imag, lw=2.0, ls='--', label='Population diagnostic')
axs[1].plot(xGHz, (chi_bare_right - chi_old_minus).imag, lw=2.0, ls=':', label='Bare-probe diagnostic')
axs[1].set_ylabel(r'$\Delta$Im$(\chi_-)$')
axs[1].set_title(r'$\chi_-$ branch')
axs[1].grid(alpha=0.25)
axs[1].legend(frameon=False)

axs[2].plot(xGHz, (chi_new_z - chi_old_z).imag, lw=2.0, label='Full DM')
axs[2].plot(xGHz, (chi_pop_z - chi_old_z).imag, lw=2.0, ls='--', label='Population diagnostic')
axs[2].plot(xGHz, (chi_bare_z - chi_old_z).imag, lw=2.0, ls=':', label='Bare-probe diagnostic')
axs[2].set_ylabel(r'$\Delta$Im$(\chi_\pi)$')
axs[2].set_xlabel('Detuning (GHz)')
axs[2].set_title(r'$\chi_\pi$ branch')
axs[2].grid(alpha=0.25)
axs[2].legend(frameon=False)

fig.tight_layout()
plt.show()

# -------------------------------------------------
# Zoomed comparison around expected narrow-feature region
# -------------------------------------------------
mask = (xGHz > -1.2) & (xGHz < 0.8)

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axs[0].plot(xGHz[mask], (chi_new_plus - chi_old_plus).imag[mask], lw=2.0, label='Full DM')
axs[0].plot(xGHz[mask], (chi_pop_left - chi_old_plus).imag[mask], lw=2.0, ls='--', label='Population diagnostic')
axs[0].plot(xGHz[mask], (chi_bare_left - chi_old_plus).imag[mask], lw=2.0, ls=':', label='Bare-probe diagnostic')
axs[0].set_ylabel(r'$\Delta$Im$(\chi_+)$')
axs[0].set_title(r'Zoomed $\chi_+$ comparison')
axs[0].grid(alpha=0.25)
axs[0].legend(frameon=False)

axs[1].plot(xGHz[mask], (chi_new_minus - chi_old_minus).imag[mask], lw=2.0, label='Full DM')
axs[1].plot(xGHz[mask], (chi_pop_right - chi_old_minus).imag[mask], lw=2.0, ls='--', label='Population diagnostic')
axs[1].plot(xGHz[mask], (chi_bare_right - chi_old_minus).imag[mask], lw=2.0, ls=':', label='Bare-probe diagnostic')
axs[1].set_ylabel(r'$\Delta$Im$(\chi_-)$')
axs[1].set_title(r'Zoomed $\chi_-$ comparison')
axs[1].grid(alpha=0.25)
axs[1].legend(frameon=False)

axs[2].plot(xGHz[mask], (chi_new_z - chi_old_z).imag[mask], lw=2.0, label='Full DM')
axs[2].plot(xGHz[mask], (chi_pop_z - chi_old_z).imag[mask], lw=2.0, ls='--', label='Population diagnostic')
axs[2].plot(xGHz[mask], (chi_bare_z - chi_old_z).imag[mask], lw=2.0, ls=':', label='Bare-probe diagnostic')
axs[2].set_ylabel(r'$\Delta$Im$(\chi_\pi)$')
axs[2].set_xlabel('Detuning (GHz)')
axs[2].set_title(r'Zoomed $\chi_\pi$ comparison')
axs[2].grid(alpha=0.25)
axs[2].legend(frameon=False)

fig.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(xGHz, (chi_new_plus - chi_old_plus).imag, lw=2.0, label='Full DM')
plt.plot(xGHz, (chi_pop_left - chi_old_plus).imag, lw=2.0, ls='--', label='Population diagnostic')
plt.plot(xGHz, (chi_bare_left - chi_old_plus).imag, lw=2.0, ls=':', label='Bare-probe diagnostic')
plt.xlabel('Detuning (GHz)')
plt.ylabel(r'$\Delta$Im$(\chi_+)$')
plt.title(r'Washed-out hole burning check for $\chi_+$')
plt.grid(alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()