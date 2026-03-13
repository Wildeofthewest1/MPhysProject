import numpy as np
import matplotlib.pyplot as plt
import os

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

# =========================================================
# SETTINGS
# =========================================================
E_in = np.array([1, 0, 0])
DETUNING_SCAN_MHZ = np.linspace(-2000, 2000, 3)

AG107_SHIFT = 229.24
AG109_SHIFT = -246.76

p_dict = {
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
    'SubDoppler': False,   # old model is weak-probe / legacy
    'pump_params': {
        'pol': 'Left',
        'probe_pol': 'Left',
        'eta_pump': 0.0029,
        'eta_probe': 1,
        'I_pump': 2030.0,      # set as needed
        'I_probe': 13.2,
        'I_sat': 867.0,
    },
    'subdop_params': {
        'Nv': 81,
        'vmax_sigma': 4.0,
        'gamma_transit_Hz': 2.0e4,
    }
}

# =========================================================
# RUN OLD MODEL
# =========================================================
chi_old_plus, chi_old_minus, chi_old_z = mf_old.calc_chi(DETUNING_SCAN_MHZ, p_dict)

S0_old = mf_old.chi_to_S0(
    DETUNING_SCAN_MHZ,
    E_in,
    [chi_old_plus, chi_old_minus, chi_old_z],
    p_dict
)

# =========================================================
# RUN NEW V3 MODEL
# =========================================================
chi_new_plus, chi_new_minus, chi_new_z, details = mf_new.calc_chi_subdoppler_agd2_dm(
    DETUNING_SCAN_MHZ,
    p_dict,
    return_details=True
)

S0_new = mf_old.chi_to_S0(
    DETUNING_SCAN_MHZ,
    E_in,
    [chi_new_plus, chi_new_minus, chi_new_z],
    p_dict
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
ax.set_title('Silver D2 transmission: old vs V3 model')
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