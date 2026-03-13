import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# =========================================================
# PLOTTING STYLE
# =========================================================
fontsz = 16
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['axes.grid'] = False

# =========================================================
# WORKING DIRECTORY
# =========================================================
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

from libs import main_functions_V2 as mf

# =========================================================
# SETTINGS
# =========================================================
E_in = np.array([1, 0, 0])

DETUNING_SCAN_MHZ = np.linspace(-4000, 4000, 201)
AG107_SHIFT = 229.24
AG109_SHIFT = -246.76

p_dict = {
    'Elem': 'Ag',
    'Dline': 'D2',
    'T': 130.23,
    'lcell': 25e-3,
    'Bfield': 0,#1e-4,
    'Btheta': 0.0,
    'AgNumden': 1.0e+16,
    'Ag107frac': 51.839,
    'Isotope_Combination': 0,
    'AgIsotope_shift': (AG107_SHIFT, AG109_SHIFT),
    'SubDoppler': True,
    'pump_params': {
        'pol': 'Left',
        'probe_pol': 'Left',
        'eta_pump': 1.0,
        'eta_probe': 1.0,
        'I_pump': 2030.0,
        'I_probe': 13.2,
        'I_sat': 867.0,
    },
    'subdop_params': {
        'Nv': 81,
        'vmax_sigma': 4.0,
        'gamma_transit_Hz': 2.0e4,
        'gamma_vcc_Hz': 1.0e2,
        'vcc_width': 20.0,
        'vcc_kernel': 'cusp',
    }
}

# =========================================================
# RUN FULL POPULATION-BASED CHI SCAN
# =========================================================
chi_plus, chi_minus, chi_z, details = mf.calc_chi_subdoppler_agd2_population_scan(
    DETUNING_SCAN_MHZ,
    p_dict,
    include_probe_pumping=False,
    return_details=True
)

scan = details['scan']

# =========================================================
# BUILD S0 FROM CHI
# =========================================================
S0 = mf.chi_to_S0(
    DETUNING_SCAN_MHZ,
    E_in,
    [chi_plus, chi_minus, chi_z],
    p_dict
)

print("chi_plus shape :", chi_plus.shape)
print("chi_minus shape:", chi_minus.shape)
print("chi_z shape    :", chi_z.shape)
print("S0 shape       :", S0.shape)
print("S0 min/max     :", np.min(S0), np.max(S0))

# =========================================================
# BUILD OLD MODEL (NO SUB-DOPPLER)
# =========================================================
p_dict_old = dict(p_dict)
p_dict_old['SubDoppler'] = False
p_dict_old.pop('pump_params', None)
p_dict_old.pop('subdop_params', None)

[S0_old] = mf.get_spectra(
    DETUNING_SCAN_MHZ,
    E_in,
    p_dict_old,
    outputs=['S0']
)

# get_spectra returns shape (1, N) here, so flatten it
S0_old = np.asarray(S0_old).squeeze()

print("S0_old shape   :", S0_old.shape)
print("S0_old min/max :", np.min(S0_old), np.max(S0_old))

# =========================================================
# PLOTS
# =========================================================
xGHz = DETUNING_SCAN_MHZ / 1e3

def style_axis(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# --- total integrated populations ---
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(xGHz, scan['ground_total'], lw=2, label='Integrated ground population')
ax.plot(xGHz, scan['excited_total'], lw=2, label='Integrated excited population')
ax.plot(xGHz, scan['population_total'], 'k--', lw=2, label='Integrated total')
style_axis(
    ax,
    'Detuning (GHz)',
    'Integrated population',
    'Total integrated populations'
)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# --- isotope-resolved totals ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for label in scan['ground_by_isotope']:
    ax.plot(xGHz, scan['ground_by_isotope'][label], lw=2, label=f'Ground {label}')
for label in scan['excited_by_isotope']:
    ax.plot(xGHz, scan['excited_by_isotope'][label], '--', lw=2, label=f'Excited {label}')
style_axis(
    ax,
    'Detuning (GHz)',
    'Integrated population',
    'Isotope-resolved populations'
)
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
plt.show()

# --- real susceptibility ---
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(xGHz, chi_plus.real, lw=2, label=r'Re$(\chi_+)$')
ax.plot(xGHz, chi_minus.real, lw=2, label=r'Re$(\chi_-)$')
ax.plot(xGHz, chi_z.real, lw=2, label=r'Re$(\chi_z)$')
style_axis(
    ax,
    'Detuning (GHz)',
    'Real susceptibility',
    'Real susceptibility components'
)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# --- imaginary susceptibility ---
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(xGHz, chi_plus.imag, lw=2, label=r'Im$(\chi_+)$')
ax.plot(xGHz, chi_minus.imag, lw=2, label=r'Im$(\chi_-)$')
ax.plot(xGHz, chi_z.imag, lw=2, label=r'Im$(\chi_z)$')
style_axis(
    ax,
    'Detuning (GHz)',
    'Imaginary susceptibility',
    'Imaginary susceptibility components'
)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# --- transmission comparison ---
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(xGHz, S0, lw=2, label=r'Population sub-Doppler $S_0$')
ax.plot(xGHz, S0_old, lw=2, ls='--', label=r'Old weak-probe $S_0$')
style_axis(
    ax,
    'Detuning (GHz)',
    'Transmission',
    'Transmission comparison'
)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()