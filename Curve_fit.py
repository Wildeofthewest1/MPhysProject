import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib import rcParams
from libs import main_functions as mf
import os

# =========================================================
# WORKING DIRECTORY
# =========================================================
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

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

# =========================================================
# FIXED PHYSICAL / EXPERIMENTAL SETTINGS
# =========================================================
Detuning = np.linspace(-10, 10, 2000) * 1e3   # MHz (ElecSus grid)
E_in = np.array([1, 0, 0])

element = 'Ag'
Dline = 'D2'
lcell = 25e-3
Bfield = 0
Btheta = 0

# =========================================================
# LOAD EXPERIMENTAL DATA
# =========================================================

choice = 2

if choice == 0:
	frequencies = pd.read_csv("frequencies1.csv")
	transmissions = pd.read_csv("transmission1.csv")
	freq_raw = np.array(frequencies["freq1"])
	trans = np.array(transmissions["Transmission1"])
	transerr = np.array(transmissions["Transmission1err"])
elif choice == 1:
	frequencies = pd.read_csv("frequencies2.csv")
	transmissions = pd.read_csv("transmission2.csv")
	freq_raw = np.array(frequencies["freq2"])
	trans = np.array(transmissions["Transmission2"])
	transerr = np.array(transmissions["Transmission2err"])
elif choice == 2:
	frequencies = pd.read_csv("frequencies3.csv")
	transmissions = pd.read_csv("transmission3.csv")
	freq_raw = np.array(frequencies["freq3"])
	trans = np.array(transmissions["Transmission3"])
	transerr = np.array(transmissions["Transmission3err"])

# =========================================================
# FREQUENCY CALIBRATION (NO OFFSET)
# =========================================================
c = 2.99792458e8
lambd = 328.1629601

# IMPORTANT: no +1.09 here
freq_base = -freq_raw * 2 + (c / lambd)

freqerr = 0.01
freqerr_array = np.full_like(freq_base, freqerr)

# =========================================================
# BASELINE NORMALISATION (EXCLUDE ABSORPTION)
# =========================================================
exclude = (freq_base > -5) & (freq_base < 2.5)
mask = ~exclude

coeffs = np.polyfit(freq_base[mask], trans[mask], 1)
baseline = np.polyval(coeffs, freq_base)

exp_detuning = freq_base
exp_transmission = trans / baseline
exp_error = np.abs(transerr / baseline)

delta_f = 1.09#1

# =========================================================
# THEORY MODEL WITH 4 FREE PARAMETERS
# =========================================================
def theory_model(exp_detuning, Temp, AgNumberDensity, a, AgIsotopeShift107, AgIsotopeShift109):
    """
    ElecSus transmission model with free parameters:
      - Temp (°C)
      - AgNumberDensity
      - ground-state population parameter a
      - 107 shift
	  - 109 shift
    """

    # Physical constraint
    if not (0.0 <= a <= 1.0):
        return np.ones_like(exp_detuning)

    b = (1.0 - a) / 3.0
    custpop = [a, b, b, b]

    p_dict = {
        'Elem': element,
        'Dline': Dline,
        'T': Temp,
        'lcell': lcell,
        'Bfield': Bfield,
        'Btheta': Btheta,
        'AgNumden': AgNumberDensity,
        'Isotope_Combination': 0,
        'CustomPop': custpop,
        'AgIsotope_shift': (AgIsotopeShift107, AgIsotopeShift109)
    }

    [S0] = mf.get_spectra(
        Detuning,
        E_in,
        p_dict,
        outputs=['S0']
    )

    theory_curve = S0[0].real
    theory_detuning = Detuning / 1e3  # GHz

    interp = interp1d(
        theory_detuning,
        theory_curve,
        kind='linear',
        bounds_error=False,
        fill_value=1.0
    )

    # APPLY FREQUENCY OFFSET HERE
    shifted_detuning = exp_detuning + delta_f

    return interp(shifted_detuning)

# =========================================================
# PERFORM FIT
# =========================================================
p0 = [90, 1.5e16, 0.4, 229.24, -246.76]  # T, N, a, 107is, 109is

bounds = (
    [20, 1e15, 0.0, -1000, -1000],
    [200, 5e16, 1.0, 1000, 1000]
)

popt, pcov = curve_fit(
    theory_model,
    exp_detuning,
    exp_transmission,
    sigma=exp_error,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds
)

Temp_fit, N_fit, a_fit, is107_fit, is109_fit = popt
Temp_err, N_err, a_err, is107_err, is109_err = np.sqrt(np.diag(pcov))

print("===== FIT RESULTS =====")
print(f"T        = {Temp_fit:.2f} ± {Temp_err:.2f} °C")
print(f"N        = {N_fit:.3e} ± {N_err:.3e}")
print(f"a        = {a_fit:.3f} ± {a_err:.3f}")
print(f"107 is (MHz) = {is107_fit:.3f} ± {is107_err:.3f}")
print(f"109 is (MHz) = {is109_fit:.3f} ± {is109_err:.3f}")
print(f"Isotope shift (MHz) = {is107_fit - is109_fit:.3f} ± {np.sqrt(is107_err**2+ is109_err**2):.3f}")

# =========================================================
# FIT CURVE & RESIDUALS
# =========================================================
fit_curve = theory_model(exp_detuning, *popt)
residuals = (exp_transmission - fit_curve) / exp_error

# =========================================================
# PLOTTING
# =========================================================
fig, (ax_main, ax_res) = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax_main.errorbar(
    exp_detuning,
    exp_transmission,
    yerr=exp_error,
    xerr=freqerr_array,
    fmt='x',
    color='black',
    label='Experiment'
)

ax_main.plot(
    exp_detuning,
    fit_curve,
    color='red',
    lw=2,
    label='Theory fit'
)

ax_main.axhline(1, color='grey', lw=1)
ax_main.set_ylabel("Transmission")
ax_main.set_ylim([0.2, 1.1])
ax_main.legend()

ax_res.axhline(0, color='grey', lw=1)
ax_res.errorbar(
    exp_detuning,
    residuals,
    yerr=np.ones_like(residuals),
    xerr=freqerr_array,
    fmt='x',
    color='black',
    markersize=4
)

ax_res.set_ylabel("Residuals (normalised)")
ax_res.set_xlabel("Linear Detuning (GHz)")

plt.subplots_adjust(hspace=0.05)
plt.show()
