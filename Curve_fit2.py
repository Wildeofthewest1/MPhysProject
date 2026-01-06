import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import rcParams
from libs import main_functions as mf
import os

# =========================================================
# USER SWITCHES
# =========================================================
FIT_POPULATION = True      # <-- MASTER SWITCH
delta_f = 1.09              # fixed frequency offset (GHz)

# =========================================================
# WORKING DIRECTORY
# =========================================================
#os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
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
Detuning = np.linspace(-10, 10, 2000) * 1e3  # MHz
E_in = np.array([1, 0, 0])

element = 'Ag'
Dline = 'D2'
lcell = 25e-3
Bfield = 0
Btheta = 0

# =========================================================
# LOAD EXPERIMENTAL DATA
# =========================================================
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
freq_base = -freq_raw * 2 + (c / lambd)

freqerr = 0.01
freqerr_array = np.full_like(freq_base, freqerr)

# =========================================================
# BASELINE NORMALISATION
# =========================================================
exclude = (freq_base > -5) & (freq_base < 2.5)
mask = ~exclude

coeffs = np.polyfit(freq_base[mask], trans[mask], 1)
baseline = np.polyval(coeffs, freq_base)

exp_detuning = freq_base
exp_transmission = trans / baseline
exp_error = np.abs(transerr / baseline)

# =========================================================
# MODEL WITH POPULATION FITTING
# =========================================================
def theory_model_with_pop(exp_detuning, Temp, AgNumberDensity,
                          a, shift107, shift109):

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
        'AgIsotope_shift': (shift107, shift109)
    }

    [S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
    theory_curve = S0[0].real
    theory_detuning = Detuning / 1e3

    return np.interp(exp_detuning + delta_f, theory_detuning, theory_curve)

# =========================================================
# MODEL WITHOUT POPULATION FITTING
# =========================================================
def theory_model_no_pop(exp_detuning, Temp, AgNumberDensity,
                        shift107, shift109):

    p_dict = {
        'Elem': element,
        'Dline': Dline,
        'T': Temp,
        'lcell': lcell,
        'Bfield': Bfield,
        'Btheta': Btheta,
        'AgNumden': AgNumberDensity,
        'Isotope_Combination': 0,
        'CustomPop': None,  # <-- CRITICAL
        'AgIsotope_shift': (shift107, shift109)
    }

    [S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
    theory_curve = S0[0].real
    theory_detuning = Detuning / 1e3

    return np.interp(exp_detuning + delta_f, theory_detuning, theory_curve)

# =========================================================
# SELECT MODEL AND FIT SETTINGS
# =========================================================
if FIT_POPULATION:
    fit_model = theory_model_with_pop
    p0 = [90, 1.5e16, 0.4, 229.24, -246.76]
    bounds = (
        [20, 1e15, 0.0,  -1000, -1000],
        [200, 5e16, 1.0,   1000,  1000]
    )
else:
    fit_model = theory_model_no_pop
    p0 = [90, 1.5e16, 229.24, -246.76]
    bounds = (
        [20, 1e15, -1000, -1000],
        [200, 5e16,  1000,  1000]
    )

# =========================================================
# PERFORM FIT
# =========================================================
popt, pcov = curve_fit(
    fit_model,
    exp_detuning,
    exp_transmission,
    sigma=exp_error,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds
)

# =========================================================
# UNPACK RESULTS
# =========================================================
if FIT_POPULATION:
    Temp_fit, N_fit, a_fit, is107_fit, is109_fit = popt
else:
    Temp_fit, N_fit, is107_fit, is109_fit = popt
    a_fit = None

errs = np.sqrt(np.diag(pcov))

print("===== FIT RESULTS =====")
print(f"T = {Temp_fit:.2f} ± {errs[0]:.2f} °C")
print(f"N = {N_fit:.3e} ± {errs[1]:.3e}")
if FIT_POPULATION:
    print(f"a = {a_fit:.3f} ± {errs[2]:.3f}")
    i0 = 3
else:
    i0 = 2

print(f"107 shift (MHz) = {is107_fit:.2f} ± {errs[i0]:.2f}")
print(f"109 shift (MHz) = {is109_fit:.2f} ± {errs[i0+1]:.2f}")
print(f"Isotope shift (MHz) = {is107_fit - is109_fit:.2f}")

# =========================================================
# FIT CURVE & RESIDUALS
# =========================================================
fit_curve = fit_model(exp_detuning, *popt)
residuals = (exp_transmission - fit_curve) / exp_error

# =========================================================
# PLOTTING
# =========================================================
fig, (ax_main, ax_res) = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True,
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
