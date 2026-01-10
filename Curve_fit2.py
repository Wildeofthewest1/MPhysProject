import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import rcParams
from libs import main_functions as mf
import os

# =========================================================
# USER SWITCHES (any combination)
# =========================================================
FIT_POPULATION = True
FIT_ISOTOPE_SHIFTS = True
FIT_DELTA_F = False

# Default used only when FIT_DELTA_F = False
delta_f_fixed = 1.1106  # GHz

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
# FREQUENCY CALIBRATION
# =========================================================
c = 2.99792458e8
lambd = 328.1629601

div = c/lambd

freq_base = div - (2 * freq_raw)

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
# MODEL HELPERS
# =========================================================
def make_pdict(Temp, AgNumberDensity, custpop=None, shifts=None):
    p_dict = {
        'Elem': element,
        'Dline': Dline,
        'T': Temp,
        'lcell': lcell,
        'Bfield': Bfield,
        'Btheta': Btheta,
        'AgNumden': AgNumberDensity,
        'Isotope_Combination': 0,
        'CustomPop': custpop,  # None -> default Boltzmann in FreqStren
    }
    if shifts is not None:
        p_dict['AgIsotope_shift'] = shifts
    # If shifts is None, ElecSus uses p_dict_defaults['AgIsotope_shift']
    return p_dict

def model(exp_detuning, Temp, AgNumberDensity, *extra):
    """
    Parameter order after (Temp, AgNumberDensity):
      [a] [shift107, shift109] [delta_f]
    depending on which FIT_* switches are True.
    """
    idx = 0

    # ---- population ----
    custpop = None
    if FIT_POPULATION:
        a = extra[idx]; idx += 1
        if not (0.0 <= a <= 1.0):
            return np.ones_like(exp_detuning)
        b = (1.0 - a) / 3.0
        custpop = [a, b, b, b]

    # ---- isotope shifts ----
    shifts = None
    if FIT_ISOTOPE_SHIFTS:
        shift107 = extra[idx]; shift109 = extra[idx + 1]
        shifts = (shift107, shift109)
        idx += 2

    # ---- delta_f (GHz) ----
    if FIT_DELTA_F:
        delta_f = extra[idx]
        idx += 1
    else:
        delta_f = delta_f_fixed

    p_dict = make_pdict(Temp, AgNumberDensity, custpop=custpop, shifts=shifts)

    [S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
    theory_curve = S0[0].real
    theory_detuning = Detuning / 1e3  # GHz

    # delta_f is applied on the experimental x-axis (GHz)
    return np.interp(exp_detuning + delta_f, theory_detuning, theory_curve)

# =========================================================
# BUILD p0 AND bounds CONSISTENTLY WITH SWITCHES
# =========================================================
p0 = [90, 1.5e16]
lower = [0, 1e15]
upper = [2000, 5e16]

if FIT_POPULATION:
    p0 += [0.4]
    lower += [0.0]
    upper += [1.0]

if FIT_ISOTOPE_SHIFTS:
    p0 += [229.24, -246.76]      # MHz
    lower += [-10000, -10000]
    upper += [10000, 10000]

if FIT_DELTA_F:
    p0 += [delta_f_fixed]        # GHz initial guess
    lower += [-5.0]              # widen/narrow as sensible for your scan/calibration
    upper += [5.0]

bounds = (lower, upper)

# =========================================================
# PERFORM FIT
# =========================================================
popt, pcov = curve_fit(
    model,
    exp_detuning,
    exp_transmission,
    sigma=exp_error,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds
)
errs = np.sqrt(np.diag(pcov))

# =========================================================
# UNPACK RESULTS (in the same order)
# =========================================================
Temp_fit, N_fit = popt[0], popt[1]
Temp_err, N_err = errs[0], errs[1]

k = 2
a_fit = a_err = None
shift107_fit = shift107_err = None
shift109_fit = shift109_err = None
delta_f_fit = delta_f_err = None

if FIT_POPULATION:
    a_fit, a_err = popt[k], errs[k]
    k += 1

if FIT_ISOTOPE_SHIFTS:
    shift107_fit, shift107_err = popt[k], errs[k]
    shift109_fit, shift109_err = popt[k+1], errs[k+1]
    k += 2

if FIT_DELTA_F:
    delta_f_fit, delta_f_err = popt[k], errs[k]
    k += 1
else:
    delta_f_fit, delta_f_err = delta_f_fixed, 0.0

print("===== FIT RESULTS =====")
print(f"T = {Temp_fit:.2f} ± {Temp_err:.2f} °C")
print(f"N = {N_fit:.3e} ± {N_err:.3e}")

if FIT_POPULATION:
    print(f"a = {a_fit:.3f} ± {a_err:.3f}")
else:
    print("a = fixed (default Boltzmann populations)")

if FIT_ISOTOPE_SHIFTS:
    print(f"107 shift (MHz) = {shift107_fit:.2f} ± {shift107_err:.2f}")
    print(f"109 shift (MHz) = {shift109_fit:.2f} ± {shift109_err:.2f}")
    print(f"Isotope separation (MHz) = {shift107_fit - shift109_fit:.2f}")
else:
    print("Isotope shifts = fixed (ElecSus defaults)")

print(f"delta_f (GHz) = {delta_f_fit:.4f} ± {delta_f_err:.4f}")

# =========================================================
# FIT CURVE & RESIDUALS
# =========================================================
fit_curve = model(exp_detuning, *popt)
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