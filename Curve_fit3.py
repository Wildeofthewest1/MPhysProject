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
FIT_POPULATION   = False   # fit 'a'
FIT_ISOTOPE      = True  # fit (shift107, shift109); if False -> use library defaults
FIT_DELTA_F      = False  # fit global detuning offset delta_f (GHz)
FIT_BASELINE     = False#True   # fit baseline polynomial multiplicatively

BASELINE_ORDER   = 1      # 0=constant, 1=linear, 2=quadratic

delta_f_fixed = 1.11171      # GHz, only used if FIT_DELTA_F=False

FITRESULT_ORDER = [
    'Temp',
    'AgNumberDensity',
    'a',
    'shift107',
    'shift109',
    'delta_f'
] + [f"b{i}" for i in range(BASELINE_ORDER + 1)]

# =========================================================
# BASELINE DEFAULTS (used when FIT_BASELINE = False)
# =========================================================
BASELINE_DEFAULTS = {
    'b0': 0.314955,
    'b1': 0.00146153,
}

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
Dline   = 'D2'
lcell   = 25e-3
Bfield  = 0
Btheta  = 0

# =========================================================
# LOAD EXPERIMENTAL DATA
# =========================================================
frequencies    = pd.read_csv("frequencies3.csv")
transmissions  = pd.read_csv("transmission3.csv")

freq_raw = np.array(frequencies["freq3"])
trans    = np.array(transmissions["Transmission3"])
transerr = np.array(transmissions["Transmission3err"])

# =========================================================
# FREQUENCY CALIBRATION (YOUR EXISTING MAPPING)
# =========================================================
c = 2.99792458e8
lambd = 328.1629601
freq_base = -freq_raw * 2 + (c / lambd)

freqerr = 0.01
freqerr_array = np.full_like(freq_base, freqerr)

exp_detuning     = freq_base
exp_transmission = trans
exp_error        = np.abs(transerr)

# Baseline numerical centring
x0 = np.mean(exp_detuning)
x  = exp_detuning - x0

# =========================================================
# OPTIONAL: READ LIBRARY DEFAULT ISOTOPE SHIFTS (recommended)
# =========================================================
# This makes the "DEFAULT" printout explicit.
try:
    # your defaults live inside mf (same module you imported)
    DEFAULT_SHIFT107, DEFAULT_SHIFT109 = mf.p_dict_defaults['AgIsotope_shift']
except Exception:
    DEFAULT_SHIFT107, DEFAULT_SHIFT109 = (np.nan, np.nan)  # fallback

# =========================================================
# THEORY CURVE GENERATOR
# =========================================================
def _theory_curve_GHz_axis(Temp, AgNumberDensity, a=None, shift107=None, shift109=None):
    p_dict = {
        'Elem': element,
        'Dline': Dline,
        'T': Temp,
        'lcell': lcell,
        'Bfield': Bfield,
        'Btheta': Btheta,
        'AgNumden': AgNumberDensity,
        'Isotope_Combination': 0,
    }

    # population
    if a is not None:
        b = (1.0 - a) / 3.0
        p_dict['CustomPop'] = [a, b, b, b]
    else:
        p_dict['CustomPop'] = None

    # isotope shifts: only include key if overriding defaults
    if (shift107 is not None) and (shift109 is not None):
        p_dict['AgIsotope_shift'] = (shift107, shift109)

    [S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
    theory_curve = S0[0].real
    theory_detuning_GHz = Detuning / 1e3
    return theory_detuning_GHz, theory_curve

# =========================================================
# BASELINE MODEL
# =========================================================
def _baseline_poly(x, coeffs):
    b0 = coeffs[0]
    if len(coeffs) == 1:
        return b0
    b1 = coeffs[1]
    if len(coeffs) == 2:
        return b0 + b1 * x
    b2 = coeffs[2]
    return b0 + b1 * x + b2 * x**2

# =========================================================
# MODEL BUILDER
# =========================================================
def build_model():
    param_names = ['Temp', 'AgNumberDensity']

    if FIT_POPULATION:
        param_names += ['a']
    if FIT_ISOTOPE:
        param_names += ['shift107', 'shift109']
    if FIT_DELTA_F:
        param_names += ['delta_f']
    if FIT_BASELINE:
        for i in range(BASELINE_ORDER + 1):
            param_names += [f"b{i}"]

    # initial guesses
    p0 = [90.0, 1.5e16]
    if FIT_POPULATION:
        p0 += [0.4]
    if FIT_ISOTOPE:
        p0 += [DEFAULT_SHIFT107 if np.isfinite(DEFAULT_SHIFT107) else 229.24,
               DEFAULT_SHIFT109 if np.isfinite(DEFAULT_SHIFT109) else -246.76]
    if FIT_DELTA_F:
        p0 += [delta_f_fixed]
    if FIT_BASELINE:
        if BASELINE_ORDER == 0:
            p0 += [1.0]
        elif BASELINE_ORDER == 1:
            p0 += [1.0, 0.0]
        else:
            p0 += [1.0, 0.0, 0.0]

    # bounds
    lo, hi = [], []
    lo += [0.0, 1e15]
    hi += [2000.0, 5e16]
    if FIT_POPULATION:
        lo += [0.0]; hi += [1.0]
    if FIT_ISOTOPE:
        lo += [-10000.0, -10000.0]; hi += [10000.0, 10000.0]
    if FIT_DELTA_F:
        lo += [delta_f_fixed - 2.0]; hi += [delta_f_fixed + 2.0]
    if FIT_BASELINE:
        if BASELINE_ORDER == 0:
            lo += [0.2]; hi += [2.0]
        elif BASELINE_ORDER == 1:
            lo += [0.2, -1.0]; hi += [2.0, 1.0]
        else:
            lo += [0.2, -1.0, -1.0]; hi += [2.0, 1.0, 1.0]

    bounds = (lo, hi)

    def model(exp_detuning_in, *params):
        idx = 0
        Temp = params[idx]; idx += 1
        Nden = params[idx]; idx += 1

        a = None
        if FIT_POPULATION:
            a = params[idx]; idx += 1

        shift107 = shift109 = None
        if FIT_ISOTOPE:
            shift107 = params[idx]; shift109 = params[idx+1]
            idx += 2

        delta_f = delta_f_fixed
        if FIT_DELTA_F:
            delta_f = params[idx]; idx += 1

        bcoeffs = []
        if FIT_BASELINE:
            bcoeffs = list(params[idx:idx + (BASELINE_ORDER + 1)])
            idx += (BASELINE_ORDER + 1)

        tG, tT = _theory_curve_GHz_axis(Temp, Nden, a=a, shift107=shift107, shift109=shift109)
        theory_interp = np.interp(exp_detuning_in + delta_f, tG, tT)

        # --- baseline (always applied) ---
        xin = exp_detuning_in - x0

        if FIT_BASELINE:
            B = _baseline_poly(xin, bcoeffs)
        else:
            # fixed baseline defaults
            bdef = [BASELINE_DEFAULTS[f"b{i}"] for i in range(BASELINE_ORDER + 1)]
            B = _baseline_poly(xin, bdef)

        return B * theory_interp


    return model, p0, bounds, param_names

# =========================================================
# PRINTING UTILITIES
# =========================================================
def _fmt(name, val, err=None):
    # Explicit "not used" marker
    if val is None:
        return "not used"

    # Handle tuples/lists nicely (e.g. (shift107, shift109))
    if isinstance(val, (tuple, list)):
        inner = ", ".join(_fmt(name, v) for v in val)
        return f"({inner})"

    # Convert numpy scalars safely
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)

    # If still not a number, avoid formatting crash
    if not isinstance(val, (int, float)):
        return str(val)

    # Error formatting rules
    if err is None or err == 0 or (isinstance(err, (float, np.floating)) and not np.isfinite(err)):
        return f"{val:.6g}"

    # Convert numpy scalar error safely
    if isinstance(err, (np.floating, np.integer)):
        err = float(err)

    return f"{val:.6g} ± {err:.6g}"


def print_full_summary(fit_dict, fit_err, status_dict, fixed_dict, default_dict):
    """
    Always prints ALL parameters, whether fitted or not.
    status_dict[name] in {"FIT","FIXED","DEFAULT"}
    """

    ordered = (
        ['Temp', 'AgNumberDensity', 'a', 'shift107', 'shift109', 'delta_f'] +
        [f"b{i}" for i in range(0, BASELINE_ORDER + 1)]
    )

    print("\n===== FIT SUMMARY (ALL PARAMETERS) =====")
    for name in ordered:
        if name not in status_dict:
            continue

        st = status_dict[name]

        if st == "FIT":
            val = fit_dict.get(name, None)
            err = fit_err.get(name, None)
            if val is None:
                print(f"{name:>16s}  [FIT]     = (missing)")
            else:
                print(f"{name:>16s}  [FIT]     = {_fmt(name, val, err)}")

        elif st == "FIXED":
            val = fixed_dict.get(name, None)
            if val is None:
                print(f"{name:>16s}  [FIXED]   = (missing)")
            else:
                print(f"{name:>16s}  [FIXED]   = {_fmt(name, val)}")

        elif st == "DEFAULT":
            # If you truly don't know the library default numerically, store None
            val = default_dict.get(name, None)

            # If None or non-finite -> print as library default
            if val is None:
                print(f"{name:>16s}  [DEFAULT] = (library default)")
            elif isinstance(val, (float, np.floating)) and (not np.isfinite(val)):
                print(f"{name:>16s}  [DEFAULT] = (library default)")
            else:
                print(f"{name:>16s}  [DEFAULT] = {_fmt(name, val)}")

    # Print baseline fixed block ONCE (not inside loop)
    if not FIT_BASELINE and BASELINE_DEFAULTS is not None:
        print("\nBaseline (fixed):")
        # Print in order b0, b1, b2... up to BASELINE_ORDER
        for i in range(0, BASELINE_ORDER + 1):
            k = f"b{i}"
            if k in BASELINE_DEFAULTS:
                print(f"{k:>16s}  [FIXED]   = {BASELINE_DEFAULTS[k]:.6g}")
            else:
                print(f"{k:>16s}  [FIXED]   = (missing default)")

    # Derived isotope shift (only if both values resolve to finite numbers)
    def _resolve(name):
        if status_dict.get(name) == "FIT":
            return fit_dict.get(name, np.nan)
        if status_dict.get(name) == "FIXED":
            return fixed_dict.get(name, np.nan)
        if status_dict.get(name) == "DEFAULT":
            return default_dict.get(name, np.nan)
        return np.nan

    s107 = _resolve('shift107')
    s109 = _resolve('shift109')
    if np.isfinite(s107) and np.isfinite(s109):
        print(f"\nDerived isotope shift (shift107 - shift109) = {s107 - s109:.3f} MHz")


def print_fitresults_tuples(status, fit_dict, fit_err, fixed_dict, default_dict):
    """
    Prints paste-ready:
        fitresults = (...)
        fitresultsErrors = (...)
    following the canonical FITRESULT_ORDER.
    """

    values = []
    errors = []

    for name in FITRESULT_ORDER:
        if name not in status:
            continue

        st = status[name]

        if st == "FIT":
            values.append(fit_dict.get(name, None))
            errors.append(fit_err.get(name, 0.0))

        elif st == "FIXED":
            values.append(fixed_dict.get(name, None))
            errors.append(0.0)

        elif st == "DEFAULT":
            # If you don't know the library default numeric value, keep None
            values.append(default_dict.get(name, None))
            errors.append(0.0)

    def fmt(v):
        # Handle numpy scalars cleanly
        if isinstance(v, (np.floating, float)):
            return f"{float(v):.6g}"
        if isinstance(v, (np.integer, int)):
            return str(int(v))
        if v is None:
            return "None"
        # handle tuples/lists (e.g. if you ever store AgIsotope_shift as a tuple)
        if isinstance(v, (tuple, list)):
            inner = ", ".join(fmt(x) for x in v)
            return f"({inner})"
        return "None"

    values_str = ", ".join(fmt(v) for v in values)
    errors_str = ", ".join(fmt(e) for e in errors)

    print("\nPaste-ready fitresults:")
    print(f"fitresults = ({values_str})")

    print("\nPaste-ready fitresultsErrors:")
    print(f"fitresultsErrors = ({errors_str})")


# =========================================================
# FIT
# =========================================================
fit_model, p0, bounds, param_names = build_model()

popt, pcov = curve_fit(
    fit_model,
    exp_detuning,
    exp_transmission,
    sigma=exp_error,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds,
    maxfev=20000
)

perr = np.sqrt(np.diag(pcov))

# --- Build dictionaries for printing ---
fit_dict = {k: v for k, v in zip(param_names, popt)}
fit_err  = {k: e for k, e in zip(param_names, perr)}

status = {}
fixed  = {}
default = {}

# Always present physics
status['Temp'] = "FIT"; status['AgNumberDensity'] = "FIT"

# population
if FIT_POPULATION:
    status['a'] = "FIT"
else:
    status['a'] = "FIXED"
    fixed['a'] = None  # interpret as "Boltzmann/unconstrained" in your model

# isotope shifts
if FIT_ISOTOPE:
    status['shift107'] = "FIT"
    status['shift109'] = "FIT"
else:
    status['shift107'] = "DEFAULT"
    status['shift109'] = "DEFAULT"
    default['shift107'] = DEFAULT_SHIFT107
    default['shift109'] = DEFAULT_SHIFT109

# delta_f
if FIT_DELTA_F:
    status['delta_f'] = "FIT"
else:
    status['delta_f'] = "FIXED"
    fixed['delta_f'] = delta_f_fixed

# baseline coefficients
if FIT_BASELINE:
    for i in range(BASELINE_ORDER + 1):
        status[f"b{i}"] = "FIT"
else:
    for i in range(BASELINE_ORDER + 1):
        k = f"b{i}"
        status[k] = "FIXED"
        fixed[k] = BASELINE_DEFAULTS[k]


print_full_summary(fit_dict, fit_err, status, fixed, default)

print_fitresults_tuples(status, fit_dict, fit_err, fixed, default)

# =========================================================
# FIT CURVE & NORMALISED PLOT (as you had)
# =========================================================
fit_curve = fit_model(exp_detuning, *popt)
residuals = (exp_transmission - fit_curve) / exp_error

# Separate baseline + theory for normalised plot
def compute_theory_interp(exp_detuning_in, pars):
    Temp = pars['Temp']
    Nden = pars['AgNumberDensity']
    a = pars.get('a', None) if FIT_POPULATION else None

    if FIT_ISOTOPE:
        shift107 = pars['shift107']; shift109 = pars['shift109']
    else:
        shift107 = shift109 = None  # -> defaults

    delta_f = pars['delta_f'] if FIT_DELTA_F else delta_f_fixed

    tG, tT = _theory_curve_GHz_axis(Temp, Nden, a=a, shift107=shift107, shift109=shift109)
    return np.interp(exp_detuning_in + delta_f, tG, tT)

def compute_baseline(exp_detuning, pars):
    """
    Compute multiplicative baseline B(x) on the experimental axis.
    If FIT_BASELINE is False, use fixed default baseline coefficients.
    """
    xin = exp_detuning - x0

    if FIT_BASELINE:
        bcoeffs = [pars[f"b{i}"] for i in range(BASELINE_ORDER + 1)]
    else:
        # Use fixed defaults
        bcoeffs = []
        for i in range(BASELINE_ORDER + 1):
            key = f"b{i}"
            if key not in BASELINE_DEFAULTS:
                raise ValueError(f"Missing default value for {key}")
            bcoeffs.append(BASELINE_DEFAULTS[key])

    return _baseline_poly(xin, bcoeffs)


pars = fit_dict
baseline_fit = compute_baseline(exp_detuning, pars)
theory_only  = compute_theory_interp(exp_detuning, pars)

data_norm = exp_transmission / baseline_fit
err_norm  = exp_error / np.abs(baseline_fit)

residuals_norm = (data_norm - theory_only) / err_norm

fig, (ax_main, ax_res) = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax_main.errorbar(
    exp_detuning, data_norm,
    yerr=err_norm, xerr=freqerr_array,
    fmt='x', color='black',
    label='Experiment / baseline'
)
ax_main.plot(
    exp_detuning, theory_only,
    color='red', lw=2,
    label='Theory (no baseline)'
)

ax_main.axhline(1, color='grey', lw=1)
ax_main.set_ylabel("Transmission (baseline-normalised)")
ax_main.set_ylim([0.2, 1.1])
ax_main.legend()

ax_res.axhline(0, color='grey', lw=1)
ax_res.errorbar(
    exp_detuning, residuals_norm,
    yerr=np.ones_like(residuals_norm),
    xerr=freqerr_array,
    fmt='x', color='black',
    markersize=4
)
ax_res.set_ylabel("Residuals (normalised)")
ax_res.set_xlabel("Linear Detuning (GHz)")

plt.subplots_adjust(hspace=0.05)
plt.show()