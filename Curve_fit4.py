# Photodiode_Data.py
# Baseline is fitted ONLY from tails (|detuning| >= 4 GHz) and then frozen.
# delta_f shifts the THEORY axis (not the experimental x-axis).

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.stats as stats

from libs import main_functions as mf


# =========================================================
# USER SWITCHES
# =========================================================
FIT_POPULATION = True          # fit 'a'
FIT_ISOTOPE    = False         # fit (shift107, shift109); if False -> use defaults below
FIT_DELTA_F    = True          # fit global detuning offset delta_f (GHz) applied to THEORY axis
FIT_BASELINE   = True          # fit baseline polynomial from tails (data-only), then freeze

# Isotope shift defaults used when FIT_ISOTOPE = False
Ag107ShiftDefault = 229.24
Ag109ShiftDefault = -246.76

# Choose dataset
curr = 11

# Baseline polynomial degree
BASELINE_ORDER = 4
MAX_BASELINE_ORDER = 15
if not (0 <= BASELINE_ORDER <= MAX_BASELINE_ORDER):
    raise ValueError(f"BASELINE_ORDER must be between 0 and {MAX_BASELINE_ORDER}, got {BASELINE_ORDER}")

# Baseline fitted only on tails:
TAIL_CUT_GHz = 4.0  # |detuning| >= 4 GHz
tailsLeft = 4
tailsRight = 4

# delta_f fixed value if FIT_DELTA_F=False
delta_f_fixed = 0.0  # GHz

# Frequency errorbar for plotting (GHz)
freqerr = 0.01


def get_fitresult_order(baseline_order: int):
    return (
        ['Temp', 'AgNumberDensity', 'a', 'shift107', 'shift109', 'delta_f']
        + [f"b{i}" for i in range(baseline_order + 1)]
    )


# =========================================================
# WORKING DIRECTORY
# =========================================================
# os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

script_dir = Path(__file__).resolve().parent


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
def load_dataset(curr_value: int):
    # Defaults (unused if overwritten below)
    frequencies = pd.read_csv("frequencies3.csv")
    transmissions = pd.read_csv("transmission3.csv")
    baseline_order = BASELINE_ORDER

    if curr_value == 8:
        frequencies = pd.read_csv("frequencies8A.csv")
        transmissions = pd.read_csv("Spec15MicW8A.csv")
        baseline_order = 6
        tailsLeft = 4
        tailsRight = 3

    elif curr_value == 6:
        frequencies = pd.read_csv("frequencies6A.csv")
        transmissions = pd.read_csv("Spec15MicW6A.csv")
        baseline_order = 3
        tailsLeft = 4
        tailsRight = 2

    elif curr_value == 4:
        frequencies = pd.read_csv("frequencies4A.csv")
        transmissions = pd.read_csv("Spec15MicW4A.csv")
        baseline_order = 5
        tailsLeft = 4
        tailsRight = 3

    elif curr_value == 7:
        frequencies = pd.read_csv("frequencies7A.csv")
        transmissions = pd.read_csv("Spec15MicW7A.csv")
        baseline_order = 1
        tailsLeft = 4
        tailsRight = 3

    elif curr_value == 9:
        frequencies = pd.read_csv("frequencies8ASD.csv")
        transmissions = pd.read_csv("SubDoppler8A.csv")
        baseline_order = 2
        tailsLeft = 5
        tailsRight = 3

    elif curr_value == 10:
        frequencies = pd.read_csv("frequencies8A_SD3_NP.csv")
        transmissions = pd.read_csv("SubDoppler3_NP_8A.csv")
        baseline_order = 5
        tailsLeft = 4.5
        tailsRight = 2.5

    elif curr_value == 11:
        frequencies = pd.read_csv("frequencies8A_SD3_WP.csv")
        transmissions = pd.read_csv("SubDoppler3_WP_8A.csv")
        baseline_order = 2
        tailsLeft = 4
        tailsRight = 3

    return frequencies, transmissions, baseline_order, tailsLeft, tailsRight


frequencies, transmissions, BASELINE_ORDER, tailsLeft, tailsRight = load_dataset(curr)
FITRESULT_ORDER = get_fitresult_order(BASELINE_ORDER)


# =========================================================
# HELPERS
# =========================================================
def sort_by_frequency_descending(frequency, transmission, transmission_err):
    if not (len(frequency) == len(transmission) == len(transmission_err)):
        raise ValueError("All arrays must have the same length")

    paired = sorted(
        zip(frequency, transmission, transmission_err),
        key=lambda x: x[0],
        reverse=True
    )
    freq_sorted, trans_sorted, err_sorted = map(list, zip(*paired))
    return np.array(freq_sorted), np.array(trans_sorted), np.array(err_sorted)


def _baseline_poly(x, coeffs):
    """
    Evaluate polynomial baseline: b0 + b1*x + ... + bN*x^N
    coeffs = [b0, b1, ..., bN]
    """
    coeffs = np.asarray(coeffs, dtype=float)
    y = 0.0
    for c in coeffs[::-1]:
        y = y * x + c
    return y


def fit_baseline_from_tails(exp_detuning, exp_transmission, exp_error, x0, order, tail_cut_GHz,tailL,tailR):
    """
    Data-only baseline fit on tails:
        |detuning| >= tail_cut_GHz
    Weighted least squares of polynomial in (detuning - x0).
    """
    xin = exp_detuning - x0
    tail_mask = (exp_detuning <= -tailL) | (exp_detuning >= tailR)

    n_tail = np.count_nonzero(tail_mask)
    if n_tail < (order + 1):
        raise ValueError(
            f"Not enough tail points ({n_tail}) to fit baseline order {order}. "
            f"Lower order or reduce tail_cut_GHz."
        )

    x_tail = xin[tail_mask]
    y_tail = exp_transmission[tail_mask]
    s_tail = exp_error[tail_mask]

    # Vandermonde: [1, x, x^2, ...]
    A = np.vstack([x_tail**i for i in range(order + 1)]).T

    # Weighted LS: minimise sum(((A b - y)/s)^2)
    w = 1.0 / s_tail
    Aw = A * w[:, None]
    yw = y_tail * w

    b, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return b, tail_mask


# =========================================================
# EXTRACT ARRAYS + CLEAN
# =========================================================
freq_raw = np.asarray(frequencies["freq"])
trans = np.asarray(transmissions["Transmission"])
transerr = np.asarray(transmissions["Transmissionerr"])

mask = np.isfinite(freq_raw) & np.isfinite(trans) & np.isfinite(transerr)
freq_raw = freq_raw[mask]
trans = trans[mask]
transerr = transerr[mask]

freq_raw, trans, transerr = sort_by_frequency_descending(freq_raw, trans, transerr)


# =========================================================
# FREQUENCY CALIBRATION (your existing mapping)
# =========================================================
c = 2.99792458e8
lambd = 328.1629601
freq_base = -freq_raw * 2 + (c / lambd)

def dettowav(det):
    return c / (((c / lambd) - det) / 2)

print(dettowav(-2.278))

freqerr_array = np.full_like(freq_base, freqerr)

exp_detuning = freq_base               # GHz axis in your code (as used previously)
exp_transmission = trans
exp_error = np.abs(transerr)

# Numerical centring for baseline polynomial
x0 = np.mean(exp_detuning)


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

    # isotope shifts override
    if (shift107 is not None) and (shift109 is not None):
        p_dict['AgIsotope_shift'] = (shift107, shift109)

    [S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
    theory_curve = S0[0].real
    theory_detuning_GHz = Detuning / 1e3
    return theory_detuning_GHz, theory_curve


# =========================================================
# BASELINE STAGE (data-only, frozen)
# =========================================================
if FIT_BASELINE:
    b_tail, tail_mask = fit_baseline_from_tails(
        exp_detuning=exp_detuning,
        exp_transmission=exp_transmission,
        exp_error=exp_error,
        x0=x0,
        order=BASELINE_ORDER,
        tail_cut_GHz=TAIL_CUT_GHz,
        tailL=tailsLeft,
        tailR=tailsRight
    )
else:
    # If you want a fixed baseline when FIT_BASELINE=False, put your coefficients here.
    # This keeps behaviour explicit.
    b_tail = np.zeros(BASELINE_ORDER + 1)
    b_tail[0] = 1.0
    tail_mask = np.ones_like(exp_detuning, dtype=bool)

baseline_frozen = _baseline_poly(exp_detuning - x0, b_tail)

exp_transmission_norm = exp_transmission / baseline_frozen
exp_error_norm = exp_error / np.abs(baseline_frozen)


# =========================================================
# MODEL BUILDER (THEORY ONLY)
# delta_f shifts THEORY axis: interp(exp_detuning, tG + delta_f, tT)
# =========================================================
def build_model_theory_only():
    param_names = ['Temp', 'AgNumberDensity']
    if FIT_POPULATION:
        param_names += ['a']
    if FIT_ISOTOPE:
        param_names += ['shift107', 'shift109']
    if FIT_DELTA_F:
        param_names += ['delta_f']

    # initial guesses
    p0 = [90.0, 1.5e16]
    if FIT_POPULATION:
        p0 += [0.4]
    if FIT_ISOTOPE:
        p0 += [Ag107ShiftDefault, Ag109ShiftDefault]
    if FIT_DELTA_F:
        p0 += [delta_f_fixed]

    # bounds
    lo = [0.0, 1e15]
    hi = [2000.0, 5e16]
    if FIT_POPULATION:
        lo += [0.0]; hi += [1.0]
    if FIT_ISOTOPE:
        lo += [-10000.0, -10000.0]; hi += [10000.0, 10000.0]
    if FIT_DELTA_F:
        lo += [delta_f_fixed - 2.0]; hi += [delta_f_fixed + 2.0]

    bounds = (lo, hi)

    def model(exp_detuning_in, *params):
        idx = 0
        Temp = params[idx]; idx += 1
        Nden = params[idx]; idx += 1

        a = None
        if FIT_POPULATION:
            a = params[idx]; idx += 1

        if FIT_ISOTOPE:
            shift107 = params[idx]; shift109 = params[idx + 1]
            idx += 2
        else:
            shift107 = Ag107ShiftDefault
            shift109 = Ag109ShiftDefault

        delta_f = delta_f_fixed
        if FIT_DELTA_F:
            delta_f = params[idx]; idx += 1

        tG, tT = _theory_curve_GHz_axis(Temp, Nden, a=a, shift107=shift107, shift109=shift109)

        # Apply delta_f to THEORY axis (not experimental):
        # Evaluate theory at exp_detuning using shifted theory grid.
        theory_interp = np.interp(exp_detuning_in, tG + delta_f, tT)

        return theory_interp

    return model, p0, bounds, param_names


# =========================================================
# FIT THEORY TO BASELINE-CORRECTED DATA
# =========================================================
fit_model, p0, bounds, param_names = build_model_theory_only()

popt, pcov = curve_fit(
    fit_model,
    exp_detuning,
    exp_transmission_norm,
    sigma=exp_error_norm,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds,
    maxfev=20000
)
perr = np.sqrt(np.diag(pcov))

fit_dict = {k: v for k, v in zip(param_names, popt)}
fit_err  = {k: e for k, e in zip(param_names, perr)}


# =========================================================
# PRINTING UTILITIES
# =========================================================
def _fmt(val, err=None):
    if val is None:
        return "not used"
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)
    if err is None or err == 0 or (isinstance(err, (float, np.floating)) and not np.isfinite(err)):
        return f"{val:.6g}"
    if isinstance(err, (np.floating, np.integer)):
        err = float(err)
    return f"{val:.6g} ± {err:.6g}"


def print_summary(fit_dict, fit_err, bcoeffs):
    print("\n===== FIT SUMMARY =====")
    for k in ['Temp', 'AgNumberDensity']:
        print(f"{k:>16s}  [FIT]     = {_fmt(fit_dict.get(k), fit_err.get(k))}")

    if FIT_POPULATION:
        print(f"{'a':>16s}  [FIT]     = {_fmt(fit_dict.get('a'), fit_err.get('a'))}")
    else:
        print(f"{'a':>16s}  [FIXED]   = not used")

    if FIT_ISOTOPE:
        print(f"{'shift107':>16s}  [FIT]     = {_fmt(fit_dict.get('shift107'), fit_err.get('shift107'))}")
        print(f"{'shift109':>16s}  [FIT]     = {_fmt(fit_dict.get('shift109'), fit_err.get('shift109'))}")
    else:
        print(f"{'shift107':>16s}  [DEFAULT] = {_fmt(Ag107ShiftDefault)}")
        print(f"{'shift109':>16s}  [DEFAULT] = {_fmt(Ag109ShiftDefault)}")
        print(f"\nDerived isotope shift (shift107 - shift109) = {Ag107ShiftDefault - Ag109ShiftDefault:.3f} MHz")

    if FIT_DELTA_F:
        print(f"{'delta_f':>16s}  [FIT]     = {_fmt(fit_dict.get('delta_f'), fit_err.get('delta_f'))}")
    else:
        print(f"{'delta_f':>16s}  [FIXED]   = {_fmt(delta_f_fixed)}")

    print("\nBaseline coefficients (TAIL-FIT, frozen; not from theory fit covariance):")
    for i, bi in enumerate(bcoeffs):
        print(f"{f'b{i}':>16s}  [TAIL-FIT]= {_fmt(bi)}")


print_summary(fit_dict, fit_err, b_tail)


# =========================================================
# EVALUATE THEORY + RESIDUALS ON NORMALISED DATA
# =========================================================
theory_only = fit_model(exp_detuning, *popt)
data_norm = exp_transmission_norm
err_norm  = exp_error_norm

residuals_norm = (data_norm - theory_only) / err_norm


# =========================================================
# SAVE BASELINE-CORRECTED PLOT DATA
# =========================================================
output_file = script_dir / f"baseline_corrected_curr{curr}.csv"

df_out = pd.DataFrame({
    "detuning_uv_GHz": np.asarray(exp_detuning, float),
    "Transmission_BaselineCorrected": np.asarray(data_norm, float),
    "TransmissionErr_BaselineCorrected": np.asarray(err_norm, float),
    "Theory_NoBaseline": np.asarray(theory_only, float),
    "Residuals_Norm": np.asarray(residuals_norm, float),
})
df_out.to_csv(output_file, index=False)
print("\nSaved baseline-corrected plot data to:")
print(output_file)


# =========================================================
# SAVE FIT PARAMETERS CSV (including baseline b's)
# =========================================================
def save_fit_parameters_csv(curr, script_dir, fit_dict, fit_err, bcoeffs, baseline_order):
    ordered = (
        ['Temp', 'AgNumberDensity', 'a', 'shift107', 'shift109', 'delta_f'] +
        [f"b{i}" for i in range(0, baseline_order + 1)]
    )

    rows = []
    for name in ordered:
        if name in fit_dict:
            rows.append({"parameter": name, "status": "FIT", "value": fit_dict[name], "error": fit_err.get(name, np.nan)})
        elif name.startswith('b'):
            i = int(name[1:])
            if 0 <= i < len(bcoeffs):
                rows.append({"parameter": name, "status": "TAIL-FIT", "value": float(bcoeffs[i]), "error": 0.0})
            else:
                rows.append({"parameter": name, "status": "TAIL-FIT", "value": np.nan, "error": 0.0})
        else:
            # Defaults / fixed
            if name == 'delta_f' and not FIT_DELTA_F:
                rows.append({"parameter": name, "status": "FIXED", "value": delta_f_fixed, "error": 0.0})
            elif name == 'shift107' and not FIT_ISOTOPE:
                rows.append({"parameter": name, "status": "DEFAULT", "value": Ag107ShiftDefault, "error": 0.0})
            elif name == 'shift109' and not FIT_ISOTOPE:
                rows.append({"parameter": name, "status": "DEFAULT", "value": Ag109ShiftDefault, "error": 0.0})
            elif name == 'a' and not FIT_POPULATION:
                rows.append({"parameter": name, "status": "FIXED", "value": np.nan, "error": 0.0})
            else:
                rows.append({"parameter": name, "status": "N/A", "value": np.nan, "error": np.nan})

    dfp = pd.DataFrame(rows)
    out = script_dir / f"fit_params_curr{curr}.csv"
    dfp.to_csv(out, index=False)
    print("\nSaved fit parameters to:")
    print(out)


save_fit_parameters_csv(
    curr=curr,
    script_dir=script_dir,
    fit_dict=fit_dict,
    fit_err=fit_err,
    bcoeffs=b_tail,
    baseline_order=BASELINE_ORDER
)


# =========================================================
# PLOTTING
# =========================================================
fig, (ax_main, ax_res) = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax_main.errorbar(
    exp_detuning, data_norm,
    yerr=err_norm, xerr=freqerr_array,
    fmt='x', color='black',
    label='Experiment / baseline (tails)'
)
ax_main.plot(
    exp_detuning, theory_only,
    color='red', lw=2,
    label='Theory (fit)'
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

# =========================================================
# SIDE HISTOGRAM (kept from your layout logic)
# =========================================================
orig_fig_w, orig_fig_h = fig.get_size_inches()
pos_main = ax_main.get_position().frozen()
pos_res  = ax_res.get_position().frozen()

extra_width_in = 2.2
new_fig_w = orig_fig_w + extra_width_in
fig.set_size_inches(new_fig_w, orig_fig_h, forward=True)

def _keep_physical_bbox(pos, old_w, old_h, new_w, new_h):
    x0_in = pos.x0 * old_w
    y0_in = pos.y0 * old_h
    w_in  = pos.width  * old_w
    h_in  = pos.height * old_h
    return [x0_in / new_w, y0_in / new_h, w_in / new_w, h_in / new_h]

ax_main.set_position(_keep_physical_bbox(pos_main, orig_fig_w, orig_fig_h, new_fig_w, orig_fig_h))
ax_res.set_position(_keep_physical_bbox(pos_res, orig_fig_w, orig_fig_h, new_fig_w, orig_fig_h))

res_pos = ax_res.get_position().frozen()
pad = 0.01
hist_width = 0.1

ax_hist = fig.add_axes(
    [res_pos.x1 + pad, res_pos.y0, hist_width, res_pos.height],
    sharey=ax_res
)

res = residuals_norm[np.isfinite(residuals_norm)]
rmin = min(np.floor(res.min()), -4)
rmax = max(np.ceil(res.max()), 4)
edges = np.arange(rmin - 0.5, rmax + 0.5 + 1e-9, 1.0)

ax_hist.hist(
    res, bins=edges, density=True,
    orientation='horizontal',
    alpha=0.6, color='blue', edgecolor='black', linewidth=0.8
)

ys = np.linspace(edges[0], edges[-1], 400)
pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ys**2)
ax_hist.plot(pdf, ys, lw=2, color='red')

ax_hist.axhline(0, color='grey', lw=1)
ax_hist.set_xlabel("PDF")
ax_hist.set_xlim(left=0)
ax_hist.tick_params(direction='in', top=True, right=True)
plt.setp(ax_hist.get_yticklabels(), visible=False)


# =========================================================
# RESIDUAL STATS
# =========================================================
N = res.size
p = len(popt)
nu = N - p

chi2 = np.sum(res**2)
chi2_red = chi2 / nu if nu > 0 else np.nan

within_1 = np.mean(np.abs(res) <= 1.0)
within_2 = np.mean(np.abs(res) <= 2.0)
within_3 = np.mean(np.abs(res) <= 3.0)

print("\n===== RESIDUAL COVERAGE CHECK =====")
print(f"N points = {N}")
print(f"DoF (nu) = {nu}")
print(f"Chi^2 = {chi2:.3f}")
print(f"Reduced Chi^2 = {chi2_red:.3f}")
print(f"Fraction with |residual| <= 1σ : {within_1:.3f}  (expected ~0.683)")
print(f"Fraction with |residual| <= 2σ : {within_2:.3f}  (expected ~0.954)")
print(f"Fraction with |residual| <= 3σ : {within_3:.3f}  (expected ~0.997)")

ks = stats.kstest(res, 'norm')
print("\nKS test vs N(0,1):")
print(f"D = {ks.statistic:.3f},  p-value = {ks.pvalue:.3f}")

plt.show()