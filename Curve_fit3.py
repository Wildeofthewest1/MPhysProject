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
FIT_POPULATION   = True # fit 'a'
FIT_ISOTOPE      = False#True  # fit (shift107, shift109); if False -> use library defaults
FIT_DELTA_F      =  True  # fit global detuning offset delta_f (GHz)
FIT_BASELINE     = True  # fit baseline polynomial multiplicatively

Ag107ShiftDefault = 229.24#400#1000.0
Ag109ShiftDefault = -246.76#-Ag107ShiftDefault #0.0

curr = 8

BASELINE_ORDER   = 4   # 0..7 polynomial degree
MAX_BASELINE_ORDER = 15
if not (0 <= BASELINE_ORDER <= MAX_BASELINE_ORDER):
	raise ValueError(f"BASELINE_ORDER must be between 0 and {MAX_BASELINE_ORDER}, got {BASELINE_ORDER}")

delta_f_fixed = 0#1.11171      # GHz, only used if FIT_DELTA_F=False

def get_fitresult_order():
	return [
		'Temp',
		'AgNumberDensity',
		'a',
		'shift107',
		'shift109',
		'delta_f'
	] + [f"b{i}" for i in range(BASELINE_ORDER + 1)]

# =========================================================
# WORKING DIRECTORY
# =========================================================
#os.chdir(r"C:\Users\Matt\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
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
#Old Data
frequencies    = pd.read_csv("frequencies3.csv")
transmissions  = pd.read_csv("transmission3.csv")

#New Data
#requencies    = pd.read_csv("frequencies5.csv")
#transmissions  = pd.read_csv("Spec30MicroWatts.csv")
#frequencies    = pd.read_csv("frequencies5.csv")
#transmissions  = pd.read_csv("Spec15MicroWatts.csv")

if curr == 8:
	frequencies    = pd.read_csv("frequencies8A.csv")
	transmissions  = pd.read_csv("Spec15MicW8A.csv")
	BASELINE_ORDER = 4

elif curr == 6:
	frequencies    = pd.read_csv("frequencies6A.csv")
	transmissions  = pd.read_csv("Spec15MicW6A.csv")
	BASELINE_ORDER = 2

elif curr == 4:
	frequencies    = pd.read_csv("frequencies4A.csv")
	transmissions  = pd.read_csv("Spec15MicW4A.csv")
	BASELINE_ORDER = 2
	print(transmissions)

elif curr == 7:
	frequencies    = pd.read_csv("frequencies7A.csv")
	transmissions  = pd.read_csv("Spec15MicW7A.csv")
	BASELINE_ORDER = 2

elif curr == 9:
	frequencies    = pd.read_csv("frequencies8ASD.csv")
	transmissions  = pd.read_csv("SubDoppler8A.csv")
	BASELINE_ORDER = 2

elif curr == 10:
	frequencies    = pd.read_csv("frequencies8A_SD3_NP.csv")
	transmissions  = pd.read_csv("SubDoppler3_NP_8A.csv")
	BASELINE_ORDER = 2

elif curr == 11:
	frequencies    = pd.read_csv("frequencies8A_SD3_WP.csv")
	transmissions  = pd.read_csv("SubDoppler3_WP_8A.csv")
	BASELINE_ORDER = 2

FITRESULT_ORDER = get_fitresult_order()

# =========================================================
# BASELINE DEFAULTS (used when FIT_BASELINE = False)
# =========================================================
BASELINE_DEFAULTS = {
	'b0': 0.31512,#0.314427,
	'b1': 0.00140447#0.00135486,
}

# Auto-fill any missing baseline defaults up to BASELINE_ORDER with 0.0
for i in range(BASELINE_ORDER + 1):
	BASELINE_DEFAULTS.setdefault(f"b{i}", 0.0)


def sort_by_frequency_descending(frequency, transmission, transmission_err):
	if not (len(frequency) == len(transmission) == len(transmission_err)):
		raise ValueError("All arrays must have the same length")

	paired = sorted(
		zip(frequency, transmission, transmission_err),
		key=lambda x: x[0],
		reverse=True
	)

	freq_sorted, trans_sorted, err_sorted = map(list, zip(*paired))

	return (
		np.array(freq_sorted),
		np.array(trans_sorted),
		np.array(err_sorted),
	)


# Extract arrays
freq_raw = np.array(frequencies["freq"])
trans    = np.array(transmissions["Transmission"])
transerr = np.array(transmissions["Transmissionerr"])


# ===== Remove NaN and Inf =====
mask = (
	np.isfinite(freq_raw) &
	np.isfinite(trans) &
	np.isfinite(transerr)
)

freq_raw = freq_raw[mask]
trans    = trans[mask]
transerr = transerr[mask]


# ===== Sort after cleaning =====
freq_raw, trans, transerr = sort_by_frequency_descending(
	freq_raw, trans, transerr
)
# =========================================================
# FREQUENCY CALIBRATION (YOUR EXISTING MAPPING)
# =========================================================
c = 2.99792458e8
lambd = 328.1629601
freq_base = -freq_raw * 2 + (c / lambd)

def dettowav(det):
	res = c/(((c / lambd) - det)/2)
	return res

print(dettowav(-2.278))

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
	DEFAULT_SHIFT107, DEFAULT_SHIFT109 = Ag107ShiftDefault, Ag109ShiftDefault
	#mf.p_dict_defaults['AgIsotope_shift']
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
	"""
	Evaluate polynomial baseline: b0 + b1*x + ... + bN*x^N
	coeffs = [b0, b1, ..., bN]
	"""
	coeffs = np.asarray(coeffs, dtype=float)
	# Horner’s method (stable + fast)
	y = 0.0
	for c in coeffs[::-1]:
		y = y * x + c
	return y

# =========================================================
# MODEL BUILDER
# =========================================================

def build_model():
	param_names = ['Temp', 'log10_AgNumberDensity']

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
	p0 = [90.0, np.log10(1.5e16)]
	if FIT_POPULATION:
		p0 += [0.4]
	if FIT_ISOTOPE:
		p0 += [DEFAULT_SHIFT107 if np.isfinite(DEFAULT_SHIFT107) else Ag107ShiftDefault,
			   DEFAULT_SHIFT109 if np.isfinite(DEFAULT_SHIFT109) else Ag109ShiftDefault]
	if FIT_DELTA_F:
		p0 += [delta_f_fixed]
	if FIT_BASELINE:
		# b0 ~ 1, others ~ 0
		p0 += [1.0] + [0.0] * BASELINE_ORDER

	# bounds
	lo, hi = [], []
	lo += [0.0, 15.0]
	hi += [2000.0, np.log10(5e16)]
	if FIT_POPULATION:
		lo += [0.0]; hi += [1.0]
	if FIT_ISOTOPE:
		lo += [-10000.0, -10000.0]; hi += [10000.0, 10000.0]
	if FIT_DELTA_F:
		lo += [delta_f_fixed - 2.0]; hi += [delta_f_fixed + 2.0]
	if FIT_BASELINE:
		lo += [0.2] + [-1.0] * BASELINE_ORDER
		hi += [2.0] + [ 1.0] * BASELINE_ORDER

	bounds = (lo, hi)

	def model(exp_detuning_in, *params):
		idx = 0
		Temp = params[idx]; idx += 1
		log10N = params[idx]; idx += 1
		Nden = 10**log10N

		a = None
		if FIT_POPULATION:
			a = params[idx]; idx += 1

		shift107 = shift109 = None
		if FIT_ISOTOPE:
			shift107 = params[idx]; shift109 = params[idx+1]
			idx += 2
		else:
			shift107 = Ag107ShiftDefault
			shift109 = Ag109ShiftDefault

		delta_f = delta_f_fixed
		if FIT_DELTA_F:
			delta_f = params[idx]; idx += 1

		bcoeffs = []
		if FIT_BASELINE:
			bcoeffs = list(params[idx:idx + (BASELINE_ORDER + 1)])
			idx += (BASELINE_ORDER + 1)

		tG, tT = _theory_curve_GHz_axis(Temp, Nden, a=a, shift107=shift107, shift109=shift109)
		theory_interp = np.interp(exp_detuning_in + delta_f, tG, tT)

		xin = exp_detuning_in - x0

		if FIT_BASELINE:
			B = _baseline_poly(xin, bcoeffs)
		else:
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
fit_dict = {}
fit_err = {}

for name, val, err in zip(param_names, popt, perr):
	if name == "log10_AgNumberDensity":
		Nval = 10**val
		Nerr = np.log(10.0) * Nval * err
		fit_dict["AgNumberDensity"] = Nval
		fit_err["AgNumberDensity"] = Nerr
	else:
		fit_dict[name] = val
		fit_err[name] = err

log10N_idx = param_names.index("log10_AgNumberDensity")
log10N_val = popt[log10N_idx]
log10N_err = perr[log10N_idx]

print("\nLog-density fit result:")
print(f"log10_AgNumberDensity = {log10N_val:.6f} ± {log10N_err:.6f}")

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

################################################################################################################################################################
from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path(__file__).resolve().parent

# Fixed name per curr (will overwrite each run for the same curr)
output_file = script_dir / f"baseline_corrected_curr{curr}.csv"

df_out = pd.DataFrame({
    "detuning_uv_GHz": np.asarray(exp_detuning, float),

    # what you plot on ax_main:
    "Transmission_BaselineCorrected": np.asarray(data_norm, float),
    "TransmissionErr_BaselineCorrected": np.asarray(err_norm, float),
    "Theory_NoBaseline": np.asarray(theory_only, float),

    # what you plot on ax_res:
    "Residuals_Norm": np.asarray(residuals_norm, float),
})

df_out.to_csv(output_file, index=False)  # overwrites automatically
print("\nSaved baseline-corrected plot data to:")
print(output_file)
################################################################################################################################################################

def save_fit_parameters_csv(
    curr: int,
    script_dir: Path,
    status: dict,
    fit_dict: dict,
    fit_err: dict,
    fixed_dict: dict,
    default_dict: dict,
    baseline_order: int
):
    """
    Saves one row per parameter with:
        name, status, value, error
    Overwrites: fit_params_curr{curr}.csv
    """

    ordered = (
        ['Temp', 'AgNumberDensity', 'a', 'shift107', 'shift109', 'delta_f'] +
        [f"b{i}" for i in range(0, baseline_order + 1)]
    )

    rows = []
    for name in ordered:
        if name not in status:
            continue

        st = status[name]
        if st == "FIT":
            val = fit_dict.get(name, np.nan)
            err = fit_err.get(name, np.nan)
        elif st == "FIXED":
            val = fixed_dict.get(name, np.nan)
            err = 0.0
        elif st == "DEFAULT":
            val = default_dict.get(name, np.nan)  # may be NaN if you don't know it numerically
            err = 0.0
        else:
            val = np.nan
            err = np.nan

        rows.append({"parameter": name, "status": st, "value": val, "error": err})

    dfp = pd.DataFrame(rows)

    out = script_dir / f"fit_params_curr{curr}.csv"
    dfp.to_csv(out, index=False)  # overwrites
    print("\nSaved fitted parameters to:")
    print(out)

save_fit_parameters_csv(
    curr=curr,
    script_dir=script_dir,
    status=status,
    fit_dict=fit_dict,
    fit_err=fit_err,
    fixed_dict=fixed,
    default_dict=default,
    baseline_order=BASELINE_ORDER
)

################################################################################################################################################################

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

#ax_res.label()

plt.subplots_adjust(hspace=0.05)

# ---------------------------------------------------------
# Coverage check: fraction within ±1 sigma (target ~ 0.68)
# ---------------------------------------------------------
res = residuals_norm
res = res[np.isfinite(res)]

import numpy as np

plt.subplots_adjust(hspace=0.05)

# -----------------------------
# Add side histogram WITHOUT moving existing axes
# -----------------------------
# Freeze current figure/axes geometry AFTER subplots_adjust
orig_fig_w, orig_fig_h = fig.get_size_inches()
pos_main = ax_main.get_position().frozen()
pos_res  = ax_res.get_position().frozen()

# Extend canvas to the right
extra_width_in = 2.2
new_fig_w = orig_fig_w + extra_width_in
fig.set_size_inches(new_fig_w, orig_fig_h, forward=True)

def _keep_physical_bbox(pos, old_w, old_h, new_w, new_h):
	x0_in = pos.x0 * old_w
	y0_in = pos.y0 * old_h
	w_in  = pos.width  * old_w
	h_in  = pos.height * old_h
	return [x0_in / new_w, y0_in / new_h, w_in / new_w, h_in / new_h]

# Re-apply so main/res stay fixed in physical size and location
ax_main.set_position(_keep_physical_bbox(pos_main, orig_fig_w, orig_fig_h, new_fig_w, orig_fig_h))
ax_res .set_position(_keep_physical_bbox(pos_res,  orig_fig_w, orig_fig_h, new_fig_w, orig_fig_h))

# Now place histogram flush to the right of the residuals axis
res_pos = ax_res.get_position().frozen()
pad = 0.01          # set to 0.003 if you want a tiny gap
hist_width = 0.1  # fraction of total NEW figure width

ax_hist = fig.add_axes(
	[res_pos.x1 + pad, res_pos.y0, hist_width, res_pos.height],
	sharey=ax_res
)

# Data
res = residuals_norm[np.isfinite(residuals_norm)]

# Bin width = 1 sigma
rmin = min(np.floor(res.min()), -4)
rmax = max(np.ceil(res.max()),  4)
edges = np.arange(rmin - 0.5, rmax + 0.5 + 1e-9, 1.0)

# Horizontal histogram (rotated)
ax_hist.hist(
	res, bins=edges, density=True,
	orientation='horizontal',
	alpha=0.6, color='blue', edgecolor='black', linewidth=0.8
)

# Ideal N(0,1) PDF overlay
ys = np.linspace(edges[0], edges[-1], 400)
pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ys**2)
ax_hist.plot(pdf, ys, lw=2, color='red')

# Cosmetics
ax_hist.axhline(0, color='grey', lw=1)
ax_hist.set_xlabel("PDF")
ax_hist.set_xlim(left=0)
ax_hist.tick_params(direction='in', top=True, right=True)
plt.setp(ax_hist.get_yticklabels(), visible=False)


N = res.size
p = len(popt)          # number of fitted parameters
nu = N - p             # degrees of freedom

chi2 = np.sum(res**2)
chi2_red = chi2 / nu

within_1 = np.mean(np.abs(res) <= 1.0)
within_2 = np.mean(np.abs(res) <= 2.0)
within_3 = np.mean(np.abs(res) <= 3.0)

N = res.size
print("\n===== RESIDUAL COVERAGE CHECK =====")
print(f"N points = {N}")
print(f"Fraction with |residual| <= 1σ : {within_1:.3f}  (expected ~0.683 for Gaussian)")
print(f"Fraction with |residual| <= 2σ : {within_2:.3f}  (expected ~0.954 for Gaussian)")
print(f"Fraction with |residual| <= 3σ : {within_3:.3f}  (expected ~0.997 for Gaussian)")

print(f"\nChi^2 = {chi2:.3f}")
print(f"DoF (nu) = {nu}")
print(f"Reduced Chi^2 = {chi2_red:.3f}")

import scipy.stats as stats

ks = ks = stats.kstest(res, 'norm')
print("\nKS test vs N(0,1):")
print(f"D = {ks.statistic:.3f},  p-value = {ks.pvalue:.3f}")

#plt.savefig("Spec15MicW"+str(curr)+"A_"+str(Ag107ShiftDefault - Ag109ShiftDefault)+"MHz_POP_NP.png", dpi=300, bbox_inches='tight')

plt.show()

N = fit_dict["AgNumberDensity"]
Nerr = fit_err["AgNumberDensity"]
print("AgNumberDensity =", N)
print("AgNumberDensity err =", Nerr)
print("Relative error =", Nerr / N)