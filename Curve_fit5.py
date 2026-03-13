import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import rcParams
from pathlib import Path
from libs import main_functions as mf
import os
from tqdm.auto import tqdm

# =========================================================
# WORKING DIRECTORY
# =========================================================
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
# GLOBAL SWITCHES
# =========================================================
FIT_POPULATION = True
FIT_ISOTOPE    = False
FIT_DELTA_F    = True
FIT_BASELINE_WEAKPROBE  = True
FIT_BASELINE_SUBDOPPLER = True

Ag107ShiftDefault = 229.24
Ag109ShiftDefault = -246.76
delta_f_fixed = 0.0

MAX_BASELINE_ORDER = 15

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
# FIXED SUB-DOPPLER INPUTS
# =========================================================
SUBDOPPLER_FIXED = {
	'I_pump': 2030,
	'I_probe': 13.2,
	'I_sat': 867,
	'gamma_transit_Hz': 2.0e4,
	'n_vcc_steps': 1,
	'beta_vcc': 1.0,
	'pump_pol': 'Left',
	'probe_pol': 'Left',
	'Nv': 101,
	'vmax_sigma': 4.0,
	'include_excited_vcc': False,
}

SUBDOPPLER_INITIAL = {
	'eta_pump': 0.01,
	'gamma_vcc_Hz': 1.0e5,
	'vcc_width': 20.0,
}

SUBDOPPLER_BASELINE_INITIAL = {
	'b0': 1.1746,
	'b1': -0.00018567,
	'b2': -0.000697833,
	'b3': -3.29483e-05,
	'b4': 5.19833e-06,
}

# =========================================================
# SUB-DOPPLER GENERAL FIT CONTROL
# =========================================================

catalog = [
	'Temp',#0
	'AgNumberDensity',#1
	'a',#2
	'a_none',#3
	'shift107',#4
	'shift109',#5
	'delta_f',#6
	'eta_pump',#7
	'gamma_vcc_Hz',#8
	'vcc_width',#9
	'baseline',#10
]

FIT_SUBDOPPLER_INDICES = []

SUBDOPPLER_OVERRIDES = {
    'a': 0,
}

# =========================================================
# DATASETS
# =========================================================

DATASETS = {
	# -------------------------
	# Weak probe data
	# -------------------------
	"weakprobe_4A": {
		"kind": "weakprobe",
		"freq_file": "frequencies4A.csv",
		"trans_file": "Spec15MicW4A.csv",
		"baseline_order": 2,
	},
	"weakprobe_6A": {
		"kind": "weakprobe",
		"freq_file": "frequencies6A.csv",
		"trans_file": "Spec15MicW6A.csv",
		"baseline_order": 2,
	},
	"weakprobe_7A": {
		"kind": "weakprobe",
		"freq_file": "frequencies7A.csv",
		"trans_file": "Spec15MicW7A.csv",
		"baseline_order": 2,
	},
	"weakprobe_8A": {
		"kind": "weakprobe",
		"freq_file": "frequencies8A.csv",
		"trans_file": "Spec15MicW8A.csv",
		"baseline_order": 4,
	},
	"weakprobe_8A_SD3_NP": {
		"kind": "weakprobe",
		"freq_file": "frequencies8A_SD3_NP.csv",
		"trans_file": "SubDoppler3_NP_8A.csv",
		"baseline_order": 4,
	},

	# -------------------------
	# Sub-Doppler data
	# -------------------------
	"subdoppler_8A_SD3_WP": {
		"kind": "subdoppler",
		"freq_file": "frequencies8A_SD3_WP.csv",
		"trans_file": "SubDoppler3_WP_8A.csv",
		"baseline_order": 4,
	},
}

# =========================================================
# USER CHOICES
# =========================================================

WEAKPROBE_DATASET_NAME  = "weakprobe_8A_SD3_NP"
SUBDOPPLER_DATASET_NAME = "subdoppler_8A_SD3_WP"

# =========================================================
# UTILITIES
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

def load_dataset(dataset_name):
	ds = DATASETS[dataset_name]

	frequencies = pd.read_csv(ds["freq_file"])
	transmissions = pd.read_csv(ds["trans_file"])

	freq_raw = np.array(frequencies["freq"])
	trans = np.array(transmissions["Transmission"])
	transerr = np.array(transmissions["Transmissionerr"])

	mask = (
		np.isfinite(freq_raw) &
		np.isfinite(trans) &
		np.isfinite(transerr)
	)

	freq_raw = freq_raw[mask]
	trans = trans[mask]
	transerr = transerr[mask]

	freq_raw, trans, transerr = sort_by_frequency_descending(
		freq_raw, trans, transerr
	)

	c = 2.99792458e8
	lambd = 328.1629601
	freq_base = -freq_raw * 2 + (c / lambd)

	exp_detuning = freq_base
	exp_transmission = trans
	exp_error = np.abs(transerr)
	freqerr = 0.01
	freqerr_array = np.full_like(freq_base, freqerr)

	x0 = np.mean(exp_detuning)

	return {
		"name": dataset_name,
		"kind": ds["kind"],
		"freq_file": ds["freq_file"],
		"trans_file": ds["trans_file"],
		"baseline_order": ds["baseline_order"],
		"exp_detuning": exp_detuning,
		"exp_transmission": exp_transmission,
		"exp_error": exp_error,
		"freqerr_array": freqerr_array,
		"x0": x0,
	}

def _baseline_poly(x, coeffs):
	coeffs = np.asarray(coeffs, dtype=float)
	y = 0.0
	for c in coeffs[::-1]:
		y = y * x + c
	return y

def apply_manual_overrides(target_dict, overrides, allowed_keys=None, label="overrides"):
	"""
	Apply manual overrides to a parameter dictionary.

	Parameters
	----------
	target_dict : dict
		Dictionary to be modified in place.
	overrides : dict
		Manual override values.
	allowed_keys : set or None
		If provided, restrict overrides to this set of keys.
	label : str
		Used in error messages.
	"""
	if overrides is None:
		return target_dict

	for k, v in overrides.items():
		if allowed_keys is not None and k not in allowed_keys:
			raise ValueError(
				f"Unknown key '{k}' in {label}. "
				f"Allowed keys are: {sorted(allowed_keys)}"
			)
		target_dict[k] = v

	return target_dict

def get_subdoppler_override_keys(baseline_order):
	keys = {
		'Temp',
		'AgNumberDensity',
		'a',
		'shift107',
		'shift109',
		'delta_f',
		'eta_pump',
		'gamma_vcc_Hz',
		'vcc_width',
		'I_pump',
		'I_probe',
		'I_sat',
		'gamma_transit_Hz',
		'n_vcc_steps',
		'beta_vcc',
		'pump_pol',
		'probe_pol',
		'Nv',
		'vmax_sigma',
		'include_excited_vcc',
	}

	for i in range(baseline_order + 1):
		keys.add(f"b{i}")

	return keys

def _theory_curve_GHz_axis_weakprobe(
	Temp, AgNumberDensity, a=None, shift107=None, shift109=None
):
	p_dict = {
		'Elem': element,
		'Dline': Dline,
		'T': Temp,
		'lcell': lcell,
		'Bfield': Bfield,
		'Btheta': Btheta,
		'AgNumden': AgNumberDensity,
		'Isotope_Combination': 0,
		'SubDoppler': False,
	}

	if a is not None:
		b = (1.0 - a) / 3.0
		p_dict['CustomPop'] = [a, b, b, b]
	else:
		p_dict['CustomPop'] = None

	if (shift107 is not None) and (shift109 is not None):
		p_dict['AgIsotope_shift'] = (shift107, shift109)

	[S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
	theory_curve = S0[0].real
	theory_detuning_GHz = Detuning / 1e3

	return theory_detuning_GHz, theory_curve

def _theory_curve_GHz_axis_subdoppler(weakprobe_fit, subdoppler_fit):
	p_dict = {
		'Elem': element,
		'Dline': Dline,
		'T': weakprobe_fit['Temp'],
		'lcell': lcell,
		'Bfield': Bfield,
		'Btheta': Btheta,
		'AgNumden': weakprobe_fit['AgNumberDensity'],
		'Isotope_Combination': 0,
		'SubDoppler': True,
	}

	if weakprobe_fit.get('a', None) is not None:
		a = weakprobe_fit['a']
		b = (1.0 - a) / 3.0
		p_dict['CustomPop'] = [a, b, b, b]
	else:
		p_dict['CustomPop'] = None

	shift107 = weakprobe_fit.get('shift107', Ag107ShiftDefault)
	shift109 = weakprobe_fit.get('shift109', Ag109ShiftDefault)
	p_dict['AgIsotope_shift'] = (shift107, shift109)

	pump_params = {
		'pol': SUBDOPPLER_FIXED['pump_pol'],
		'probe_pol': SUBDOPPLER_FIXED['probe_pol'],
		'eta_pump': subdoppler_fit['eta_pump'],
		'I_pump': SUBDOPPLER_FIXED['I_pump'],
		'I_probe': SUBDOPPLER_FIXED['I_probe'],
		'I_sat': SUBDOPPLER_FIXED['I_sat'],
	}

	subdop_params = {
		'Nv': SUBDOPPLER_FIXED['Nv'],
		'vmax_sigma': SUBDOPPLER_FIXED['vmax_sigma'],
		'gamma_transit_Hz': SUBDOPPLER_FIXED['gamma_transit_Hz'],
		'gamma_vcc_Hz': subdoppler_fit['gamma_vcc_Hz'],
		'vcc_width': subdoppler_fit['vcc_width'],
		'include_excited_vcc': SUBDOPPLER_FIXED['include_excited_vcc'],
		'n_vcc_steps': SUBDOPPLER_FIXED['n_vcc_steps'],
		'beta_vcc': SUBDOPPLER_FIXED['beta_vcc'],
	}

	p_dict['pump_params'] = pump_params
	p_dict['subdop_params'] = subdop_params

	[S0] = mf.get_spectra(Detuning, E_in, p_dict, outputs=['S0'])
	theory_curve = S0[0].real
	theory_detuning_GHz = Detuning / 1e3

	return theory_detuning_GHz, theory_curve

class FitProgress:
	def __init__(self, desc="Fitting", total=None, min_update=1):
		self.pbar = tqdm(total=total, desc=desc)
		self.calls = 0
		self.min_update = min_update

	def update(self):
		self.calls += 1
		if self.calls % self.min_update == 0:
			self.pbar.update(self.min_update)

	def close(self):
		self.pbar.close()

def expand_baseline_defaults(baseline_defaults, baseline_order):
	"""
	Ensure baseline default dictionary contains b0...bN.
	Missing terms are filled with 0.0, except b0 which defaults to 1.0.
	"""
	out = dict(baseline_defaults)
	for i in range(baseline_order + 1):
		key = f"b{i}"
		if key not in out:
			out[key] = 1.0 if i == 0 else 0.0
	return out

def get_subdoppler_fit_catalog():
	"""
	Master list of parameters that may be varied in the sub-Doppler fit.

	'a'      -> fit a numeric population parameter
	'a_none' -> force a = None so the internal scaling is used
	'baseline' -> fit the full sub-Doppler baseline b0...bN
	"""
	catalog = [
		'Temp',#0
		'AgNumberDensity',#1
		'a',#2
		'a_none',#3
		'shift107',#4
		'shift109',#5
		'delta_f',#6
		'eta_pump',#7
		'gamma_vcc_Hz',#8
		'vcc_width',#9
		'baseline',#10
	]
	return catalog

def resolve_subdoppler_fit_params(indices):
	catalog = get_subdoppler_fit_catalog()

	if len(indices) != len(set(indices)):
		raise ValueError("FIT_SUBDOPPLER_INDICES contains duplicates")

	for idx in indices:
		if not (0 <= idx < len(catalog)):
			raise ValueError(
				f"FIT_SUBDOPPLER_INDICES contains {idx}, "
				f"but valid indices are 0 to {len(catalog)-1}"
			)

	selected = [catalog[i] for i in indices]

	if ('a' in selected) and ('a_none' in selected):
		raise ValueError("Choose either 'a' or 'a_none', not both")

	return catalog, selected

def print_subdoppler_fit_catalog():
	catalog = get_subdoppler_fit_catalog()
	print("\nSub-Doppler fit parameter catalogue:")
	for i, p in enumerate(catalog):
		print(f"  {i}: {p}")

def build_subdoppler_result_without_fit(
    dataset,
    weakprobe_fit,
    fixed_subdoppler_params,
    manual_overrides=None
):
    """
    Build the final sub-Doppler parameter dictionary when no parameters
    are being fitted.
    """
    BASELINE_ORDER = dataset["baseline_order"]

    fit_dict = dict(weakprobe_fit)

    # Start from fixed/default sub-Doppler values
    for k, v in fixed_subdoppler_params.items():
        fit_dict[k] = v

    fit_err = {k: 0.0 for k in fit_dict}

    # Ensure baseline exists
    for i in range(BASELINE_ORDER + 1):
        k = f"b{i}"
        if k not in fit_dict:
            fit_dict[k] = fixed_subdoppler_params.get(k, 1.0 if i == 0 else 0.0)
            fit_err[k] = 0.0

    # Apply manual overrides last
    if manual_overrides:
        allowed_override_keys = get_subdoppler_override_keys(BASELINE_ORDER)
        for k, v in manual_overrides.items():
            if k not in allowed_override_keys:
                raise ValueError(
                    f"Unknown key '{k}' in SUBDOPPLER_OVERRIDES. "
                    f"Allowed keys are: {sorted(allowed_override_keys)}"
                )
            fit_dict[k] = v
            fit_err[k] = 0.0

    # Ensure fixed sub-Doppler constants are present
    fit_dict['I_pump'] = SUBDOPPLER_FIXED['I_pump']
    fit_dict['I_probe'] = SUBDOPPLER_FIXED['I_probe']
    fit_dict['I_sat'] = SUBDOPPLER_FIXED['I_sat']
    fit_dict['gamma_transit_Hz'] = SUBDOPPLER_FIXED['gamma_transit_Hz']
    fit_dict['n_vcc_steps'] = SUBDOPPLER_FIXED['n_vcc_steps']
    fit_dict['beta_vcc'] = SUBDOPPLER_FIXED['beta_vcc']
    fit_dict['pump_pol'] = SUBDOPPLER_FIXED['pump_pol']
    fit_dict['probe_pol'] = SUBDOPPLER_FIXED['probe_pol']
    fit_dict['Nv'] = SUBDOPPLER_FIXED['Nv']
    fit_dict['vmax_sigma'] = SUBDOPPLER_FIXED['vmax_sigma']
    fit_dict['include_excited_vcc'] = SUBDOPPLER_FIXED['include_excited_vcc']

    fit_err['I_pump'] = 0.0
    fit_err['I_probe'] = 0.0
    fit_err['I_sat'] = 0.0
    fit_err['gamma_transit_Hz'] = 0.0
    fit_err['n_vcc_steps'] = 0.0
    fit_err['beta_vcc'] = 0.0
    fit_err['pump_pol'] = 0.0
    fit_err['probe_pol'] = 0.0
    fit_err['Nv'] = 0.0
    fit_err['vmax_sigma'] = 0.0
    fit_err['include_excited_vcc'] = 0.0

    return fit_dict, fit_err

# =========================================================
# WEAK-PROBE FIT
# =========================================================

def build_weakprobe_model(dataset):
	BASELINE_ORDER = dataset["baseline_order"]
	x0 = dataset["x0"]

	if not (0 <= BASELINE_ORDER <= MAX_BASELINE_ORDER):
		raise ValueError(f"BASELINE_ORDER must be between 0 and {MAX_BASELINE_ORDER}")

	param_names = ['Temp', 'AgNumberDensity']

	if FIT_POPULATION:
		param_names += ['a']
	if FIT_ISOTOPE:
		param_names += ['shift107', 'shift109']
	if FIT_DELTA_F:
		param_names += ['delta_f']
	if FIT_BASELINE_WEAKPROBE:
		param_names += [f"b{i}" for i in range(BASELINE_ORDER + 1)]

	p0 = [90.0, 1.5e16]
	if FIT_POPULATION:
		p0 += [0.4]
	if FIT_ISOTOPE:
		p0 += [Ag107ShiftDefault, Ag109ShiftDefault]
	if FIT_DELTA_F:
		p0 += [delta_f_fixed]
	if FIT_BASELINE_WEAKPROBE:
		p0 += [1.0] + [0.0] * BASELINE_ORDER

	lo, hi = [], []
	lo += [0.0, 1e15]
	hi += [2000.0, 5e16]

	if FIT_POPULATION:
		lo += [0.0]
		hi += [1.0]
	if FIT_ISOTOPE:
		lo += [-10000.0, -10000.0]
		hi += [10000.0, 10000.0]
	if FIT_DELTA_F:
		lo += [delta_f_fixed - 2.0]
		hi += [delta_f_fixed + 2.0]
	if FIT_BASELINE_WEAKPROBE:
		lo += [0.2] + [-1.0] * BASELINE_ORDER
		hi += [2.0] + [1.0] * BASELINE_ORDER

	bounds = (lo, hi)

	def model(exp_detuning_in, *params):
		idx = 0
		Temp = params[idx]; idx += 1
		Nden = params[idx]; idx += 1

		a = None
		if FIT_POPULATION:
			a = params[idx]; idx += 1

		if FIT_ISOTOPE:
			shift107 = params[idx]
			shift109 = params[idx + 1]
			idx += 2
		else:
			shift107 = Ag107ShiftDefault
			shift109 = Ag109ShiftDefault

		if FIT_DELTA_F:
			delta_f = params[idx]
			idx += 1
		else:
			delta_f = delta_f_fixed

		if FIT_BASELINE_WEAKPROBE:
			bcoeffs = list(params[idx:idx + (BASELINE_ORDER + 1)])
		else:
			bcoeffs = [1.0] + [0.0] * BASELINE_ORDER

		tG, tT = _theory_curve_GHz_axis_weakprobe(
			Temp, Nden, a=a, shift107=shift107, shift109=shift109
		)
		theory_interp = np.interp(exp_detuning_in + delta_f, tG, tT)

		xin = exp_detuning_in - x0
		B = _baseline_poly(xin, bcoeffs)

		return B * theory_interp

	return model, p0, bounds, param_names

def run_weakprobe_fit(dataset):
	fit_model, p0, bounds, param_names = build_weakprobe_model(dataset)

	popt, pcov = curve_fit(
		fit_model,
		dataset["exp_detuning"],
		dataset["exp_transmission"],
		sigma=dataset["exp_error"],
		absolute_sigma=True,
		p0=p0,
		bounds=bounds,
		maxfev=30000
	)

	perr = np.sqrt(np.diag(pcov))

	fit_dict = {k: v for k, v in zip(param_names, popt)}
	fit_err = {k: e for k, e in zip(param_names, perr)}

	if not FIT_POPULATION:
		fit_dict['a'] = None
		fit_err['a'] = 0.0

	if not FIT_ISOTOPE:
		fit_dict['shift107'] = Ag107ShiftDefault
		fit_dict['shift109'] = Ag109ShiftDefault
		fit_err['shift107'] = 0.0
		fit_err['shift109'] = 0.0

	if not FIT_DELTA_F:
		fit_dict['delta_f'] = delta_f_fixed
		fit_err['delta_f'] = 0.0

	if not FIT_BASELINE_WEAKPROBE:
		for i in range(dataset["baseline_order"] + 1):
			fit_dict[f"b{i}"] = 1.0 if i == 0 else 0.0
			fit_err[f"b{i}"] = 0.0

	return popt, pcov, fit_dict, fit_err, fit_model, param_names

# =========================================================
# SUB-DOPPLER FIT
# =========================================================
def build_subdoppler_model(
	dataset,
	weakprobe_fit,
	free_params,
	fixed_subdoppler_params,
	manual_overrides=None,
	progress=None
):
	BASELINE_ORDER = dataset["baseline_order"]
	x0 = dataset["x0"]

	allowed = set(get_subdoppler_fit_catalog())
	for p in free_params:
		if p not in allowed:
			raise ValueError(f"Unknown free parameter: {p}")

	fit_full_baseline = ('baseline' in free_params)

	# actual curve_fit parameter list
	param_names = []
	for p in free_params:
		if p == 'baseline':
			param_names += [f"b{i}" for i in range(BASELINE_ORDER + 1)]
		elif p == 'a_none':
			# selector only, no numeric fit parameter
			continue
		else:
			param_names.append(p)

	p0 = []
	lo = []
	hi = []

	for p in free_params:
		if p == 'Temp':
			p0.append(weakprobe_fit['Temp'])
			lo.append(0.0)
			hi.append(2000.0)

		elif p == 'AgNumberDensity':
			p0.append(weakprobe_fit['AgNumberDensity'])
			lo.append(1e15)
			hi.append(5e16)

		elif p == 'a':
			a0 = weakprobe_fit.get('a', None)
			if a0 is None:
				a0 = 0.4
			p0.append(a0)
			lo.append(0.0)
			hi.append(1.0)

		elif p == 'a_none':
			# Selector only: no numeric fit parameter
			continue

		elif p == 'shift107':
			p0.append(weakprobe_fit.get('shift107', Ag107ShiftDefault))
			lo.append(-10000.0)
			hi.append(10000.0)

		elif p == 'shift109':
			p0.append(weakprobe_fit.get('shift109', Ag109ShiftDefault))
			lo.append(-10000.0)
			hi.append(10000.0)

		elif p == 'delta_f':
			p0.append(weakprobe_fit.get('delta_f', delta_f_fixed))
			lo.append(delta_f_fixed - 2.0)
			hi.append(delta_f_fixed + 2.0)

		elif p == 'eta_pump':
			p0.append(fixed_subdoppler_params.get('eta_pump', 1.0))
			lo.append(0.0)
			hi.append(100.0)

		elif p == 'gamma_vcc_Hz':
			p0.append(fixed_subdoppler_params.get('gamma_vcc_Hz', 0.0))
			lo.append(0.0)
			hi.append(1e8)

		elif p == 'vcc_width':
			p0.append(fixed_subdoppler_params.get('vcc_width', 20.0))
			lo.append(0.0)
			hi.append(1e4)

		elif p == 'baseline':
			for i in range(BASELINE_ORDER + 1):
				key = f"b{i}"
				default = 1.0 if i == 0 else 0.0
				p0.append(fixed_subdoppler_params.get(key, default))
			lo += [0.2] + [-1.0] * BASELINE_ORDER
			hi += [2.0] + [1.0] * BASELINE_ORDER

		else:
			raise ValueError(f"No bounds defined for parameter '{p}'")

	bounds = (lo, hi)

	def model(exp_detuning_in, *params):
		if progress is not None:
			progress.update()

		idx = 0

		# Start from weak-probe fitted values
		wp_local = dict(weakprobe_fit)

		# Start from fixed sub-Doppler values
		subdoppler_fit = dict(fixed_subdoppler_params)

		force_a_none = ('a_none' in free_params)

		for p in free_params:
			if p == 'baseline':
				for i in range(BASELINE_ORDER + 1):
					subdoppler_fit[f"b{i}"] = params[idx]
					idx += 1

			elif p in ['Temp', 'AgNumberDensity', 'a', 'shift107', 'shift109', 'delta_f']:
				wp_local[p] = params[idx]
				idx += 1

			elif p == 'a_none':
				continue

			else:
				subdoppler_fit[p] = params[idx]
				idx += 1

		#########
		if force_a_none:
			wp_local['a'] = None

		# Ensure baseline exists even if not being varied
		for i in range(BASELINE_ORDER + 1):
			key = f"b{i}"
			if key not in subdoppler_fit:
				subdoppler_fit[key] = fixed_subdoppler_params.get(key, 1.0 if i == 0 else 0.0)

		# Apply manual overrides last so they take precedence over both
		# weak-probe values and fitted/fixed sub-Doppler values.
		if manual_overrides:
			allowed_override_keys = get_subdoppler_override_keys(BASELINE_ORDER)

			for k, v in manual_overrides.items():
				if k not in allowed_override_keys:
					raise ValueError(
						f"Unknown key '{k}' in SUBDOPPLER_OVERRIDES. "
						f"Allowed keys are: {sorted(allowed_override_keys)}"
					)

				if k in {
					'Temp', 'AgNumberDensity', 'a', 'shift107', 'shift109', 'delta_f'
				}:
					wp_local[k] = v
				else:
					subdoppler_fit[k] = v
		##########
		tG, tT = _theory_curve_GHz_axis_subdoppler(wp_local, subdoppler_fit)
		theory_interp = np.interp(exp_detuning_in + wp_local['delta_f'], tG, tT)

		xin = exp_detuning_in - x0
		bcoeffs = [subdoppler_fit[f"b{i}"] for i in range(BASELINE_ORDER + 1)]
		B = _baseline_poly(xin, bcoeffs)

		return B * theory_interp

	return model, p0, bounds, param_names

def run_subdoppler_fit(
    dataset,
    weakprobe_fit,
    free_params,
    fixed_subdoppler_params,
    manual_overrides=None,
    desc="Sub-Doppler fit"
):
    # Case 1: no free parameters -> no fitting, just assemble result
    if len(free_params) == 0:
        fit_model, p0, bounds, param_names = build_subdoppler_model(
            dataset,
            weakprobe_fit,
            free_params,
            fixed_subdoppler_params,
            manual_overrides=manual_overrides,
            progress=None
        )

        fit_dict, fit_err = build_subdoppler_result_without_fit(
            dataset,
            weakprobe_fit,
            fixed_subdoppler_params,
            manual_overrides=manual_overrides
        )

        popt = np.array([])
        pcov = np.zeros((0, 0))

        return popt, pcov, fit_dict, fit_err, fit_model, param_names

    # Case 2: normal fit
    progress = FitProgress(desc=desc)

    fit_model, p0, bounds, param_names = build_subdoppler_model(
        dataset,
        weakprobe_fit,
        free_params,
        fixed_subdoppler_params,
        manual_overrides=manual_overrides,
        progress=progress
    )

    try:
        popt, pcov = curve_fit(
            fit_model,
            dataset["exp_detuning"],
            dataset["exp_transmission"],
            sigma=dataset["exp_error"],
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=40000
        )
    finally:
        progress.close()

    perr = np.sqrt(np.diag(pcov))

    fit_dict = dict(weakprobe_fit)

    for k, v in fixed_subdoppler_params.items():
        fit_dict[k] = v

    fit_err = {k: 0.0 for k in fit_dict}

    for k, v, e in zip(param_names, popt, perr):
        fit_dict[k] = v
        fit_err[k] = e

    if manual_overrides:
        allowed_override_keys = get_subdoppler_override_keys(dataset["baseline_order"])
        for k, v in manual_overrides.items():
            if k not in allowed_override_keys:
                raise ValueError(
                    f"Unknown key '{k}' in SUBDOPPLER_OVERRIDES. "
                    f"Allowed keys are: {sorted(allowed_override_keys)}"
                )
            fit_dict[k] = v
            fit_err[k] = 0.0

    fit_dict['I_pump'] = SUBDOPPLER_FIXED['I_pump']
    fit_dict['I_probe'] = SUBDOPPLER_FIXED['I_probe']
    fit_dict['I_sat'] = SUBDOPPLER_FIXED['I_sat']
    fit_dict['gamma_transit_Hz'] = SUBDOPPLER_FIXED['gamma_transit_Hz']
    fit_dict['n_vcc_steps'] = SUBDOPPLER_FIXED['n_vcc_steps']
    fit_dict['beta_vcc'] = SUBDOPPLER_FIXED['beta_vcc']
    fit_dict['pump_pol'] = SUBDOPPLER_FIXED['pump_pol']
    fit_dict['probe_pol'] = SUBDOPPLER_FIXED['probe_pol']
    fit_dict['Nv'] = SUBDOPPLER_FIXED['Nv']
    fit_dict['vmax_sigma'] = SUBDOPPLER_FIXED['vmax_sigma']
    fit_dict['include_excited_vcc'] = SUBDOPPLER_FIXED['include_excited_vcc']

    fit_err['I_pump'] = 0.0
    fit_err['I_probe'] = 0.0
    fit_err['I_sat'] = 0.0
    fit_err['gamma_transit_Hz'] = 0.0
    fit_err['n_vcc_steps'] = 0.0
    fit_err['beta_vcc'] = 0.0
    fit_err['pump_pol'] = 0.0
    fit_err['probe_pol'] = 0.0
    fit_err['Nv'] = 0.0
    fit_err['vmax_sigma'] = 0.0
    fit_err['include_excited_vcc'] = 0.0

    for i in range(dataset["baseline_order"] + 1):
        k = f"b{i}"
        if k not in fit_dict:
            fit_dict[k] = fixed_subdoppler_params.get(k, 1.0 if i == 0 else 0.0)
            fit_err[k] = 0.0

    return popt, pcov, fit_dict, fit_err, fit_model, param_names

# =========================================================
# NORMALISATION HELPERS
# =========================================================

def compute_weakprobe_baseline_and_theory(dataset, fit_dict):
	BASELINE_ORDER = dataset["baseline_order"]
	x0 = dataset["x0"]

	xin = dataset["exp_detuning"] - x0
	bcoeffs = [fit_dict[f"b{i}"] for i in range(BASELINE_ORDER + 1)]
	baseline = _baseline_poly(xin, bcoeffs)

	a = fit_dict.get('a', None)
	shift107 = fit_dict.get('shift107', Ag107ShiftDefault)
	shift109 = fit_dict.get('shift109', Ag109ShiftDefault)
	delta_f = fit_dict.get('delta_f', delta_f_fixed)

	tG, tT = _theory_curve_GHz_axis_weakprobe(
		fit_dict['Temp'],
		fit_dict['AgNumberDensity'],
		a=a,
		shift107=shift107,
		shift109=shift109
	)
	theory_only = np.interp(dataset["exp_detuning"] + delta_f, tG, tT)

	return baseline, theory_only

def compute_subdoppler_baseline_and_theory(dataset, weakprobe_fit, subdoppler_fit):
	BASELINE_ORDER = dataset["baseline_order"]
	x0 = dataset["x0"]

	xin = dataset["exp_detuning"] - x0
	bcoeffs = [subdoppler_fit[f"b{i}"] for i in range(BASELINE_ORDER + 1)]
	baseline = _baseline_poly(xin, bcoeffs)

	delta_f = weakprobe_fit.get('delta_f', delta_f_fixed)
	tG, tT = _theory_curve_GHz_axis_subdoppler(weakprobe_fit, subdoppler_fit)
	theory_only = np.interp(dataset["exp_detuning"] + delta_f, tG, tT)

	return baseline, theory_only

# =========================================================
# PRINT / PLOT HELPERS
# =========================================================

def print_fit_dict(title, fit_dict, fit_err):
	print(f"\n===== {title} =====")
	for k in fit_dict:
		v = fit_dict[k]
		e = fit_err.get(k, None)

		if isinstance(v, (int, float, np.floating)):
			if e is None:
				print(f"{k:>20s} = {v:.6g}")
			else:
				print(f"{k:>20s} = {v:.6g} ± {e:.6g}")
		else:
			print(f"{k:>20s} = {v}")

def plot_normalised_fit_result(dataset, data_norm, err_norm, theory_only, title):
	residuals = (data_norm - theory_only) / err_norm

	fig, (ax_main, ax_res) = plt.subplots(
		2, 1, figsize=(8, 6), sharex=True,
		gridspec_kw={"height_ratios": [3, 1]}
	)

	ax_main.errorbar(
		dataset["exp_detuning"],
		data_norm,
		yerr=err_norm,
		xerr=dataset["freqerr_array"],
		fmt='x', color='black',
		label='Data / baseline'
	)
	ax_main.plot(
		dataset["exp_detuning"],
		theory_only,
		color='red', lw=2,
		label='Theory'
	)
	ax_main.axhline(1, color='grey', lw=1)
	ax_main.set_ylabel("Transmission")
	ax_main.legend()
	ax_main.set_title(title)

	ax_res.axhline(0, color='grey', lw=1)
	ax_res.errorbar(
		dataset["exp_detuning"],
		residuals,
		yerr=np.ones_like(residuals),
		xerr=dataset["freqerr_array"],
		fmt='x', color='black', markersize=4
	)
	ax_res.set_ylabel("Norm. residual")
	ax_res.set_xlabel("Linear Detuning (GHz)")

	plt.subplots_adjust(hspace=0.05)
	plt.show()

def print_subdoppler_pasteback_block(dataset, sd_fit_dict):
	print("\nPaste back into the top of the script:\n")
	print("SUBDOPPLER_INITIAL = {")
	print(f"    'eta_pump': {sd_fit_dict['eta_pump']:.10g},")
	print(f"    'gamma_vcc_Hz': {sd_fit_dict['gamma_vcc_Hz']:.10g},")
	print(f"    'vcc_width': {sd_fit_dict['vcc_width']:.10g},")
	print("}")
	print("")
	print("SUBDOPPLER_BASELINE_INITIAL = {")
	for i in range(dataset["baseline_order"] + 1):
		k = f"b{i}"
		print(f"    '{k}': {sd_fit_dict[k]:.10g},")
	print("}")

# =========================================================
# RUN PIPELINE
# =========================================================
weakprobe_dataset = load_dataset(WEAKPROBE_DATASET_NAME)
subdoppler_dataset = load_dataset(SUBDOPPLER_DATASET_NAME)

SUBDOPPLER_BASELINE_INITIAL = expand_baseline_defaults(
	SUBDOPPLER_BASELINE_INITIAL,
	subdoppler_dataset["baseline_order"]
)

# ---------- Stage 1: weak-probe fit ----------
wp_popt, wp_pcov, wp_fit_dict, wp_fit_err, wp_fit_model, wp_param_names = run_weakprobe_fit(weakprobe_dataset)
print_fit_dict("WEAK-PROBE FIT", wp_fit_dict, wp_fit_err)

wp_baseline, wp_theory_only = compute_weakprobe_baseline_and_theory(weakprobe_dataset, wp_fit_dict)
wp_data_norm = weakprobe_dataset["exp_transmission"] / wp_baseline
wp_err_norm = weakprobe_dataset["exp_error"] / np.abs(wp_baseline)

plot_normalised_fit_result(
	weakprobe_dataset,
	wp_data_norm,
	wp_err_norm,
	wp_theory_only,
	f"Weak-probe fit: {WEAKPROBE_DATASET_NAME}"
)
# ---------- Stage 2: single-parameter sub-Doppler fit ----------
sd_fixed = dict(SUBDOPPLER_INITIAL)

# initialise sub-Doppler baseline starting values
for i in range(subdoppler_dataset["baseline_order"] + 1):
	k = f"b{i}"
	sd_fixed[k] = SUBDOPPLER_BASELINE_INITIAL.get(k, 1.0 if i == 0 else 0.0)

catalog, free_params = resolve_subdoppler_fit_params(FIT_SUBDOPPLER_INDICES)

print_subdoppler_fit_catalog()
print(f"\nSelected sub-Doppler fit parameters: {free_params}")

sd_popt, sd_pcov, sd_fit_dict, sd_fit_err, sd_fit_model, sd_param_names = run_subdoppler_fit(
	subdoppler_dataset,
	wp_fit_dict,
	free_params=free_params,
	fixed_subdoppler_params=sd_fixed,
	manual_overrides=SUBDOPPLER_OVERRIDES,
	desc=f"Sub-Doppler fit: {', '.join(free_params)}"
)

print_fit_dict(f"SUB-DOPPLER FIT ({', '.join(free_params)})", sd_fit_dict, sd_fit_err)
print_subdoppler_pasteback_block(subdoppler_dataset, sd_fit_dict)

sd_baseline, sd_theory_only = compute_subdoppler_baseline_and_theory(
	subdoppler_dataset,
	wp_fit_dict,
	sd_fit_dict
)
sd_data_norm = subdoppler_dataset["exp_transmission"] / sd_baseline
sd_err_norm = subdoppler_dataset["exp_error"] / np.abs(sd_baseline)

plot_normalised_fit_result(
	subdoppler_dataset,
	sd_data_norm,
	sd_err_norm,
	sd_theory_only,
	f"Sub-Doppler fit ({', '.join(free_params)}): {SUBDOPPLER_DATASET_NAME}"
)