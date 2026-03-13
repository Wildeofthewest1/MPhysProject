import numpy as np
from scipy.constants import physical_constants, epsilon_0, hbar, c, h

from libs import atomic_constants as ac
from libs import Hamiltonian as ht

# ---------------------------------------------------------
# Import legacy/stable helpers from the original module
# ---------------------------------------------------------
from libs.main_functions import (
	p_dict_defaults,
	calc_chi,
	get_Efield,
	chi_to_S0,
	_lorentz_complex,
	_sat_lineshape,
	_maxwell_1d,
	_build_velocity_grid,
	_build_gaussian_vcc_kernel,
	_build_cusp_vcc_kernel,
	_build_thermal_reset_vcc_kernel,
	_build_vcc_kernel,
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
S = 0.5
gs = -physical_constants['electron g factor'][0]
muB = physical_constants['Bohr magneton'][0]
kB = physical_constants['Boltzmann constant'][0]
amu = physical_constants['atomic mass constant'][0]
e0 = epsilon_0
a0 = physical_constants['Bohr radius'][0]

# =========================================================
# RAW / UNGROUPED TRANSITIONS
# =========================================================
def FreqStren_raw(
	groundLevels,
	excitedLevels,
	groundDim,
	excitedDim,
	Dline,
	hand,
	BoltzmannFactor=True,
	T=293.16,
	custom_pop=None,
	strength_tol=0.0005
):
	"""
	Return ungrouped state-to-state transitions.
	"""

	transitions = []

	if custom_pop is not None:
		BoltzDist = np.array(custom_pop, dtype=float)
		BoltzDist /= BoltzDist.sum()
	else:
		if BoltzmannFactor:
			groundEnergies = np.array(groundLevels)[:, 0].real
			lowestEnergy = np.min(groundEnergies)
			BoltzDist = np.exp(-(groundEnergies - lowestEnergy) * h * 1e6 / (kB * T))
			BoltzDist /= BoltzDist.sum()
		else:
			BoltzDist = np.ones(groundDim) / groundDim

	if hand == 'Right':
		bottom, top = 1, groundDim + 1
	elif hand == 'Z':
		bottom, top = groundDim + 1, 2 * groundDim + 1
	elif hand == 'Left':
		bottom, top = 2 * groundDim + 1, excitedDim + 1
	else:
		raise ValueError(f"Unknown hand type '{hand}'")

	if Dline == 'D1':
		iteratorList = range(groundDim)
	elif Dline == 'D2':
		iteratorList = range(groundDim, excitedDim)
	else:
		raise ValueError(f"Unknown D-line '{Dline}'")

	for gg in range(groundDim):
		for ee in iteratorList:
			cleb = np.dot(groundLevels[gg][1:], excitedLevels[ee][bottom:top]).real
			cleb2 = cleb * cleb

			if cleb2 > strength_tol:
				dE = (-groundLevels[gg][0].real + excitedLevels[ee][0].real)
				strength_raw = (1.0 / 3.0) * cleb2
				strength_pop = strength_raw * BoltzDist[gg]

				transitions.append({
					'g_idx': gg,
					'e_idx': ee,
					'freq_MHz': dE,
					'clebsch': cleb,
					'clebsch2': cleb2,
					'strength_raw': strength_raw,
					'strength_pop': strength_pop,
					'ground_pop': BoltzDist[gg],
					'hand': hand,
				})

	return transitions


def FreqStren_raw_arrays(
	groundLevels,
	excitedLevels,
	groundDim,
	excitedDim,
	Dline,
	hand,
	BoltzmannFactor=True,
	T=293.16,
	custom_pop=None,
	strength_tol=0.0005
):
	trans = FreqStren_raw(
		groundLevels,
		excitedLevels,
		groundDim,
		excitedDim,
		Dline,
		hand,
		BoltzmannFactor=BoltzmannFactor,
		T=T,
		custom_pop=custom_pop,
		strength_tol=strength_tol
	)

	if len(trans) == 0:
		return (
			np.array([], dtype=int),
			np.array([], dtype=int),
			np.array([], dtype=float),
			np.array([], dtype=float),
			np.array([], dtype=float),
			np.array([], dtype=float),
		)

	g_idx = np.array([t['g_idx'] for t in trans], dtype=int)
	e_idx = np.array([t['e_idx'] for t in trans], dtype=int)
	freq_MHz = np.array([t['freq_MHz'] for t in trans], dtype=float)
	strength_raw = np.array([t['strength_raw'] for t in trans], dtype=float)
	strength_pop = np.array([t['strength_pop'] for t in trans], dtype=float)
	ground_pop = np.array([t['ground_pop'] for t in trans], dtype=float)

	return g_idx, e_idx, freq_MHz, strength_raw, strength_pop, ground_pop


def build_branching_matrix_raw(
	groundLevels,
	excitedLevels,
	groundDim,
	excitedDim,
	Dline,
	strength_tol=0.0
):
	"""
	Build spontaneous-emission branching probabilities e -> g.
	"""

	if Dline == 'D1':
		e_indices = list(range(groundDim))
	elif Dline == 'D2':
		e_indices = list(range(groundDim, excitedDim))
	else:
		raise ValueError(f"Unknown D-line '{Dline}'")

	pol_slices = {
		'Right': (1, groundDim + 1),
		'Z':     (groundDim + 1, 2 * groundDim + 1),
		'Left':  (2 * groundDim + 1, excitedDim + 1),
	}

	Ne = len(e_indices)
	Ng = groundDim

	B = np.zeros((Ne, Ng), dtype=float)
	details = []

	for jj, ee in enumerate(e_indices):
		per_ground = np.zeros(Ng, dtype=float)

		for gg in range(Ng):
			w_sum = 0.0
			for _, (bottom, top) in pol_slices.items():
				amp = np.dot(groundLevels[gg][1:], excitedLevels[ee][bottom:top]).real
				amp2 = amp * amp
				if amp2 > strength_tol:
					w_sum += amp2

			per_ground[gg] = (1.0 / 3.0) * w_sum

		total_weight = per_ground.sum()
		if total_weight > 0.0:
			B[jj, :] = per_ground / total_weight

		details.append({
			'e_idx': ee,
			'weights_raw': per_ground.copy(),
			'branching': B[jj, :].copy(),
			'sum': B[jj, :].sum(),
		})

	return B, np.array(e_indices, dtype=int), details


def build_raw_transition_table(
	groundLevels,
	excitedLevels,
	groundDim,
	excitedDim,
	Dline,
	BoltzmannFactor=True,
	T=293.16,
	custom_pop=None,
	strength_tol=0.0005
):
	hands = ['Right', 'Z', 'Left']
	transitions = []

	for hand in hands:
		transitions.extend(
			FreqStren_raw(
				groundLevels,
				excitedLevels,
				groundDim,
				excitedDim,
				Dline,
				hand,
				BoltzmannFactor=BoltzmannFactor,
				T=T,
				custom_pop=custom_pop,
				strength_tol=strength_tol
			)
		)

	by_hand = {
		'Right': [t for t in transitions if t['hand'] == 'Right'],
		'Z':     [t for t in transitions if t['hand'] == 'Z'],
		'Left':  [t for t in transitions if t['hand'] == 'Left'],
	}

	return transitions, by_hand


def build_raw_transition_table_arrays(
	groundLevels,
	excitedLevels,
	groundDim,
	excitedDim,
	Dline,
	BoltzmannFactor=True,
	T=293.16,
	custom_pop=None,
	strength_tol=0.0005
):
	transitions, by_hand = build_raw_transition_table(
		groundLevels,
		excitedLevels,
		groundDim,
		excitedDim,
		Dline,
		BoltzmannFactor=BoltzmannFactor,
		T=T,
		custom_pop=custom_pop,
		strength_tol=strength_tol
	)

	if len(transitions) == 0:
		return {
			'g_idx': np.array([], dtype=int),
			'e_idx': np.array([], dtype=int),
			'hand': np.array([], dtype=object),
			'freq_MHz': np.array([], dtype=float),
			'clebsch': np.array([], dtype=float),
			'clebsch2': np.array([], dtype=float),
			'strength_raw': np.array([], dtype=float),
			'strength_pop': np.array([], dtype=float),
			'ground_pop': np.array([], dtype=float),
		}, by_hand

	arr = {
		'g_idx': np.array([t['g_idx'] for t in transitions], dtype=int),
		'e_idx': np.array([t['e_idx'] for t in transitions], dtype=int),
		'hand': np.array([t['hand'] for t in transitions], dtype=object),
		'freq_MHz': np.array([t['freq_MHz'] for t in transitions], dtype=float),
		'clebsch': np.array([t['clebsch'] for t in transitions], dtype=float),
		'clebsch2': np.array([t['clebsch2'] for t in transitions], dtype=float),
		'strength_raw': np.array([t['strength_raw'] for t in transitions], dtype=float),
		'strength_pop': np.array([t['strength_pop'] for t in transitions], dtype=float),
		'ground_pop': np.array([t['ground_pop'] for t in transitions], dtype=float),
	}

	return arr, by_hand

# =========================================================
# INDEX HELPERS
# =========================================================
def gpos(k, gi, Ng, Ne):
	return k * (Ng + Ne) + gi


def epos(k, ej_local, Ng, Ne):
	return k * (Ng + Ne) + Ng + ej_local

# =========================================================
# GROUND-STATE POPULATION INPUTS
# =========================================================
def ground_state_population_vector(
	groundLevels,
	groundDim,
	BoltzmannFactor=True,
	T=293.16,
	custom_pop=None
):
	if custom_pop is not None:
		p_g = np.array(custom_pop, dtype=float)
		p_g /= p_g.sum()
		return p_g

	if BoltzmannFactor:
		groundEnergies = np.array(groundLevels)[:, 0].real
		lowestEnergy = np.min(groundEnergies)
		p_g = np.exp(-(groundEnergies - lowestEnergy) * h * 1e6 / (kB * T))
		p_g /= p_g.sum()
		return p_g

	return np.ones(groundDim, dtype=float) / groundDim

# =========================================================
# ONE-ISOTOPE MODEL BUILDERS
# =========================================================
def build_population_model_inputs_ag(
	isotope,
	Bfield,
	T_K,
	AgIsotopeShift,
	custom_pop=None,
	BoltzmannFactor=True,
	Dline='D2'
):
	ES = ht.Hamiltonian(isotope, Dline, 1.0, Bfield, AgIsotopeShift)

	p_g_in = ground_state_population_vector(
		ES.groundManifold,
		ES.ds,
		BoltzmannFactor=BoltzmannFactor,
		T=T_K,
		custom_pop=custom_pop
	)

	transitions, transitions_by_hand = build_raw_transition_table(
		ES.groundManifold,
		ES.excitedManifold,
		ES.ds,
		ES.dp,
		Dline,
		BoltzmannFactor=BoltzmannFactor,
		T=T_K,
		custom_pop=custom_pop
	)

	B_decay, e_used, decay_details = build_branching_matrix_raw(
		ES.groundManifold,
		ES.excitedManifold,
		ES.ds,
		ES.dp,
		Dline
	)

	e_local_map = {e_idx: j for j, e_idx in enumerate(e_used)}

	return {
		'ES': ES,
		'Ng': ES.ds,
		'Ne': len(e_used),
		'p_g_in': p_g_in,
		'transitions': transitions,
		'transitions_by_hand': transitions_by_hand,
		'B_decay': B_decay,
		'e_used': e_used,
		'e_local_map': e_local_map,
		'decay_details': decay_details,
	}

# =========================================================
# ONE-ISOTOPE SOLVER
# =========================================================
def solve_population_steady_state_one_detuning(
    det_MHz,
    model_inputs,
    v,
    dv,
    f0,
    K,
    wavenumber,
    gamma_rad_s,
    pump_params,
    gamma_transit_Hz,
    gamma_vcc_Hz,
    include_probe_pumping=True,
    renormalise_output=True
):
    """
    Solve one isotope at one detuning.
    """

    Ng = model_inputs['Ng']
    Ne = model_inputs['Ne']
    p_g_in = model_inputs['p_g_in']
    transitions = model_inputs['transitions']
    e_local_map = model_inputs['e_local_map']
    B_decay = model_inputs['B_decay']

    Nv = len(v)
    Ntot = Nv * (Ng + Ne)

    gamma_transit = 2.0 * np.pi * gamma_transit_Hz
    gamma_vcc = 2.0 * np.pi * gamma_vcc_Hz

    I_pump = pump_params.get('I_pump', 0.0)
    I_probe = pump_params.get('I_probe', 0.0)
    I_sat = pump_params.get('I_sat', 1.0)
    eta_pump = pump_params.get('eta_pump', 1.0)
    eta_probe = pump_params.get('eta_probe', 1.0)
    pump_pol = pump_params.get('pol', 'Left')
    probe_pol = pump_params.get('probe_pol', pump_pol)

    s0_pump = I_pump / I_sat
    s0_probe = I_probe / I_sat if include_probe_pumping else 0.0

    A = np.zeros((Ntot, Ntot), dtype=float)
    b = np.zeros(Ntot, dtype=float)

    # -------------------------------------------------
    # Base transit source/loss
    # -------------------------------------------------
    for k in range(Nv):
        for gi in range(Ng):
            pg = gpos(k, gi, Ng, Ne)
            A[pg, pg] += gamma_transit
            b[pg] += gamma_transit * p_g_in[gi] * f0[k]

        for ej in range(Ne):
            pe = epos(k, ej, Ng, Ne)
            A[pe, pe] += gamma_rad_s + gamma_transit

    # -------------------------------------------------
    # Ground-state VCC: gamma_vcc * (I - K)
    # -------------------------------------------------
    if gamma_vcc != 0.0:
        for gi in range(Ng):
            for kout in range(Nv):
                row = gpos(kout, gi, Ng, Ne)
                A[row, row] += gamma_vcc
                for kin in range(Nv):
                    col = gpos(kin, gi, Ng, Ne)
                    A[row, col] -= gamma_vcc * K[kout, kin]

    # -------------------------------------------------
    # Optical pumping
    # -------------------------------------------------
    doppler_MHz = (wavenumber * v) / (2.0 * np.pi * 1.0e6)

    for t in transitions:
        gi = t['g_idx']
        e_abs = t['e_idx']
        if e_abs not in e_local_map:
            continue
        ej = e_local_map[e_abs]

        freq0 = t['freq_MHz']
        strength_raw = t['strength_raw']
        hand = t['hand']

        pump_active = (hand == pump_pol)
        probe_active = (hand == probe_pol)

        for k in range(Nv):
            delta_pump = det_MHz - freq0 - doppler_MHz[k]
            delta_probe = det_MHz - freq0 + doppler_MHz[k]

            s_pump = 0.0
            s_probe = 0.0

            if pump_active and s0_pump > 0.0:
                s_pump = eta_pump * s0_pump * strength_raw * _sat_lineshape(delta_pump, gamma_rad_s)

            if probe_active and include_probe_pumping and s0_probe > 0.0:
                s_probe = eta_probe * s0_probe * strength_raw * _sat_lineshape(delta_probe, gamma_rad_s)

            s_tot = s_pump + s_probe
            if s_tot == 0.0:
                continue

            Rge = 0.5 * gamma_rad_s * s_tot / (1.0 + s_tot)

            pg = gpos(k, gi, Ng, Ne)
            pe = epos(k, ej, Ng, Ne)

            # remove from ground, add to excited
            A[pg, pg] += Rge
            A[pe, pg] -= Rge

    # -------------------------------------------------
    # Spontaneous decay
    # -------------------------------------------------
    for ej in range(Ne):
        for k in range(Nv):
            pe = epos(k, ej, Ng, Ne)
            for gi in range(Ng):
                pg = gpos(k, gi, Ng, Ne)
                A[pg, pe] -= gamma_rad_s * B_decay[ej, gi]

    x = np.linalg.solve(A, b)

    n_g = np.zeros((Nv, Ng), dtype=float)
    n_e = np.zeros((Nv, Ne), dtype=float)

    for k in range(Nv):
        for gi in range(Ng):
            n_g[k, gi] = x[gpos(k, gi, Ng, Ne)]
        for ej in range(Ne):
            n_e[k, ej] = x[epos(k, ej, Ng, Ne)]

    # -------------------------------------------------
    # Optional clean renormalisation
    # -------------------------------------------------
    if renormalise_output:
        total = (np.sum(n_g) + np.sum(n_e)) * dv
        if total > 0.0:
            n_g /= total
            n_e /= total

    return n_g, n_e, A, b

# =========================================================
# BOTH-ISOTOPE AG HELPERS
# =========================================================
def get_ag_isotope_mass_from_label(label):
	if label == '107':
		return ac.Ag107.mass
	if label == '109':
		return ac.Ag109.mass
	raise ValueError(f"Unknown Ag isotope label '{label}'")


def build_population_model_inputs_ag_all(
	Bfield,
	T_K,
	AgIsotopeShift,
	Ag107frac=51.839 / 100.0,
	custom_pop=None,
	BoltzmannFactor=True,
	Dline='D2',
	Isotope_Combination=0
):
	"""
	Build model inputs for Ag107 and/or Ag109.
	"""
	Ag109frac = 1.0 - Ag107frac

	if Isotope_Combination == 1:
		Ag107frac_eff, Ag109frac_eff = Ag107frac, 0.0
	elif Isotope_Combination == 2:
		Ag107frac_eff, Ag109frac_eff = 0.0, Ag109frac
	elif Isotope_Combination == 0:
		Ag107frac_eff, Ag109frac_eff = Ag107frac, Ag109frac
	else:
		raise ValueError("Isotope_Combination must be 0, 1, or 2")

	isotopes = {}
	fractions = {}
	active_labels = []

	if Ag107frac_eff > 0.0:
		isotopes['107'] = build_population_model_inputs_ag(
			isotope='Ag107',
			Bfield=Bfield,
			T_K=T_K,
			AgIsotopeShift=AgIsotopeShift,
			custom_pop=custom_pop,
			BoltzmannFactor=BoltzmannFactor,
			Dline=Dline
		)
		fractions['107'] = Ag107frac_eff
		active_labels.append('107')

	if Ag109frac_eff > 0.0:
		isotopes['109'] = build_population_model_inputs_ag(
			isotope='Ag109',
			Bfield=Bfield,
			T_K=T_K,
			AgIsotopeShift=AgIsotopeShift,
			custom_pop=custom_pop,
			BoltzmannFactor=BoltzmannFactor,
			Dline=Dline
		)
		fractions['109'] = Ag109frac_eff
		active_labels.append('109')

	if not active_labels:
		raise ValueError("No active isotopes selected")

	return {
		'isotopes': isotopes,
		'fractions': fractions,
		'active_labels': active_labels,
	}


def build_velocity_grids_ag_all(
	model_inputs_all,
	DoppTemp_K,
	Nv=301,
	vmax_sigma=4.0,
	subdop_params=None
):
	if subdop_params is None:
		subdop_params = {}

	grids = {}

	for label in model_inputs_all['active_labels']:
		atom_mass = get_ag_isotope_mass_from_label(label)
		v, dv, f0, u = _build_velocity_grid(
			DoppTemp_K,
			atom_mass,
			Nv=Nv,
			vmax_sigma=vmax_sigma
		)
		W = _build_vcc_kernel(v, f0, subdop_params)
		K = W * dv

		grids[label] = {
			'v': v,
			'dv': dv,
			'f0': f0,
			'u': u,
			'W': W,
			'K': K,
			'mass': atom_mass,
		}

	return grids


def solve_population_steady_state_one_detuning_ag_all(
	det_MHz,
	model_inputs_all,
	grids_all,
	wavenumber,
	gamma_rad_s,
	pump_params,
	gamma_transit_Hz,
	gamma_vcc_Hz,
	include_probe_pumping=True
):
	per_isotope = {}
	ground_by_isotope = {}
	excited_by_isotope = {}
	total_by_isotope = {}

	for label in model_inputs_all['active_labels']:
		mi = model_inputs_all['isotopes'][label]
		gg = grids_all[label]

		n_g, n_e, A, b = solve_population_steady_state_one_detuning(
			det_MHz=det_MHz,
			model_inputs=mi,
			v=gg['v'],
			dv=gg['dv'],
			f0=gg['f0'],
			K=gg['K'],
			wavenumber=wavenumber,
			gamma_rad_s=gamma_rad_s,
			pump_params=pump_params,
			gamma_transit_Hz=gamma_transit_Hz,
			gamma_vcc_Hz=gamma_vcc_Hz,
			include_probe_pumping=include_probe_pumping
		)

		per_isotope[label] = {
			'n_g': n_g,
			'n_e': n_e,
			'A': A,
			'b': b,
			'v': gg['v'],
			'dv': gg['dv'],
			'f0': gg['f0'],
		}

		ground_int = np.sum(n_g) * gg['dv']
		excited_int = np.sum(n_e) * gg['dv']
		total_int = ground_int + excited_int

		ground_by_isotope[label] = ground_int
		excited_by_isotope[label] = excited_int
		total_by_isotope[label] = total_int

	ground_total = sum(
		model_inputs_all['fractions'][label] * ground_by_isotope[label]
		for label in model_inputs_all['active_labels']
	)
	excited_total = sum(
		model_inputs_all['fractions'][label] * excited_by_isotope[label]
		for label in model_inputs_all['active_labels']
	)
	population_total = ground_total + excited_total

	return {
		'per_isotope': per_isotope,
		'fractions': dict(model_inputs_all['fractions']),
		'active_labels': list(model_inputs_all['active_labels']),
		'integrals': {
			'ground_total': ground_total,
			'excited_total': excited_total,
			'population_total': population_total,
			'ground_by_isotope': ground_by_isotope,
			'excited_by_isotope': excited_by_isotope,
			'total_by_isotope': total_by_isotope,
		}
	}


def scan_population_steady_state_ag_all(
	detuning_array_MHz,
	model_inputs_all,
	grids_all,
	wavenumber,
	gamma_rad_s,
	pump_params,
	gamma_transit_Hz,
	gamma_vcc_Hz,
	include_probe_pumping=True
):
	detuning_array_MHz = np.asarray(detuning_array_MHz, dtype=float)

	ground_total = np.zeros_like(detuning_array_MHz, dtype=float)
	excited_total = np.zeros_like(detuning_array_MHz, dtype=float)
	population_total = np.zeros_like(detuning_array_MHz, dtype=float)

	ground_by_isotope = {
		label: np.zeros_like(detuning_array_MHz, dtype=float)
		for label in model_inputs_all['active_labels']
	}
	excited_by_isotope = {
		label: np.zeros_like(detuning_array_MHz, dtype=float)
		for label in model_inputs_all['active_labels']
	}
	total_by_isotope = {
		label: np.zeros_like(detuning_array_MHz, dtype=float)
		for label in model_inputs_all['active_labels']
	}

	solved = []

	for i, det_MHz in enumerate(detuning_array_MHz):
		out = solve_population_steady_state_one_detuning_ag_all(
			det_MHz=det_MHz,
			model_inputs_all=model_inputs_all,
			grids_all=grids_all,
			wavenumber=wavenumber,
			gamma_rad_s=gamma_rad_s,
			pump_params=pump_params,
			gamma_transit_Hz=gamma_transit_Hz,
			gamma_vcc_Hz=gamma_vcc_Hz,
			include_probe_pumping=include_probe_pumping
		)

		solved.append(out)

		ints = out['integrals']
		ground_total[i] = ints['ground_total']
		excited_total[i] = ints['excited_total']
		population_total[i] = ints['population_total']

		for label in model_inputs_all['active_labels']:
			ground_by_isotope[label][i] = ints['ground_by_isotope'][label]
			excited_by_isotope[label][i] = ints['excited_by_isotope'][label]
			total_by_isotope[label][i] = ints['total_by_isotope'][label]

	return {
		'detuning_MHz': detuning_array_MHz,
		'detuning_GHz': detuning_array_MHz / 1e3,
		'ground_total': ground_total,
		'excited_total': excited_total,
		'population_total': population_total,
		'ground_by_isotope': ground_by_isotope,
		'excited_by_isotope': excited_by_isotope,
		'total_by_isotope': total_by_isotope,
		'solved': solved,
	}

# =========================================================
# CHI FROM SOLVED POPULATIONS
# =========================================================
def _prepare_isotope_chi_inputs_ag(
	isotope,
	Bfield,
	T_K,
	AgIsotopeShift,
	custom_pop=None,
	BoltzmannFactor=True,
	Dline='D2',
	strength_tol=0.0005
):
	ES = ht.Hamiltonian(isotope, Dline, 1.0, Bfield, AgIsotopeShift)

	if Dline == 'D1':
		e_used = np.array(list(range(ES.ds)), dtype=int)
	elif Dline == 'D2':
		e_used = np.array(list(range(ES.ds, ES.dp)), dtype=int)
	else:
		raise ValueError(f"Unknown D-line '{Dline}'")

	e_local_map = {e_idx: j for j, e_idx in enumerate(e_used)}

	by_hand = {}
	for hand in ['Left', 'Right', 'Z']:
		by_hand[hand] = FreqStren_raw(
			ES.groundManifold,
			ES.excitedManifold,
			ES.ds,
			ES.dp,
			Dline,
			hand,
			BoltzmannFactor=BoltzmannFactor,
			T=T_K,
			custom_pop=custom_pop,
			strength_tol=strength_tol
		)

	return {
		'ES': ES,
		'Ng': ES.ds,
		'Ne': len(e_used),
		'e_used': e_used,
		'e_local_map': e_local_map,
		'transitions_by_hand': by_hand,
	}


def _chi_from_populations_one_isotope(
    X,
    isotope_fraction,
    chi_inputs,
    solved_scan_per_isotope,
    wavenumber,
    gamma_rad_s,
    prefactor,
    p_g_in
):
    """
    Build ONLY the population-induced correction to chi for one isotope.

    This is normalised so that if the solved populations equal the thermal
    baseline:
        n_g(v,gi) = p_g_in[gi] * f0(v)
        n_e(v,ej) = 0
    then delta_chi = 0 exactly.
    """
    X = np.asarray(X, dtype=float)

    v = solved_scan_per_isotope['v']
    dv = solved_scan_per_isotope['dv']
    f0 = solved_scan_per_isotope['f0']
    n_g = solved_scan_per_isotope['n_g']   # shape (Ndet, Nv, Ng)
    n_e = solved_scan_per_isotope['n_e']   # shape (Ndet, Nv, Ne)

    e_local_map = chi_inputs['e_local_map']
    transitions_by_hand = chi_inputs['transitions_by_hand']

    chi = {
        'Left': np.zeros(len(X), dtype=complex),
        'Right': np.zeros(len(X), dtype=complex),
        'Z': np.zeros(len(X), dtype=complex),
    }

    doppler_MHz = (wavenumber * v) / (2.0 * np.pi * 1.0e6)

    for hand in ['Left', 'Right', 'Z']:
        for j, det in enumerate(X):
            accum = 0.0j

            for t in transitions_by_hand[hand]:
                gi = t['g_idx']
                e_abs = t['e_idx']
                if e_abs not in e_local_map:
                    continue

                ej = e_local_map[e_abs]
                freq0 = t['freq_MHz']
                strength_raw = t['strength_raw']

                delta_probe = det - freq0 + doppler_MHz
                resp = _lorentz_complex(delta_probe, gamma_rad_s)

                # thermal weak-probe baseline for this transition
                baseline_ground = p_g_in[gi] * f0

                # correction relative to baseline
                delta_pop = (n_g[j, :, gi] - n_e[j, :, ej]) - baseline_ground

                accum += strength_raw * np.sum(delta_pop * resp) * dv

            chi[hand][j] = isotope_fraction * prefactor * accum

    return chi['Left'], chi['Right'], chi['Z']


def calc_chi_subdoppler_agd2_population_scan(
    X,
    p_dict,
    include_probe_pumping=False,
    return_details=False
):
    """
    Fully population-resolved Ag D2 susceptibility.

    Normalisation strategy:
        chi_total = chi_legacy_weak_probe + delta_chi_population

    so that with:
        pump = 0,
        gamma_vcc = 0,
        include_probe_pumping = False

    the result reduces to legacy calc_chi().
    """
    Elem = p_dict.get('Elem', 'Ag')
    Dline = p_dict.get('Dline', 'D2')
    if Elem != 'Ag':
        raise ValueError("This function only supports Elem='Ag'")
    if Dline != 'D2':
        raise ValueError("This function only supports Dline='D2'")

    X = np.asarray(X, dtype=float)

    T_C = p_dict.get('T', 20.0)
    Bfield = p_dict.get('Bfield', 0.0)
    GammaBuf_MHz = p_dict.get('GammaBuf', 0.0)
    Constrain = p_dict.get('Constrain', True)
    DoppTemp_C = p_dict.get('DoppTemp', T_C)
    Ag107frac = p_dict.get('Ag107frac', 51.839) / 100.0
    CustomPop = p_dict.get('CustomPop', None)
    AgIsotopeShift = p_dict.get('AgIsotope_shift', p_dict_defaults['AgIsotope_shift'])
    Isotope_Combination = p_dict.get('Isotope_Combination', 0)

    pump_params = p_dict.get('pump_params', {})
    subdop_params = p_dict.get('subdop_params', {})

    if Bfield == 0.0:
        Bfield = 1e-4

    if Constrain:
        DoppTemp_C = T_C

    T_K = T_C + 273.15
    DoppTemp_K = DoppTemp_C + 273.15

    # -------------------------------------------------
    # 1. Legacy baseline
    # -------------------------------------------------
    p_weak = dict(p_dict)
    p_weak['SubDoppler'] = False
    chi_plus_base, chi_minus_base, chi_z_base = calc_chi(X, p_weak)

    # -------------------------------------------------
    # 2. Population-model inputs
    # -------------------------------------------------
    model_inputs_all = build_population_model_inputs_ag_all(
        Bfield=Bfield,
        T_K=T_K,
        AgIsotopeShift=AgIsotopeShift,
        Ag107frac=Ag107frac,
        custom_pop=CustomPop,
        BoltzmannFactor=True,
        Dline='D2',
        Isotope_Combination=Isotope_Combination
    )

    grids_all = build_velocity_grids_ag_all(
        model_inputs_all,
        DoppTemp_K=DoppTemp_K,
        Nv=subdop_params.get('Nv', 301),
        vmax_sigma=subdop_params.get('vmax_sigma', 4.0),
        subdop_params=subdop_params
    )

    transitionConst = ac.AgD2Transition
    NDensity = p_dict.get('AgNumden', p_dict_defaults['AgNumden'])

    gamma0 = 2.0 * np.pi * transitionConst.NatGamma * 1.0e6
    gammaself = 2.0 * np.pi * gamma0 * NDensity * 1.414213562373095 * (
        transitionConst.wavelength / (2.0 * np.pi)
    ) ** 3
    gamma_rad_s = gamma0 + gammaself + 2.0 * np.pi * GammaBuf_MHz * 1.0e6

    wavenumber = transitionConst.wavevectorMagnitude
    dipole = transitionConst.dipoleStrength
    prefactor = 2.0 * NDensity * dipole**2 / (hbar * e0)

    # -------------------------------------------------
    # 3. Solve populations across scan
    # -------------------------------------------------
    scan = scan_population_steady_state_ag_all(
        X,
        model_inputs_all=model_inputs_all,
        grids_all=grids_all,
        wavenumber=wavenumber,
        gamma_rad_s=gamma_rad_s,
        pump_params=pump_params,
        gamma_transit_Hz=subdop_params.get('gamma_transit_Hz', 2.0e4),
        gamma_vcc_Hz=subdop_params.get('gamma_vcc_Hz', 0.0),
        include_probe_pumping=include_probe_pumping
    )

    # -------------------------------------------------
    # 4. Population correction
    # -------------------------------------------------
    delta_chi_plus = np.zeros(len(X), dtype=complex)
    delta_chi_minus = np.zeros(len(X), dtype=complex)
    delta_chi_z = np.zeros(len(X), dtype=complex)

    chi_inputs_all = {}

    for label in model_inputs_all['active_labels']:
        isotope = 'Ag107' if label == '107' else 'Ag109'

        chi_inputs = _prepare_isotope_chi_inputs_ag(
            isotope=isotope,
            Bfield=Bfield,
            T_K=T_K,
            AgIsotopeShift=AgIsotopeShift,
            custom_pop=CustomPop,
            BoltzmannFactor=True,
            Dline='D2'
        )
        chi_inputs_all[label] = chi_inputs

        solved_per_iso = {
            'v': scan['solved'][0]['per_isotope'][label]['v'],
            'dv': scan['solved'][0]['per_isotope'][label]['dv'],
            'f0': scan['solved'][0]['per_isotope'][label]['f0'],
            'n_g': np.array([scan['solved'][j]['per_isotope'][label]['n_g'] for j in range(len(X))]),
            'n_e': np.array([scan['solved'][j]['per_isotope'][label]['n_e'] for j in range(len(X))]),
        }

        frac = model_inputs_all['fractions'][label]
        p_g_in = model_inputs_all['isotopes'][label]['p_g_in']

        dL, dR, dZ = _chi_from_populations_one_isotope(
            X=X,
            isotope_fraction=frac,
            chi_inputs=chi_inputs,
            solved_scan_per_isotope=solved_per_iso,
            wavenumber=wavenumber,
            gamma_rad_s=gamma_rad_s,
            prefactor=prefactor,
            p_g_in=p_g_in
        )

        delta_chi_plus += dL
        delta_chi_minus += dR
        delta_chi_z += dZ

    chi_plus = chi_plus_base + delta_chi_plus
    chi_minus = chi_minus_base + delta_chi_minus
    chi_z = chi_z_base + delta_chi_z

    if return_details:
        return chi_plus, chi_minus, chi_z, {
            'scan': scan,
            'model_inputs_all': model_inputs_all,
            'grids_all': grids_all,
            'chi_inputs_all': chi_inputs_all,
            'gamma_rad_s': gamma_rad_s,
            'wavenumber': wavenumber,
            'prefactor': prefactor,
            'chi_base': (chi_plus_base, chi_minus_base, chi_z_base),
            'delta_chi': (delta_chi_plus, delta_chi_minus, delta_chi_z),
        }

    return chi_plus, chi_minus, chi_z

# =========================================================
# V2 ENTRY POINT
# =========================================================

def calc_chi_subdoppler_agd2(X, p_dict, pump_params=None, subdop_params=None, return_components=False):
    """
    V2 wrapper: population-resolved Ag D2 model.

    Default behaviour is set so that baseline comparison against legacy calc_chi
    is straightforward:
        - probe pumping OFF unless explicitly requested
    """
    p_local = dict(p_dict)

    if pump_params is not None:
        p_local['pump_params'] = pump_params
    if subdop_params is not None:
        p_local['subdop_params'] = subdop_params

    include_probe_pumping = p_local.get('include_probe_pumping', False)

    out = calc_chi_subdoppler_agd2_population_scan(
        X,
        p_local,
        include_probe_pumping=include_probe_pumping,
        return_details=return_components
    )

    if return_components:
        chi_plus, chi_minus, chi_z, details = out
        return chi_plus, chi_minus, chi_z, details

    return out


def get_spectra(X, E_in, p_dict, outputs=None):
    """
    V2 get_spectra:
    - legacy calc_chi for ordinary mode
    - population-resolved Ag D2 solver for SubDoppler mode
    """
    SubDoppler = p_dict.get('SubDoppler', p_dict_defaults['SubDoppler'])

    if SubDoppler:
        if p_dict.get('Elem', p_dict_defaults['Elem']) == 'Ag' and p_dict.get('Dline', p_dict_defaults['Dline']) == 'D2':
            ChiPlus, ChiMinus, ChiZ = calc_chi_subdoppler_agd2_population_scan(
                X,
                p_dict,
                include_probe_pumping=p_dict.get('include_probe_pumping', False),
                return_details=False
            )
        else:
            raise ValueError("SubDoppler mode is currently only implemented for Elem='Ag' and Dline='D2'")
    else:
        ChiPlus, ChiMinus, ChiZ = calc_chi(X, p_dict)

    E_out, _ = get_Efield(X, E_in, [ChiPlus, ChiMinus, ChiZ], p_dict)

    E_in_arr = np.array(E_in)
    if E_in_arr.shape == (3,):
        E_in_arr = np.array([
            np.ones(len(X)) * E_in_arr[0],
            np.ones(len(X)) * E_in_arr[1],
            np.ones(len(X)) * E_in_arr[2]
        ])

    I_in = (E_in_arr * E_in_arr.conjugate()).sum(axis=0)

    from libs import rotations as rot
    from libs import convert_basis as cb

    S0 = ((E_out * E_out.conjugate()).sum(axis=0) / I_in).real

    Ex = np.array(rot.HorizPol_xy * E_out[:2])
    Ey = np.array(rot.VertPol_xy * E_out[:2])
    Ix = (Ex * Ex.conjugate()).sum(axis=0) / I_in
    Iy = (Ey * Ey.conjugate()).sum(axis=0) / I_in
    S1 = Ix - Iy

    E_P45 = np.array(rot.LPol_P45_xy * E_out[:2])
    E_M45 = np.array(rot.LPol_M45_xy * E_out[:2])
    I_P45 = (E_P45 * E_P45.conjugate()).sum(axis=0) / I_in
    I_M45 = (E_M45 * E_M45.conjugate()).sum(axis=0) / I_in
    S2 = I_P45 - I_M45

    E_out_lrz = cb.xyz_to_lrz(E_out)
    El = np.array(rot.CPol_L_lr * E_out_lrz[:2])
    Er = np.array(rot.CPol_R_lr * E_out_lrz[:2])
    Il = (El * El.conjugate()).sum(axis=0) / I_in
    Ir = (Er * Er.conjugate()).sum(axis=0) / I_in
    S3 = Ir - Il

    op = {
        'S0': np.array([S0]),
        'S1': np.array([S1.real]),
        'S2': np.array([S2.real]),
        'S3': np.array([S3.real]),
        'Ix': np.array([Ix.real]),
        'Iy': np.array([Iy.real]),
        'E_out': np.array([E_out]),
    }

    if outputs is None or 'All' in outputs:
        return op['S0'], op['S1'], op['S2'], op['S3'], op['Ix'], op['Iy']
    return [op[o] for o in outputs]