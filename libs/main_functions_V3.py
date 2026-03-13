import numpy as np
from scipy.linalg import solve
from scipy.constants import physical_constants, epsilon_0, hbar, c, h

from libs import atomic_constants as ac
from libs import Hamiltonian as ht

from tqdm import tqdm

# ---------------------------------------------------------
# Import stable helpers from the original module
# ---------------------------------------------------------
from libs.main_functions import (
	p_dict_defaults,
	calc_chi,          # legacy weak-probe model
	get_Efield,        # propagation
	chi_to_S0,         # convert chi -> transmission
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
# BASIC HELPERS
# =========================================================
def _lorentz_complex(delta_MHz, gamma_rad_s):
	delta_rad_s = 2.0 * np.pi * 1.0e6 * delta_MHz
	return 1.0 / (-delta_rad_s - 1j * gamma_rad_s / 2.0)

def _maxwell_1d(v, u):
	"""
	1D Maxwell-Boltzmann distribution for one velocity component.
	u = sqrt(2 k_B T / m)
	"""
	return np.exp(-(v / u) ** 2) / (np.sqrt(np.pi) * u)

def _build_velocity_grid(DoppTemp_K, atom_mass, Nv=301, vmax_sigma=4.0):
	"""
	Build a 1D velocity grid and properly normalised thermal distribution.
	"""
	u = np.sqrt(2.0 * kB * DoppTemp_K / atom_mass)
	v = np.linspace(-vmax_sigma * u, vmax_sigma * u, Nv)
	dv = v[1] - v[0]
	f0 = _maxwell_1d(v, u)
	f0 /= np.sum(f0) * dv
	return v, dv, f0, u

def ground_state_population_vector(
	groundLevels,
	groundDim,
	BoltzmannFactor=True,
	T=293.16,
	custom_pop=None
):
	"""
	Return the normalised incoming ground-state population vector.
	"""
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
	
	for d in details:
		if d['e_idx'] in [4, 5, 7, 8, 9]:
			print(d['e_idx'], d['weights_raw'], d['branching'], d['sum'])

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

# =========================================================
# DENSITY MATRIX MODEL INPUTS
# =========================================================

def build_dm_model_inputs_ag(
	isotope,
	Bfield,
	T_K,
	AgIsotopeShift,
	custom_pop=None,
	BoltzmannFactor=True,
	Dline='D2',
	strength_tol=0.0005
):
	"""
	Build state-resolved inputs for one-isotope Ag density-matrix model.
	"""

	ES = ht.Hamiltonian(isotope, Dline, 1.0, Bfield, AgIsotopeShift)

	Ng = ES.ds

	if Dline == 'D1':
		e_used = np.array(list(range(ES.ds)), dtype=int)
	elif Dline == 'D2':
		e_used = np.array(list(range(ES.ds, ES.dp)), dtype=int)
	else:
		raise ValueError(f"Unknown D-line '{Dline}'")

	Ne = len(e_used)
	e_local_map = {e_abs: j for j, e_abs in enumerate(e_used)}

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
		custom_pop=custom_pop,
		strength_tol=strength_tol
	)

	B_decay, _, decay_details = build_branching_matrix_raw(
		ES.groundManifold,
		ES.excitedManifold,
		ES.ds,
		ES.dp,
		Dline,
		strength_tol=0.0
	)

	E_g_MHz = np.array([ES.groundManifold[g][0].real for g in range(Ng)], dtype=float)
	E_e_MHz = np.array([ES.excitedManifold[e_abs][0].real for e_abs in e_used], dtype=float)

	omega_g = 2.0 * np.pi * 1.0e6 * E_g_MHz
	omega_e = 2.0 * np.pi * 1.0e6 * E_e_MHz

	return {
		'ES': ES,
		'Ng': Ng,
		'Ne': Ne,
		'Ns': Ng + Ne,
		'e_used': e_used,
		'e_local_map': e_local_map,
		'p_g_in': p_g_in,
		'transitions': transitions,
		'transitions_by_hand': transitions_by_hand,
		'B_decay': B_decay,
		'decay_details': decay_details,
		'omega_g': omega_g,
		'omega_e': omega_e,
	}

# =========================================================
# INDEX HELPERS
# =========================================================
def _dm_g_index(gi, Ng):
	return gi

def _dm_e_index(ej, Ng):
	return Ng + ej

def vec_index(a, b_, Ns):
		return a * Ns + b_

# =========================================================
# HAMILTONIAN + LIOUVILLIANS
# =========================================================

def build_pump_hamiltonian_ag(
	model_inputs,
	det_MHz,
	v,
	wavenumber,
	pump_params,
	gamma_rad_s
):
	Ng = model_inputs['Ng']
	Ne = model_inputs['Ne']
	Ns = model_inputs['Ns']
	e_local_map = model_inputs['e_local_map']
	transitions_by_hand = model_inputs['transitions_by_hand']
	omega_g = model_inputs['omega_g']
	omega_e = model_inputs['omega_e']

	pump_pol = pump_params.get('pol', 'Left')
	I_pump = pump_params.get('I_pump', 0.0)
	I_sat = pump_params.get('I_sat', 1.0)
	eta_pump = pump_params.get('eta_pump', 1.0)

	s0_pump = I_pump / I_sat
	H = np.zeros((Ns, Ns), dtype=complex)

	doppler_MHz = (wavenumber * v) / (2.0 * np.pi * 1.0e6)

	# Counter-propagating pump vs probe:
	omega_L_eff = 2.0 * np.pi * 1.0e6 * (det_MHz - doppler_MHz)

	for gi in range(Ng):
		g = _dm_g_index(gi, Ng)
		H[g, g] = -omega_g[gi]

	for ej in range(Ne):
		e = _dm_e_index(ej, Ng)
		H[e, e] = -(omega_e[ej] - omega_L_eff)

	for t in transitions_by_hand[pump_pol]:
		gi = t['g_idx']
		e_abs = t['e_idx']
		if e_abs not in e_local_map:
			continue

		ej = e_local_map[e_abs]
		g = _dm_g_index(gi, Ng)
		e = _dm_e_index(ej, Ng)

		amp = t['clebsch'] / np.sqrt(3.0)
		Omega = eta_pump * gamma_rad_s * np.sqrt(max(2.0 * s0_pump, 0.0)) * amp

		H[g, e] += 0.5 * Omega
		H[e, g] += 0.5 * np.conjugate(Omega)

	return H

def build_probe_hamiltonian_ag(
	model_inputs,
	det_MHz,
	v,
	wavenumber,
	probe_params,
	gamma_rad_s,
	branch
):
	Ng = model_inputs['Ng']
	Ne = model_inputs['Ne']
	Ns = model_inputs['Ns']
	e_local_map = model_inputs['e_local_map']
	transitions_by_hand = model_inputs['transitions_by_hand']
	omega_g = model_inputs['omega_g']
	omega_e = model_inputs['omega_e']

	I_probe = probe_params.get('I_probe', 0.0)
	I_sat = probe_params.get('I_sat', 1.0)
	eta_probe = probe_params.get('eta_probe', 1.0)

	s0_probe = I_probe / I_sat
	H = np.zeros((Ns, Ns), dtype=complex)

	doppler_MHz = (wavenumber * v) / (2.0 * np.pi * 1.0e6)
	omega_L_eff = 2.0 * np.pi * 1.0e6 * (det_MHz + doppler_MHz)

	for gi in range(Ng):
		g = _dm_g_index(gi, Ng)
		H[g, g] = -omega_g[gi]

	for ej in range(Ne):
		e = _dm_e_index(ej, Ng)
		H[e, e] = -(omega_e[ej] - omega_L_eff)

	for t in transitions_by_hand[branch]:
		gi = t['g_idx']
		e_abs = t['e_idx']
		if e_abs not in e_local_map:
			continue

		ej = e_local_map[e_abs]
		g = _dm_g_index(gi, Ng)
		e = _dm_e_index(ej, Ng)

		amp = t['clebsch'] / np.sqrt(3.0)
		Omega = eta_probe * gamma_rad_s * np.sqrt(max(2.0 * s0_probe, 0.0)) * amp

		H[g, e] += 0.5 * Omega
		H[e, g] += 0.5 * np.conjugate(Omega)

	return H

def build_hamiltonian_liouvillian(H):
	Ns = H.shape[0]
	I = np.eye(Ns, dtype=complex)
	return -1j * (np.kron(I, H) - np.kron(H.T, I))

def build_spontaneous_decay_superoperator_ag(model_inputs, gamma_rad_s):
	"""
	Lindblad spontaneous-emission superoperator in Liouville space.
	"""

	Ng = model_inputs['Ng']
	Ne = model_inputs['Ne']
	Ns = model_inputs['Ns']
	B_decay = model_inputs['B_decay']

	dim = Ns * Ns
	L = np.zeros((dim, dim), dtype=complex)

	def vec_index(a, b):
		return a * Ns + b

	for ej in range(Ne):
		e = _dm_e_index(ej, Ng)

		for gi in range(Ng):
			g = _dm_g_index(gi, Ng)
			br = B_decay[ej, gi]
			L[vec_index(g, g), vec_index(e, e)] += gamma_rad_s * br

		L[vec_index(e, e), vec_index(e, e)] += -gamma_rad_s

	for gi in range(Ng):
		g = _dm_g_index(gi, Ng)
		for ej in range(Ne):
			e = _dm_e_index(ej, Ng)
			L[vec_index(g, e), vec_index(g, e)] += -0.5 * gamma_rad_s
			L[vec_index(e, g), vec_index(e, g)] += -0.5 * gamma_rad_s

	for ej1 in range(Ne):
		e1 = _dm_e_index(ej1, Ng)
		for ej2 in range(Ne):
			e2 = _dm_e_index(ej2, Ng)
			if ej1 != ej2:
				L[vec_index(e1, e2), vec_index(e1, e2)] += -gamma_rad_s

	return L

def build_transit_superoperator_ag(model_inputs, gamma_transit_Hz):
	"""
	Transit relaxation and repopulation.
	"""

	Ng = model_inputs['Ng']
	Ns = model_inputs['Ns']
	p_g_in = model_inputs['p_g_in']

	gamma_t = 2.0 * np.pi * gamma_transit_Hz

	dim = Ns * Ns
	L = np.zeros((dim, dim), dtype=complex)
	b = np.zeros(dim, dtype=complex)

	def vec_index(a, b_):
		return a * Ns + b_

	for a in range(Ns):
		for b_ in range(Ns):
			L[vec_index(a, b_), vec_index(a, b_)] += -gamma_t

	for gi in range(Ng):
		g = _dm_g_index(gi, Ng)
		b[vec_index(g, g)] += gamma_t * p_g_in[gi]

	return L, b

# =========================================================
# STEADY-STATE DENSITY MATRIX SOLVER
# =========================================================

def solve_dm_steady_state_one_velocity_ag(
	det_MHz,
	v,
	model_inputs,
	wavenumber,
	gamma_rad_s,
	pump_params,
	gamma_transit_Hz=0.0
):
	Ns = model_inputs['Ns']

	H = build_pump_hamiltonian_ag(
		model_inputs=model_inputs,
		det_MHz=det_MHz,
		v=v,
		wavenumber=wavenumber,
		pump_params=pump_params,
		gamma_rad_s=gamma_rad_s
	)

	L_H = build_hamiltonian_liouvillian(H)
	L_sp = build_spontaneous_decay_superoperator_ag(model_inputs, gamma_rad_s)
	L_tr, s = build_transit_superoperator_ag(model_inputs, gamma_transit_Hz)

	M = L_H + L_sp + L_tr

	rho_vec = solve(M, -s)
	rho = rho_vec.reshape((Ns, Ns), order='F')

	return rho, M, s

# =========================================================
# OPTIONAL WRAPPER
# =========================================================
def calc_chi_subdoppler_agd2_dm_placeholder(X, p_dict, return_details=False):
	return calc_chi_subdoppler_agd2_dm(X, p_dict, return_details=return_details)

# =========================================================
# PROBE HAMILTONIAN / LINEAR RESPONSE
# =========================================================

def liouvillian_action_from_hamiltonian(H, rho):
	"""
	Return -i[H, rho] as a matrix.
	"""
	return -1j * (H @ rho - rho @ H)

def solve_dm_linear_probe_response_one_velocity_ag(
	det_MHz,
	v,
	model_inputs,
	wavenumber,
	gamma_rad_s,
	pump_params,
	probe_params,
	branch,
	gamma_transit_Hz=0.0
):
	"""
	Solve:
		0 = L0(delta_rho) + L_probe(rho0)

	where rho0 is the pump-only steady state and delta_rho is the first-order
	probe response on the chosen branch.
	"""

	Ns = model_inputs['Ns']

	# Pump-only steady state
	rho0, M0, _ = solve_dm_steady_state_one_velocity_ag(
		det_MHz=det_MHz,
		v=v,
		model_inputs=model_inputs,
		wavenumber=wavenumber,
		gamma_rad_s=gamma_rad_s,
		pump_params=pump_params,
		gamma_transit_Hz=gamma_transit_Hz
	)

	# Weak probe Hamiltonian
	H_probe = build_probe_hamiltonian_ag(
		model_inputs=model_inputs,
		det_MHz=det_MHz,
		v=v,
		wavenumber=wavenumber,
		probe_params=probe_params,
		gamma_rad_s=gamma_rad_s,
		branch=branch
	)

	# Source term from probe acting on rho0
	source_mat = -liouvillian_action_from_hamiltonian(H_probe, rho0)
	source_vec = source_mat.reshape(Ns * Ns, order='F')

	# Solve linear response with trace(delta_rho)=0
	M = M0.copy()
	b = source_vec.copy()

	trace_row = np.zeros(Ns * Ns, dtype=complex)
	for i in range(Ns):
		trace_row[i * Ns + i] = 1.0

	#M[-1, :] = trace_row
	#b[-1] = 0.0

	delta_rho_vec = solve(M, b)
	delta_rho = delta_rho_vec.reshape((Ns, Ns), order='F')

	return rho0, delta_rho

# =========================================================
# SUSCEPTIBILITY EXTRACTION FROM LINEAR RESPONSE
# =========================================================
def _coherence_sum_for_branch(model_inputs, delta_rho, branch):
	Ng = model_inputs['Ng']
	e_local_map = model_inputs['e_local_map']
	transitions_by_hand = model_inputs['transitions_by_hand']

	accum = 0.0j

	for t in transitions_by_hand[branch]:
		gi = t['g_idx']
		e_abs = t['e_idx']
		if e_abs not in e_local_map:
			continue

		ej = e_local_map[e_abs]
		g = _dm_g_index(gi, Ng)
		e = _dm_e_index(ej, Ng)

		amp = t['clebsch'] / np.sqrt(3.0)
		accum += amp * delta_rho[g, e]

	return accum

def _probe_reference_rabi(gamma_rad_s, probe_params, strength_ref=1.0):
	"""
	Reference probe Rabi scale used to divide out the assumed probe amplitude
	from the linear-response solution.
	"""
	I_probe = probe_params.get('I_probe', 0.0)
	I_sat = probe_params.get('I_sat', 1.0)
	eta_probe = probe_params.get('eta_probe', 1.0)

	s0_probe = I_probe / I_sat
	return eta_probe * gamma_rad_s * np.sqrt(max(2.0 * s0_probe * strength_ref, 0.0))

def chi_branch_from_dm_scan_one_isotope(
	X,
	model_inputs,
	isotope_fraction,
	v_grid,
	f0,
	dv,
	wavenumber,
	gamma_rad_s,
	prefactor,
	pump_params,
	probe_params,
	gamma_transit_Hz,
	branch
):
	"""
	Build one branch susceptibility from the density-matrix weak-probe response
	for one isotope.
	"""
	X = np.asarray(X, dtype=float)
	chi = np.zeros(len(X), dtype=complex)

	# divide out the probe amplitude so chi does not scale with chosen I_probe
	Omega_ref = _probe_reference_rabi(gamma_rad_s, probe_params, strength_ref=1.0)
	if Omega_ref == 0.0:
		return chi

	for j, det_MHz in enumerate(tqdm(X, desc=f"{branch} scan", leave=False)):
		accum_v = 0.0j

		for k, v in enumerate(tqdm(v_grid, desc=f"{branch} velocities", leave=False)):
			_, delta_rho = solve_dm_linear_probe_response_one_velocity_ag(
				det_MHz=det_MHz,
				v=v,
				model_inputs=model_inputs,
				wavenumber=wavenumber,
				gamma_rad_s=gamma_rad_s,
				pump_params=pump_params,
				probe_params=probe_params,
				branch=branch,
				gamma_transit_Hz=gamma_transit_Hz
			)

			coh = _coherence_sum_for_branch(model_inputs, delta_rho, branch)
			accum_v += f0[k] * coh * dv

		chi[j] = isotope_fraction * prefactor * accum_v / Omega_ref

	return chi

# =========================================================
# FULL V3 CHI SCAN
# =========================================================
def calc_chi_subdoppler_agd2_dm(
	X,
	p_dict,
	return_details=False
):
	"""
	Density-matrix Ag D2 susceptibility.
	Pump defines rho0, probe is treated by linear response.
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
	AgIsotopeShift = p_dict.get('AgIsotope_shift', p_dict_defaults['AgIsotope_shift'])
	Isotope_Combination = p_dict.get('Isotope_Combination', 0)
	CustomPop = p_dict.get('CustomPop', None)
	BoltzmannFactor = p_dict.get('BoltzmannFactor', True)

	pump_params = p_dict.get('pump_params', {})
	subdop_params = p_dict.get('subdop_params', {})

	probe_params = {
		'I_probe': pump_params.get('I_probe', 0.0),
		'I_sat': pump_params.get('I_sat', 1.0),
		'eta_probe': pump_params.get('eta_probe', 1.0),
	}

	if Bfield == 0.0:
		Bfield = 1e-4

	if Constrain:
		DoppTemp_C = T_C

	T_K = T_C + 273.15
	DoppTemp_K = DoppTemp_C + 273.15

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

	active = []
	fractions = {}

	Ag109frac = 1.0 - Ag107frac

	if Isotope_Combination == 0:
		# Use both isotopes with the specified mixture
		if Ag107frac > 0.0:
			active.append('107')
			fractions['107'] = Ag107frac
		if Ag109frac > 0.0:
			active.append('109')
			fractions['109'] = Ag109frac

	elif Isotope_Combination == 1:
		# Ag107 only
		active = ['107']
		fractions['107'] = 1.0

	elif Isotope_Combination == 2:
		# Ag109 only
		active = ['109']
		fractions['109'] = 1.0

	else:
		raise ValueError("Isotope_Combination must be 0, 1, or 2")

	chi_plus = np.zeros(len(X), dtype=complex)
	chi_minus = np.zeros(len(X), dtype=complex)
	chi_z = np.zeros(len(X), dtype=complex)

	details = {
		'per_isotope': {},
		'gamma_rad_s': gamma_rad_s,
		'wavenumber': wavenumber,
		'prefactor': prefactor,
	}

	for label in tqdm(active, desc="Isotopes"):

		isotope = 'Ag107' if label == '107' else 'Ag109'

		model_inputs = build_dm_model_inputs_ag(
			isotope=isotope,
			Bfield=Bfield,
			T_K=T_K,
			AgIsotopeShift=AgIsotopeShift,
			custom_pop=CustomPop,
			BoltzmannFactor=BoltzmannFactor,
			Dline='D2'
		)

		atom_mass = ac.Ag107.mass if label == '107' else ac.Ag109.mass
		Nv = subdop_params.get('Nv', 81)
		vmax_sigma = subdop_params.get('vmax_sigma', 4.0)

		v, dv, f0, u = _build_velocity_grid(
			DoppTemp_K,
			atom_mass,
			Nv=Nv,
			vmax_sigma=vmax_sigma
		)

		gamma_transit_Hz = subdop_params.get('gamma_transit_Hz', 2.0e4)

		chiL = chi_branch_from_dm_scan_one_isotope(
			X=X,
			model_inputs=model_inputs,
			isotope_fraction=fractions[label],
			v_grid=v,
			f0=f0,
			dv=dv,
			wavenumber=wavenumber,
			gamma_rad_s=gamma_rad_s,
			prefactor=prefactor,
			pump_params=pump_params,
			probe_params=probe_params,
			gamma_transit_Hz=gamma_transit_Hz,
			branch='Left'
		)

		chiR = chi_branch_from_dm_scan_one_isotope(
			X=X,
			model_inputs=model_inputs,
			isotope_fraction=fractions[label],
			v_grid=v,
			f0=f0,
			dv=dv,
			wavenumber=wavenumber,
			gamma_rad_s=gamma_rad_s,
			prefactor=prefactor,
			pump_params=pump_params,
			probe_params=probe_params,
			gamma_transit_Hz=gamma_transit_Hz,
			branch='Right'
		)

		chiZ = chi_branch_from_dm_scan_one_isotope(
			X=X,
			model_inputs=model_inputs,
			isotope_fraction=fractions[label],
			v_grid=v,
			f0=f0,
			dv=dv,
			wavenumber=wavenumber,
			gamma_rad_s=gamma_rad_s,
			prefactor=prefactor,
			pump_params=pump_params,
			probe_params=probe_params,
			gamma_transit_Hz=gamma_transit_Hz,
			branch='Z'
		)

		chi_plus += chiL
		chi_minus += chiR
		chi_z += chiZ

		details['per_isotope'][label] = {
			'model_inputs': model_inputs,
			'v': v,
			'dv': dv,
			'f0': f0,
		}

	if return_details:
		return chi_plus, chi_minus, chi_z, details

	return chi_plus, chi_minus, chi_z

def get_spectra(X, E_in, p_dict, outputs=None):
	"""
	V3 get_spectra:
	- ordinary mode uses legacy calc_chi
	- SubDoppler mode uses V3 density-matrix susceptibility
	"""
	from libs import rotations as rot
	from libs import convert_basis as cb

	SubDoppler = p_dict.get('SubDoppler', p_dict_defaults['SubDoppler'])

	if SubDoppler:
		Elem = p_dict.get('Elem', p_dict_defaults['Elem'])
		Dline = p_dict.get('Dline', p_dict_defaults['Dline'])

		if Elem == 'Ag' and Dline == 'D2':
			ChiPlus, ChiMinus, ChiZ = calc_chi_subdoppler_agd2_dm(
				X,
				p_dict,
				return_details=False
			)
		else:
			raise ValueError(
				"SubDoppler mode is currently only implemented for Elem='Ag' and Dline='D2'"
			)
	else:
		ChiPlus, ChiMinus, ChiZ = calc_chi(X, p_dict)

	E_out, _ = get_Efield(X, E_in, [ChiPlus, ChiMinus, ChiZ], p_dict)

	X = np.asarray(X)
	E_in_arr = np.array(E_in)
	if E_in_arr.shape == (3,):
		E_in_arr = np.array([
			np.ones(len(X)) * E_in_arr[0],
			np.ones(len(X)) * E_in_arr[1],
			np.ones(len(X)) * E_in_arr[2],
		])
	elif E_in_arr.shape != (3, len(X)):
		raise ValueError("E_in must have shape (3,) or (3, len(X))")

	I_in = (E_in_arr * E_in_arr.conjugate()).sum(axis=0)

	S0 = ((E_out * E_out.conjugate()).sum(axis=0) / I_in).real

	Ex = np.array(rot.HorizPol_xy * E_out[:2])
	Ey = np.array(rot.VertPol_xy * E_out[:2])
	Ix = ((Ex * Ex.conjugate()).sum(axis=0) / I_in).real
	Iy = ((Ey * Ey.conjugate()).sum(axis=0) / I_in).real
	S1 = Ix - Iy

	E_P45 = np.array(rot.LPol_P45_xy * E_out[:2])
	E_M45 = np.array(rot.LPol_M45_xy * E_out[:2])
	I_P45 = ((E_P45 * E_P45.conjugate()).sum(axis=0) / I_in).real
	I_M45 = ((E_M45 * E_M45.conjugate()).sum(axis=0) / I_in).real
	S2 = I_P45 - I_M45

	E_out_lrz = cb.xyz_to_lrz(E_out)
	El = np.array(rot.CPol_L_lr * E_out_lrz[:2])
	Er = np.array(rot.CPol_R_lr * E_out_lrz[:2])
	Il = ((El * El.conjugate()).sum(axis=0) / I_in).real
	Ir = ((Er * Er.conjugate()).sum(axis=0) / I_in).real
	S3 = Ir - Il

	op = {
		'S0': np.array([S0]),
		'S1': np.array([S1]),
		'S2': np.array([S2]),
		'S3': np.array([S3]),
		'Ix': np.array([Ix]),
		'Iy': np.array([Iy]),
		'E_out': np.array([E_out]),
	}

	if outputs is None or 'All' in outputs:
		return op['S0'], op['S1'], op['S2'], op['S3'], op['Ix'], op['Iy']
	else:
		return [op[o] for o in outputs]
