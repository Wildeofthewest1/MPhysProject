import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

from libs import main_functions_V3 as mf
from libs import Hamiltonian as ht
from joblib import Parallel, delayed

BFIELD = 1e-4
T_C = 130.23
T_K = T_C + 273.15
DOPP_TEMP_K = T_K

AG107_SHIFT = 229.24
AG109_SHIFT = -246.76
AG_SHIFT = (AG107_SHIFT, AG109_SHIFT)

pump_params = {
    'pol': 'Left',
    'probe_pol': 'Left',
    'eta_pump': 1.0,
    'eta_probe': 1.0,
    'I_probe': 13.2,
    'I_sat': 867.0,
    'I_pump': 2000.0,
}

gamma_transit_Hz = 2.0e4
gamma_rep_Hz = 1.0e5
gamma_vcc_Hz = 5.0e4

target_pop = np.array([0.45, 0.55/3, 0.55/3, 0.55/3], dtype=float)

gamma_rad_s = 2.0 * np.pi * mf.ac.AgD2Transition.NatGamma * 1.0e6
wavenumber = mf.ac.AgD2Transition.wavevectorMagnitude

det_scan = np.linspace(-4000, 4000, 401)
xGHz = det_scan / 1e3

# velocity grid settings
Nv = 81
vmax_sigma = 4.0

# parallel settings
n_jobs = 6          # adjust for your CPU
max_iter = 15       # faster than 50 for testing
tol = 1e-7          # looser than 1e-9 for testing

isotopes = ['Ag107', 'Ag109']
results = {}

check_det_MHz = 0.0
check_idx = np.argmin(np.abs(det_scan - check_det_MHz))


def solve_one_detuning_with_vcc(det, model_inputs, v_grid, f0, dv,
                                wavenumber, gamma_rad_s, pump_params,
                                gamma_transit_Hz, L_sp, L_tr, L_rep, s,
                                gamma_vcc_Hz, max_iter, tol):
    """
    Solve one detuning using the iterative VCC solver, then return
    velocity-integrated populations plus the full rho_list for optional inspection.
    """
    rho_list = mf.solve_dm_steady_state_all_velocities_with_vcc_ag(
        det_MHz=det,
        v_grid=v_grid,
        f0=f0,
        dv=dv,
        model_inputs=model_inputs,
        wavenumber=wavenumber,
        gamma_rad_s=gamma_rad_s,
        pump_params=pump_params,
        gamma_transit_Hz=gamma_transit_Hz,
        L_sp=L_sp,
        L_tr=L_tr,
        L_rep=L_rep,
        s=s,
        gamma_vcc_Hz=gamma_vcc_Hz,
        mix_excited=False,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    Ng = model_inputs['Ng']
    Ne = model_inputs['Ne']

    rho_gg_int = np.zeros(Ng)
    rho_ee_int = np.zeros(Ne)
    trace_val = 0.0

    for k, rho in enumerate(rho_list):
        diag = np.real(np.diag(rho))
        rho_gg_int += diag[:Ng] * f0[k] * dv
        rho_ee_int += diag[Ng:Ng + Ne] * f0[k] * dv
        trace_val += np.trace(rho).real * f0[k] * dv

    return {
        'det': det,
        'rho_gg_int': rho_gg_int,
        'rho_ee_int': rho_ee_int,
        'trace': trace_val,
        'rho_list': rho_list,
    }


for isotope in isotopes:
    print(f"\n=== Starting isotope {isotope} ===")

    model_inputs = mf.build_dm_model_inputs_ag(
        isotope=isotope,
        Bfield=BFIELD,
        T_K=T_K,
        AgIsotopeShift=AG_SHIFT,
        custom_pop=None,
        BoltzmannFactor=True,
        Dline='D2'
    )

    atom_mass = mf.ac.Ag107.mass if isotope == 'Ag107' else mf.ac.Ag109.mass
    v_grid, dv, f0, u = mf._build_velocity_grid(
        DOPP_TEMP_K,
        atom_mass,
        Nv=Nv,
        vmax_sigma=vmax_sigma
    )

    L_sp, L_tr, L_rep, s = mf.build_static_superoperators_ag(
        model_inputs=model_inputs,
        gamma_rad_s=gamma_rad_s,
        gamma_transit_Hz=gamma_transit_Hz,
        gamma_rep_Hz=gamma_rep_Hz,
        target_pop=target_pop,
    )

    Ng = model_inputs['Ng']
    Ne = model_inputs['Ne']

    rho_gg_int = np.zeros((len(det_scan), Ng))
    rho_ee_int = np.zeros((len(det_scan), Ne))
    trace_arr = np.zeros(len(det_scan))

    rho_gg_vmap = None
    rho_ee_vmap = None

    print(f"{isotope}: launching {len(det_scan)} detuning jobs with n_jobs={n_jobs}")
    parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(solve_one_detuning_with_vcc)(
            det,
            model_inputs,
            v_grid,
            f0,
            dv,
            wavenumber,
            gamma_rad_s,
            pump_params,
            gamma_transit_Hz,
            L_sp,
            L_tr,
            L_rep,
            s,
            gamma_vcc_Hz,
            max_iter,
            tol,
        )
        for det in det_scan
    )

    print(f"{isotope}: unpacking results")

    for j, out in enumerate(parallel_results):
        rho_gg_int[j, :] = out['rho_gg_int']
        rho_ee_int[j, :] = out['rho_ee_int']
        trace_arr[j] = out['trace']

        if j == check_idx:
            rho_gg_vmap = np.array([np.real(np.diag(rho)[:Ng]) for rho in out['rho_list']])
            rho_ee_vmap = np.array([np.real(np.diag(rho)[Ng:Ng + Ne]) for rho in out['rho_list']])

    results[isotope] = {
        'model_inputs': model_inputs,
        'v_grid': v_grid,
        'dv': dv,
        'f0': f0,
        'rho_gg_int': rho_gg_int,
        'rho_ee_int': rho_ee_int,
        'rho_gg_vmap': rho_gg_vmap,
        'rho_ee_vmap': rho_ee_vmap,
        'trace_arr': trace_arr,
    }

    print(f"{isotope}: done")


# -------------------------------------------------
# Ground populations: velocity-integrated, both isotopes
# -------------------------------------------------
plt.figure(figsize=(10, 6))

for gi in range(results['Ag107']['model_inputs']['Ng']):
    plt.plot(xGHz, results['Ag107']['rho_gg_int'][:, gi], label=f'Ag107 g{gi}')

for gi in range(results['Ag109']['model_inputs']['Ng']):
    plt.plot(xGHz, results['Ag109']['rho_gg_int'][:, gi], '--', label=f'Ag109 g{gi}')

plt.plot(xGHz, results['Ag107']['rho_gg_int'].sum(axis=1), 'k-', lw=2.5, label='Ag107 sum ground')
plt.plot(xGHz, results['Ag109']['rho_gg_int'].sum(axis=1), 'k--', lw=2.5, label='Ag109 sum ground')

plt.xlabel('Detuning (GHz)')
plt.ylabel('Velocity-integrated population')
plt.title('Ground-state populations with VCC')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Excited populations: velocity-integrated, both isotopes
# -------------------------------------------------
plt.figure(figsize=(10, 6))

for ej in range(results['Ag107']['model_inputs']['Ne']):
    plt.plot(xGHz, results['Ag107']['rho_ee_int'][:, ej], label=f'Ag107 e{ej}')

for ej in range(results['Ag109']['model_inputs']['Ne']):
    plt.plot(xGHz, results['Ag109']['rho_ee_int'][:, ej], '--', label=f'Ag109 e{ej}')

plt.plot(xGHz, results['Ag107']['rho_ee_int'].sum(axis=1), 'k-', lw=2.5, label='Ag107 sum excited')
plt.plot(xGHz, results['Ag109']['rho_ee_int'].sum(axis=1), 'k--', lw=2.5, label='Ag109 sum excited')

plt.xlabel('Detuning (GHz)')
plt.ylabel('Velocity-integrated population')
plt.title('Excited-state populations with VCC')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Trace check
# -------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(xGHz, results['Ag107']['trace_arr'], label='Ag107 trace')
plt.plot(xGHz, results['Ag109']['trace_arr'], '--', label='Ag109 trace')
plt.xlabel('Detuning (GHz)')
plt.ylabel('Velocity-integrated Tr(rho)')
plt.title('Trace check')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Velocity-resolved ground populations at one detuning
# -------------------------------------------------
check_det_MHz = det_scan[check_idx]
v107 = results['Ag107']['v_grid']
v109 = results['Ag109']['v_grid']

plt.figure(figsize=(10, 6))
for gi in range(results['Ag107']['model_inputs']['Ng']):
    plt.plot(v107, results['Ag107']['rho_gg_vmap'][:, gi], label=f'Ag107 g{gi}')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Population')
plt.title(f'Ag107 ground-state populations vs velocity at detuning = {check_det_MHz:.1f} MHz')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for gi in range(results['Ag109']['model_inputs']['Ng']):
    plt.plot(v109, results['Ag109']['rho_gg_vmap'][:, gi], label=f'Ag109 g{gi}')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Population')
plt.title(f'Ag109 ground-state populations vs velocity at detuning = {check_det_MHz:.1f} MHz')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Print populations at one detuning for quick check
# -------------------------------------------------
print(f"\nVelocity-integrated ground populations at detuning ~ {check_det_MHz:.1f} MHz:")
for isotope in isotopes:
    print(f"\n{isotope}:")
    for gi, val in enumerate(results[isotope]['rho_gg_int'][check_idx]):
        print(f"g{gi}: {val:.6f}")
    print(f"sum ground: {results[isotope]['rho_gg_int'][check_idx].sum():.6f}")
    print(f"sum excited: {results[isotope]['rho_ee_int'][check_idx].sum():.6e}")
    print(f"trace: {results[isotope]['trace_arr'][check_idx]:.6f}")

# -------------------------------------------------
# Ground energies for checking state ordering
# -------------------------------------------------
ES = ht.Hamiltonian('Ag107', 'D2', 1.0, 1e-4, (229.24, -246.76))

print("\nGround energies (MHz):")
for i, st in enumerate(ES.groundManifold):
    print(i, st[0].real)