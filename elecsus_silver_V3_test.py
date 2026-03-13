import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
print("Now running in:", os.getcwd())

from libs import main_functions_V3 as mf

BFIELD = 1e-4
T_C = 130.23
T_K = T_C + 273.15

AG107_SHIFT = 229.24
AG109_SHIFT = -246.76
AG_SHIFT = (AG107_SHIFT, AG109_SHIFT)

pump_params = {
    'pol': 'Left',
    'probe_pol': 'Left',
    'eta_pump': 0.0039,
    'eta_probe': 1,
    'I_pump': 2030.0,
    'I_probe': 13.2,
    'I_sat': 867.0,
}

gamma_rad_s = 2.0 * np.pi * mf.ac.AgD2Transition.NatGamma * 1.0e6
wavenumber = mf.ac.AgD2Transition.wavevectorMagnitude

det_scan = np.linspace(-4000, 4000, 201)
xGHz = det_scan / 1e3

isotopes = ['Ag107', 'Ag109']
results = {}

for isotope in isotopes:
    model_inputs = mf.build_dm_model_inputs_ag(
        isotope=isotope,
        Bfield=BFIELD,
        T_K=T_K,
        AgIsotopeShift=AG_SHIFT,
        custom_pop=None,
        BoltzmannFactor=True,
        Dline='D2'
    )

    Ng = model_inputs['Ng']
    Ne = model_inputs['Ne']

    rho_gg = np.zeros((len(det_scan), Ng))
    rho_ee = np.zeros((len(det_scan), Ne))
    trace_arr = np.zeros(len(det_scan))

    for j, det in enumerate(det_scan):
        rho, M, b = mf.solve_dm_steady_state_one_velocity_ag(
            det_MHz=det,
            v=0.0,
            model_inputs=model_inputs,
            wavenumber=wavenumber,
            gamma_rad_s=gamma_rad_s,
            pump_params=pump_params,
            gamma_transit_Hz=2.0e4
        )

        for gi in range(Ng):
            rho_gg[j, gi] = rho[gi, gi].real
        for ej in range(Ne):
            rho_ee[j, ej] = rho[Ng + ej, Ng + ej].real

        trace_arr[j] = np.trace(rho).real

    results[isotope] = {
        'model_inputs': model_inputs,
        'rho_gg': rho_gg,
        'rho_ee': rho_ee,
        'trace_arr': trace_arr,
    }

# -------------------------------------------------
# Ground populations: both isotopes on same axes
# -------------------------------------------------
plt.figure(figsize=(10, 6))

for gi in range(results['Ag107']['model_inputs']['Ng']):
    plt.plot(xGHz, results['Ag107']['rho_gg'][:, gi], label=f'Ag107 g{gi}')

for gi in range(results['Ag109']['model_inputs']['Ng']):
    plt.plot(xGHz, results['Ag109']['rho_gg'][:, gi], '--', label=f'Ag109 g{gi}')

plt.plot(xGHz, results['Ag107']['rho_gg'].sum(axis=1), 'k-', lw=2.5, label='Ag107 sum ground')
plt.plot(xGHz, results['Ag109']['rho_gg'].sum(axis=1), 'k--', lw=2.5, label='Ag109 sum ground')

plt.xlabel('Detuning (GHz)')
plt.ylabel('Population')
plt.title('Ground-state populations, both isotopes, v = 0')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Excited populations: both isotopes on same axes
# -------------------------------------------------
plt.figure(figsize=(10, 6))

for ej in range(results['Ag107']['model_inputs']['Ne']):
    plt.plot(xGHz, results['Ag107']['rho_ee'][:, ej], label=f'Ag107 e{ej}')

for ej in range(results['Ag109']['model_inputs']['Ne']):
    plt.plot(xGHz, results['Ag109']['rho_ee'][:, ej], '--', label=f'Ag109 e{ej}')

plt.plot(xGHz, results['Ag107']['rho_ee'].sum(axis=1), 'k-', lw=2.5, label='Ag107 sum excited')
plt.plot(xGHz, results['Ag109']['rho_ee'].sum(axis=1), 'k--', lw=2.5, label='Ag109 sum excited')

plt.xlabel('Detuning (GHz)')
plt.ylabel('Population')
plt.title('Excited-state populations, both isotopes, v = 0')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()