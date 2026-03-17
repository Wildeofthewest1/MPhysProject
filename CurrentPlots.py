import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------
# Plot styling
# -----------------------------
fontsz = 16
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True

# -----------------------------
# Data from table
# -----------------------------
current = np.array([4, 6, 7, 8], dtype=float)

AgNumberDensity = np.array([1.0e15, 4.23e15, 9.69e15, 1.442e16], dtype=float)
AgNumberDensity_err = np.array([3e13, 3e13, 3e13, 4e13], dtype=float)

Temperature = np.array([90, 132, 147, 155], dtype=float)
Temperature_err = np.array([20, 6, 3, 3], dtype=float)

F0_population = np.array([0.483, 0.428, 0.4278, 0.4332], dtype=float)
F0_population_err = np.array([0.013, 0.004, 0.0018, 0.0014], dtype=float)

# -----------------------------
# Number density plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(
    current, AgNumberDensity,
    yerr=AgNumberDensity_err,
    fmt='o', capsize=3
)
ax.set_xlabel("Current (mA)")
ax.set_ylabel(r"Ag Number Density (atoms m$^{-3}$)")
ax.minorticks_on()
fig.tight_layout()
plt.show()

# -----------------------------
# Temperature plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(
    current, Temperature,
    yerr=Temperature_err,
    fmt='o', capsize=3
)
ax.set_xlabel("Current (mA)")
ax.set_ylabel(r"Temperature ($^{\circ}$C)")
ax.minorticks_on()
fig.tight_layout()
plt.show()

# -----------------------------
# F=0 population plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(
    current, F0_population,
    yerr=F0_population_err,
    fmt='o', capsize=3
)
ax.set_xlabel("Current (mA)")
ax.set_ylabel(r"$F=0$ Population")
ax.minorticks_on()
fig.tight_layout()
plt.show()