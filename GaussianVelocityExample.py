import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

fontsz = 16
rcParams['font.family'] = 'serif' # e.g. 'sans-serif', 'monospace', etc.
rcParams['font.serif'] = ['Times New Roman'] # specify a particular font
rcParams['font.size'] = fontsz
rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm', 'stix', 'custom'

from matplotlib.ticker import AutoMinorLocator

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['xtick.minor.size'] = 2
rcParams['ytick.minor.size'] = 2


# -----------------------------
# Parameters
# -----------------------------
N = 1_000_000
T = 300.0
kB = 1.380649e-23
m = 6.6335209e-26

sigma = np.sqrt(kB * T / m)      # broad (Doppler) width
sigma_narrow = sigma / 8         # dip width

mu = -1 * sigma                 # <-- VARIABLE: dip centre (offset) in m/s

dip_fraction = 0.7             # dip peak as a fraction of broad height at mu (0..1 is sensible)

# -----------------------------
# Helpers
# -----------------------------
def gaussian_pdf(x, mean, sig):
    return (1/(np.sqrt(2*np.pi)*sig)) * np.exp(-(x - mean)**2/(2*sig**2))

# -----------------------------
# Velocity axis
# -----------------------------
v = np.linspace(-5*sigma, 5*sigma, 2000)

# Broad Gaussian (normalised PDF)
broad = gaussian_pdf(v, 0.0, sigma)

# Broad value at the dip centre (local envelope height)
broad_mu = gaussian_pdf(mu, 0.0, sigma)
broad_0  = gaussian_pdf(0.0, 0.0, sigma)

# Ramp factor: 0 at mu=0, increases with |mu|
# (this is exactly 1 - exp(-mu^2/(2 sigma^2)))
ramp = 1.0# - (broad_mu / broad_0)

# Choose dip peak height so it "fits under" the broad Gaussian at that location
dip_peak_height = dip_fraction * ramp * broad_mu

# Build a narrow Gaussian whose peak equals dip_peak_height
narrow_pdf = gaussian_pdf(v, mu, sigma_narrow)
narrow_peak = gaussian_pdf(mu, mu, sigma_narrow)  # peak of narrow_pdf
dip = (dip_peak_height / narrow_peak) * narrow_pdf

# Final profile
profile_with_moving_dip = broad - dip

# Convert to counts per bin
num_bins = 120
bin_width = (v.max() - v.min()) / num_bins
counts_no_dip = N * broad * bin_width
counts_with_dip = N * profile_with_moving_dip * bin_width

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(v, counts_no_dip, linestyle = "--", color = "red")
plt.plot(v, counts_with_dip, color = "black")

plt.xlabel("Velocity component v")
plt.ylabel("Number of atoms (arb.)")
plt.grid(False)

plt.ylim(0, 50000)
plt.xlim(-1200,1200)

# Remove y-axis numbers
plt.yticks([])

# Keep only zero on x-axis
plt.xticks([0], ['0'])

# Remove tick marks
plt.tick_params(axis='x', length=0)
plt.tick_params(axis='y', length=0)

# Get top of plot
y_top = plt.ylim()[1]

# Interpolate Gaussian value at ±mu (using counts_no_dip for envelope)
y_mu_pos = np.interp(mu, v, counts_with_dip)
y_mu_neg = np.interp(-mu, v, counts_with_dip)

# Draw vertical lines only above Gaussian
plt.vlines(mu, y_mu_pos, y_top, color="red", label="Pump Beam at -v")
plt.vlines(-mu, y_mu_neg, y_top, color="blue", label="Probe Beam at +v")

plt.fill_between(v, counts_with_dip, color = "grey", alpha = 1)

plt.legend()
plt.tight_layout()
plt.show()