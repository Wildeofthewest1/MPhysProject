import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")

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

Transmission = pd.read_csv("WeakProbe2.csv")["Transmission"]
TransmissionError = pd.read_csv("WeakProbe2.csv")["Transmissionerr"]
Powers = pd.read_csv("WeakProbe2.csv")["Powers"]
yscale = 0.087

fig, ax = plt.subplots()

ax.errorbar(Powers,
			Transmission*yscale,
			yerr=np.abs(TransmissionError)*yscale,
			marker='.', linestyle = "")

ax.set_xlabel(r"Beam Power ($\mu$W)")
ax.set_ylabel("Transmission")

ax.axhline(3.535*yscale, color = "red", linestyle = "--", alpha = 0.5)

plt.show()