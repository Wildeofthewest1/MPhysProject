import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

# ----------------------------------------------------
# Matplotlib styling
# ----------------------------------------------------
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

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

c = 2.99792458e8

transmissionss = pd.read_csv("WeakProbeTransmissions.csv")

transmissions = np.array(transmissionss["Transmission"])[2:]


a = transmissions[4]
b = transmissions[5]
transmissions[4] = b
transmissions[5] = a

powers = (#(238.1-0.179),
		  (522-0.237),
		  (119.8-0.231),
		  (26.01-0.232),
		  (1.273-0.225))

newArray = []
for i in range(len(transmissions)):
    if not i%2:
        newArray.append(transmissions[i]/transmissions[i+1])
        #print("")

#print(newArray)


plt.plot(powers,newArray)

plt.xlabel("Power")

plt.ylabel("Transmission")

plt.show()

#for i in range(len(transmissions)):
	#plt.plot(np.arange(0,144,1),transmissions[i])

#plt.show()

#print(transmissions)

#print(len(transmissions))