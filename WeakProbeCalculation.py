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
#os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
os.chdir(r"C:\\Users\\Matt\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

c = 2.99792458e8

transmissionss = pd.read_csv("WeakProbeTransmissions2.csv")

transmissions = np.array(transmissionss["Transmission"])

powers = ((238.1-0.179),
		  (522-0.237),
		  (119.8-0.231),
		  (26.01-0.232),
		  (1.273-0.225))

powers2 = ((1.036-0.184),
           (14.10-0.196),
           (63.4-0.184),
           (167.8-0.183),
           (355-0.181),
           (453-0.179))

print(transmissions)

#newArray = []
#for i in range(len(transmissions)):
   # if not i%2:
    #    newArray.append(transmissions[i]/transmissions[i+1])
        #print("")

#print(newArray)

#plt.plot(powers2,newArray)

plt.plot(np.arange(0,len(transmissions),1),transmissions)

plt.xlabel("Power")

plt.ylabel("Transmission")

plt.show()