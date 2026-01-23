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
#os.chdir(r"C:\\Users\\Matt\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")
print("Now running in:", os.getcwd())

c = 2.99792458e8

#4 and 6 didn't work
for j in range(9,18):

    if j == 4 or j == 6:
        continue

    measurement = j
    pointIndex = 20

    transmissionss = pd.read_csv("WeakProbeTransmissions{}.csv".format(measurement + 1))

    transmissions = np.array(transmissionss["Transmission"])
    transmissions_errors = np.array(transmissionss["Transmissionerr"])

    powers = ((238.1-0.179),
            (522-0.237),
            (119.8-0.231),
            (26.01-0.232),
            (1.273-0.225))

    powers2 = ((398-0.214),
            (195.8-0.193),
            (93.5-0.210),
            (25.45-0.217),
            (5.33-0.206),
            (1.429-0.199),
            (15.33-0.177),
            (1.334-0.212),
            (1.141-0.219),
            (6.72-0.216),
            (10.33-0.217),
            (16.76-0.207),
            (3.38-0.205),
            (32.11-0.214),
            (71.8-0.214),
            (371-0.199),
            (142.5-0.210))

    print(transmissions)

    newArray = []
    for i in range(len(transmissions)):
        if i >= 31:
            newArray.append((transmissions[i],transmissions_errors[i]))
    newArray = np.array(newArray)

    meanOffres = np.mean(newArray[:,0])
    meanOffres_error = np.sqrt(np.sum(newArray[:,1]**2))/len(newArray)

    print(meanOffres,meanOffres_error)

    #plt.plot(powers2,newArray)

    results = transmissions/meanOffres
    results_error = results * np.sqrt((transmissions_errors/transmissions)**2+(meanOffres_error/meanOffres)**2)

    plt.errorbar(np.arange(0,len(transmissions),1),results, yerr = results_error*10, label = "Power = {} microwatts".format(powers2[j-1]))

    result = results[pointIndex]
    result_error = results_error[pointIndex]

    print("power = {} microwatts".format(powers2[measurement-1]))
    print("transmission = {} +/- {}".format(result,result_error))

plt.xlabel("Datapoint")

plt.ylabel("Transmission")

plt.legend()

plt.show()