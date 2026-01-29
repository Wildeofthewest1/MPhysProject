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

r1 = []
r2 = []
r4 = []

for j in range(9,18):

    if j == 4 or j == 6:
        continue

    measurement = j
    pointIndex = 20

    transmissionss = pd.read_csv("WeakProbeTransmissions{}.csv".format(measurement + 1))

    transmissions = np.array(transmissionss["Transmission"])
    transmissions_errors = np.array(transmissionss["Transmissionerr"])#*10

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
            (142.5-0.210),
            (30-0.2))

    #print(transmissions)

    newArray = []
    for i in range(len(transmissions)):
        if i >= 31:
            newArray.append((transmissions[i],transmissions_errors[i]))
    newArray = np.array(newArray)

    meanOffres = np.mean(newArray[:,0])
    meanOffres_error = np.sqrt(np.sum(newArray[:,1]**2))/len(newArray)

    #print(meanOffres,meanOffres_error)

    #plt.plot(powers2,newArray)

    results = transmissions/meanOffres
    results_error = results * np.sqrt((transmissions_errors/transmissions)**2+(meanOffres_error/meanOffres)**2)

    #plt.errorbar( Voltage, results, yerr = results_error*10, label = "Power = {} microwatts".format(powers2[j-1]))

    result = results[pointIndex]
    result_error = results_error[pointIndex]
    power = powers2[measurement-1]

    #print("power = {} microwatts".format(power))
    #print("transmission = {} +/- {}".format(result,result_error))

    r1.append(results)
    r2.append(results_error)
    r4.append(power)
    #print(j)

#r1 = np.array(r1)
#r2 = np.array(r2)
r3 = np.arange(0,len(transmissions),1)
#r4 = np.array(r4)

for k in range(0,9):
    print(len(r1[k]))
    if len(r1[k]) != 42:
           print(k)
           continue
    plt.errorbar( r3, r1[k], yerr = r2[k], label = "Power = {} microwatts".format(r4[k]))

plt.xlabel("Datapoint")

plt.ylabel("Transmission")

plt.legend()

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def fit_ymin_index(y, yerr=None, window=7):
    """
    Fit y(i) with a parabola locally around the minimum (i = index),
    return ymin and its uncertainty from covariance.
    """
    y = np.asarray(y, float)
    n = len(y)
    i = np.arange(n, dtype=float)

    if yerr is None:
        yerr = np.ones_like(y, float)
    else:
        yerr = np.asarray(yerr, float)

    i0 = int(np.nanargmin(y))
    lo = max(0, i0 - window)
    hi = min(n, i0 + window + 1)

    xfit = i[lo:hi]
    yfit = y[lo:hi]
    sfit = yerr[lo:hi]

    popt, pcov = curve_fit(
        parabola, xfit, yfit,
        sigma=sfit, absolute_sigma=True,
        p0=[1e-3, 0.0, np.min(yfit)],
        maxfev=10000
    )
    a, b, c = popt

    # Vertex location (index) and minimum value
    xmin = -b/(2*a)
    ymin = parabola(xmin, a, b, c)

    # Uncertainty on ymin = c - b^2/(4a)
    dya = (b*b) / (4*a*a)
    dyb = -b / (2*a)
    dyc = 1.0
    J = np.array([dya, dyb, dyc])
    ymin_var = J @ pcov @ J
    ymin_err = np.sqrt(ymin_var) if ymin_var > 0 else np.nan

    return ymin, ymin_err, xmin, popt, pcov

rows = []
for k in range(len(r1)):
    y = np.array(r1[k])
    yerr = np.array(r2[k])
    power = r4[k]

    if len(y) < 5:
        continue

    try:
        ymin, ymin_err, xmin_idx, popt, pcov = fit_ymin_index(y, yerr=yerr, window=7)
    except Exception as e:
        print(f"Fit failed for k={k}, power={power}: {e}")
        continue

    rows.append({
        "k": k,
        "power_uW": power,
        "ymin": ymin,
        "ymin_err": ymin_err,
        "xmin_index": xmin_idx,
    })

df_min = pd.DataFrame(rows).sort_values("power_uW").reset_index(drop=True)
print(df_min[["power_uW", "ymin", "ymin_err", "xmin_index"]])

# Plot minimum transmission vs power
plt.figure()
rem = 0
plt.errorbar(df_min["power_uW"][rem:], df_min["ymin"][rem:], yerr=df_min["ymin_err"][rem:], fmt='.', capsize=3)
plt.xlabel("Power (µW)")
plt.ylabel("Minimum transmission (from parabola fit)")
plt.title("Minimum transmission vs power")
plt.xscale("log")
plt.show()