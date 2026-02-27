

import pandas as pd
import os
import numpy as np

os.chdir(r"C:\\Users\\Alienware\\OneDrive - Durham University\\Level_4_Project\\Lvl_4\\Repo")

array = np.array(pd.read_csv("baseline_corrected_curr6.csv")["Transmission_BaselineCorrected"])

min_index = np.argmin(array)

print(min_index)

print(1.8 + 0.01*(94 - 18))