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

"""

frequencies = (456777.182,
			   456777.031,
			   456776.861,
			   456776.687,
			   456776.513,
			   456776.324,
			   456776.140,
			   456775.963,
			   456775.772,
			   456775.582,
			   456775.389,
			   456775.196,
			   456774.994,
			   456774.795,
			   456774.602,
			   456774.402,
			   456774.214,
			   456774.026,
			   456773.816,
			   456773.620,
			   456773.396,
			   456773.201,
			   456772.985,
			   456772.769,
			   456772.562,
			   456772.341,
			   456772.131,
			   456771.902,
			   456771.694,
			   456771.464,
			   456771.252)

frequencies2 = (456775.401,
				456775.362,
				456775.318,
				456775.272,
				456775.223,
				456775.169,
				456775.115,
				456775.058,
				456775.006,
				456774.944,
				456774.884,
				456774.829,
				456774.773,
				456774.710,
				456774.652,
				456774.592,
				456774.534,
				456774.468,
				456774.399,
				456774.341,
				456774.286,
				456774.228,
				456774.152,
				456774.101,
				456774.044,
				456773.982,
				456773.921,
				456773.867,
				456773.809,
				456773.750,
				456773.692,
				456773.632,
				456773.565,
				456773.506,
				456773.442,
				456773.384,
				456773.317,
				456773.253,
				456773.191,
				456773.128,
				456773.064,
				456773.007,
				456772.947,
				456772.886,
				456772.810,
				456772.754,
				456772.686,
				456772.623,
				456772.559,
				456772.504,
				456772.446)



df = pd.DataFrame({
	"freq1": frequencies,
})

df.to_csv("frequencies1.csv", index=False)

df = pd.DataFrame({
	"freq2": frequencies2
})

df.to_csv("frequencies2.csv", index=False)

"""

"""
frequencies3 = (456778.966,
				456778.967,
				456778.884,
				456778.773,
				456778.638,
				456778.495,
				456778.346,
				456778.185,
				456778.025,
				456777.838,
				456777.668,
				456777.507,
				456777.308,
				456777.124,
				456776.932,
				456776.743,
				456776.544,
				456776.334,
				456776.148,
				456775.954,
				456775.864,
				456775.787,
				456775.740,
				456775.625,
				456775.569,
				456775.506,
				456775.446,
				456775.382,
				456775.319,
				456775.263,
				456775.203,
				456775.144,
				456775.079,
				456775.020,
				456774.961,
				456774.902,
				456774.835,
				456774.772,
				456774.715,
				456774.636,
				456774.578,
				456774.518,
				456774.448,
				456774.847,
				456774.828,
				456774.815,
				456774.793,
				456774.777,
				456774.757,
				456774.732,
				456774.714,
				456774.697,
				456774.684,
				456774.663,
				456774.647,
				456774.627,
				456774.610,
				456774.589,
				456774.570,
				456774.552,
				456774.531,
				456774.513,
				456774.495,
				456774.471,
				456774.452,
				456774.434,
				456774.415,
				456774.398,
				456774.378,
				456774.361,
				456774.341,
				456774.320,
				456774.301,
				456774.284,
				456774.266,
				456774.243,
				456774.233,
				456774.203,
				456774.185,
				456774.165,
				456774.143,
				456774.125,
				456774.101,
				456774.084,
				456774.061,
				456774.043,
				456774.024,
				456774.004,
				456773.980,
				456773.962,
				456773.939,
				456773.919,
				456773.902,
				456773.879,
				456773.858,
				456773.835,
				456773.815,
				456773.793,
				456773.778,
				456773.759,
				456773.734,
				456773.712,
				456773.690,
				456773.670,
				456773.648,
				456773.629,
				456773.611,
				456773.586,
				456773.565,
				456773.544,
				456773.527,
				456773.505,
				456773.485,
				456773.463,
				456773.442,
				456773.421,
				456773.399,
				456773.378,
				456773.357,
				456773.338,
				456773.280,
				456773.226,
				456773.167,
				456773.106,
				456773.046,
				456772.984,
				456772.917,
				456772.857,
				456772.664,
				456772.469,
				456772.256,
				456772.050,
				456771.792,
				456771.517,
				456771.377,
				456771.177,
				456770.972,
				456770.763,
				456770.508,
				456770.332,
				456770.134,
				456769.923,
				456769.711,
				456769.501,
				456769.286)

df = pd.DataFrame({
	"freq3": frequencies3
})

df.to_csv("frequencies3.csv", index=False)

times = (15,
		 60*1,
		 60*1+59,
		 60*2+30,
		 60*3+00,
		 60*3+30,
		 60*3+51,
		 60*4+00,
		 60*4+30,
		 60*5+00,
		 60*5+31,
		 60*6+00,
		 60*6+30,
		 60*7+00,
		 60*7+30,
		 60*8+2,
		 60*8+31,
		 60*9+00,
		 60*9+30,
		 60*10+00,
		 60*10+30,
		 60*11+7,
		 60*11+30,
		 60*12+00,
		 60*13+00,
		 60*13+35,
		 60*14+30,
		 60*15+00,
		 60*15+30,
		 60*16+00,
		 60*16+37,
		 60*17+00,
		 60*17+32,
		 60*18+00,
		 60*18+30,
		 60*19+00,
		 60*19+30,
		 60*20+00,
		 60*20+30,
		 60*22+38,
		 60*28+48,
		 60*30+00,
		 60*36+00)

frequencies4 = times#()

df = pd.DataFrame({
	"times": times,
	"freq4": frequencies4
})

df.to_csv("times.csv", index=False)
"""

"""
frequencies5 = (456779.175,
				456779.167,
				456779.163,
				456779.155,
				456779.153,
				456779.151,
				456779.135,
				456779.114,
				456779.113,
				456779.115,
				456779.108,
				456779.096,
				456779.075,
				456779.074,
				456779.068,
				456779.058,
				456779.044,
				456779.028,
				456779.014,
				456779.008,
				456779.009,
				456778.983,
				456778.978,
				456778.959,
				456778.953,
				456778.942,
				456778.933,
				456778.918,
				456778.890,
				456778.893,
				456778.880,
				456778.863,
				456778.848,
				456778.833,
				456778.817,
				456778.814,
				456778.797,
				456778.776,
				456778.762,
				456778.751,
				456778.740,
				456778.719,
				#456778.704,
				456778.702,
				456778.683,
				456778.664,
				456778.648,
				456778.631,
				456778.621,
				456778.602,
				456778.594,
				456778.570,
				456778.561,
				456778.535,
				456778.518,
				456778.516,
				456778.500,
				456778.488,
				456778.469,
				456778.451,
				456778.432,
				456778.422,
				456778.400,
				456778.379,
				456778.374,
				456778.350,
				456778.337,
				456778.315,
				456778.302,
				456778.286,
				456778.270,
				456778.252,
				456778.232,
				456778.229,
				456778.201,
				456778.184,
				456778.171,
				456778.162,
				456778.148,
				456778.125,
				456778.109,
				456778.099,
				456778.073,
				456778.053,
				456778.036,
				456778.020,
				456778.007,
				456777.990,
				456777.969,
				456777.951,
				456777.929,
				456777.913,
				456777.903,
				456777.903,
				456777.878,
				456777.886,
				456777.844,
				456777.829,
				456777.816,
				456777.796,
				456777.774,
				456777.766,
				456777.744,
				456777.719,
				456777.696,
				456777.682,
				456777.667,
				456777.646,
				456777.622,
				456777.612,
				456777.589,
				456777.574,
				456777.559,
				456777.529,
				456777.515,
				456777.499,
				456777.479,
				456777.460,
				456777.440,
				456777.416,
				456777.401,
				456777.389,
				456777.369,
				456777.341,
				456777.319,
				456777.301,
				456777.288,
				456777.279,
				456777.262,
				456777.226,
				456777.216,
				456777.199,
				456777.180,
				456777.156,
				456777.132,
				456777.109,
				456777.096,
				456777.079,
				456777.061,
				456777.034,
				456777.011,
				456776.996,
				456776.979,
				456776.956,
				456776.942,
				456776.919,
				456776.899,
				456776.872,
				456776.860,
				456776.841,
				456776.826,
				456776.800,
				456776.776,
				456776.759,
				456776.749,
				456776.718,
				456776.708,
				456776.684,
				456776.660,
				456776.646,
				456776.622,
				456776.597,
				456776.588,
				456776.570,
				456776.550,
				456776.523,
				456776.501,
				456776.488,
				456776.463,
				456776.446,
				456776.423,
				456776.403,
				456776.389,
				456776.360,
				456776.340,
				456776.321,
				456776.299,
				456776.291,
				456776.262,
				456776.239,
				456776.223,
				456776.204,
				456776.178,
				456776.164,
				456776.138,
				456776.117,
				456776.096,
				456776.073,
				456776.055,
				456776.041,
				456776.023,
				456775.988,
				456775.973,
				456775.955,
				456775.937,
				456775.907,
				456775.891,
				456775.869,
				456775.848,
				456775.836,
				456775.803,
				456775.792,
				456775.767,
				456775.744,
				456775.728,
				456775.702,
				456775.682,
				456775.660,
				456775.643,
				456775.633,
				456775.606,
				456775.577,
				456775.556,
				456775.536,
				456775.504,
				456775.491,
				456775.472,
				456775.459,
				456775.438,
				456775.411,
				456775.380,
				456775.374,
				456775.342,
				456775.316,
				456775.302,
				456775.274,
				456775.254,
				456775.239,
				456775.210,
				456775.200,
				456775.173,
				456775.141,
				456775.123,
				456775.109,
				456775.088,
				456775.069,
				456775.039,
				456775.024,
				456774.993,
				456774.977,
				456774.950,
				456774.930,
				456774.913,
				456774.891,
				456774.875,
				456774.848,
				456774.838,
				456774.809,
				456774.790,
				456774.767,
				456774.742,
				456774.717,
				456774.692,
				456774.680,
				456774.661,
				456774.633,
				456774.618,
				456774.598,
				456774.568,
				456774.546,
				456774.520,
				456774.506,
				456774.490,
				456774.471,
				456774.453,
				456774.422,
				456774.395,
				456774.378,
				456774.352,
				456774.332,
				456774.314,
				456774.290,
				456774.271,
				456774.255,
				456774.225,
				456774.216,
				456774.188,
				456774.163,
				456774.141,
				456774.116,
				456774.091,
				456774.068,
				456774.055,
				456774.028,
				456774.009,
				456773.992,
				456773.976,
				456773.944,
				456773.922,
				456773.902,
				456773.880,
				456773.858,
				456773.832,
				456773.807,
				456773.791,
				456773.773,
				456773.757,
				456773.733,
				456773.704,
				456773.673,
				456773.659,
				456773.634,
				456773.617,
				456773.592,
				456773.569,
				456773.551,
				456773.524,
				456773.516,
				456773.491,
				456773.471,
				456773.443,
				456773.419,
				456773.395,
				456773.372,
				456773.352,
				456773.331,
				456773.305,
				456773.287,
				456773.263,
				456773.246,
				456773.218,
				456773.198,
				456773.171,
				456773.148,
				456773.131,
				456773.105,
				456773.081,
				456773.066,
				456773.040,
				456773.021,
				456773.005,
				456772.975,
				456772.956,
				456772.935,
				456772.908,
				456772.891,
				456772.855,
				456772.836,
				456772.826,
				456772.796,
				456772.779,
				456772.756,
				456772.727,
				456772.719,
				456772.693,
				456772.674,
				456772.651,
				456772.622,
				456772.608,
				456772.576,
				456772.558,
				456772.534,
				456772.516,
				456772.494,
				456772.467,
				456772.452,
				456772.422,
				456772.403,
				456772.386,
				456772.363,
				456772.338,
				456772.330,
				456772.306,
				456772.276,
				456772.246,
				456772.229,
				456772.204,
				456772.184,
				456772.164,
				456772.140,
				456772.115,
				456772.102,
				456772.077,
				456772.052,
				456772.046,
				456772.016,
				456771.996,
				456771.970,
				456771.950,
				456771.920,
				456771.906,
				456771.881,
				456771.852,
				456771.834,
				456771.811,
				456771.793,
				456771.770,
				456771.747,
				456771.727,
				456771.709,
				456771.685,
				456771.668,
				456771.643,
				456771.621,
				456771.593,
				456771.577,
				456771.553,
				456771.530,
				456771.503,
				456771.483,
				456771.465,
				456771.443,
				456771.417,
				456771.405,
				456771.384,
				456771.358,
				456771.340,
				456771.313,
				456771.295,
				456771.266,
				456771.245,
				456771.230,
				456771.198,
				456771.180,
				456771.155,
				456771.132,
				456771.106,
				456771.079,
				456771.066,
				456771.051,
				456771.019,
				456771.001,
				456770.979,
				456770.961,
				456770.935,
				456770.916,
				456770.890,
				456770.873,
				456770.858,
				456770.837,
				456770.817,
				456770.797,
				456770.772,
				456770.753,
				456770.724,
				456770.701,
				456770.770,
				456770.657,
				456770.627,
				456770.611,
				456770.589,
				456770.565,
				456770.547,
				456770.520,
				456770.490,
				456770.477,
				456770.454,
				456770.437,
				456770.409,
				456770.399,
				456770.375,
				456770.351,
				456770.327,
				456770.301,
				456770.281,
				456770.260,
				456770.235,
				456770.219,
				456770.199,
				456770.182,
				456770.150,
				456770.124,
				456770.114,
				456770.096,
				456770.067,
				456770.046,
				456770.023,
				456770.009,
				456769.986,
				456769.964,
				456769.937,
				456769.913,
				456769.887,
				456769.870,
				456769.846,
				456769.821,
				456769.801,
				456769.779,
				456769.768,
				456769.741,
				456769.722,
				456769.707,
				456769.681,
				456769.661,
				456769.633,
				456769.611,
				456769.587,
				456769.566,
				456769.543,
				456769.520,
				456769.497,
				456769.482,
				456769.449,
				456769.430,
				456769.410,
				456769.390,
				456769.367,
				456769.346,
				456769.331,
				456769.311,
				456769.284,
				456769.249)

df = pd.DataFrame({
	"freq": frequencies5
})

df.to_csv("frequencies5.csv", index=False)

#"""

powers_ = np.array((1.00, 1.02, 1.02, 1.02, 1.02, 1.00, 1.02, 1.02, 1.01, 1.01, 1.02, 1.01, 1.01, 1.00, 1.01, 1.02, 1.02, 1.03, 1.02, 1.02, 1.02, 1.02, 1.02, 1.01, 1.01, 1.01, 1.02, 1.02, 1.01, 1.02, 1.02, 1.01, 1.01, 1.02, 1.01, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.01, 1.02, 1.02, 1.02, 1.04, 1.03, 1.04, 1.04, 1.04, 1.04, 1.05, 1.04, 1.04, 1.03, 1.04, 1.05, 1.05, 1.04, 1.04, 1.04, 1.05, 1.05, 1.05, 1.05, 1.05, 1.07, 1.06, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.06, 1.05, 1.06, 1.07, 1.07, 1.06, 1.07, 1.07, 1.06, 1.07, 1.07, 1.05, 1.06, 1.07, 1.06, 1.07, 1.07, 1.07, 1.06, 1.07, 1.07, 1.06, 1.07, 1.07, 1.08, 1.07, 1.07, 1.06, 1.07, 1.07, 1.07, 1.08, 1.08, 1.08, 1.08, 1.07, 1.08, 1.08, 1.07, 1.08, 1.09, 1.08, 1.07, 1.07, 1.08, 1.08, 1.08, 1.08, 1.09, 1.07, 1.08, 1.08, 1.08, 1.08, 1.09, 1.09, 1.08, 1.09, 1.08, 1.08, 1.08, 1.08, 1.07, 1.08, 1.08, 1.08, 1.09, 1.10, 1.10, 1.08, 1.10, 1.09, 1.10, 1.09, 1.10, 1.09, 1.08, 1.08, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.11, 1.10, 1.10, 1.10, 1.10, 1.11, 1.10, 1.11, 1.10, 1.1, 1.11, 1.11, 1.11, 1.10, 1.11, 1.11, 1.11, 1.11, 1.10, 1.10, 1.11, 1.11, 1.11, 1.11, 1.10, 1.11, 1.11, 1.12, 1.12, 1.11, 1.11, 1.11, 1.12, 1.10, 1.12, 1.12, 1.11, 1.12, 1.12, 1.13, 1.11, 1.12, 1.12, 1.12, 1.12, 1.12, 1.11, 1.11, 1.1, 1.13, 1.13, 1.13, 1.13, 1.12, 1.13, 1.12, 1.12, 1.13, 1.13, 1.13, 1.13, 1.12, 1.13, 1.13, 1.12, 1.11, 1.1, 1.11, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.14, 1.14, 1.15, 1.13, 1.14, 1.14, 1.13, 1.14, 1.15, 1.13, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.14, 1.13, 1.15, 1.13, 1.13, 1.13, 1.15, 1.13, 1.14, 1.14, 1.15, 1.15, 1.15, 1.13, 1.15, 1.17, 1.15, 1.14, 1.14, 1.15, 1.15, 1.14, 1.16, 1.15, 1.16, 1.15, 1.15, 1.15, 1.15, 1.16, 1.17, 1.15, 1.16, 1.14, 1.15, 1.16, 1.16, 1.16, 1.15, 1.16, 1.16, 1.15, 1.16, 1.16, 1.16, 1.15, 1.16, 1.17, 1.16, 1.16, 1.17, 1.16, 1.16, 1.18, 1.15, 1.16, 1.15, 1.16, 1.16, 1.17, 1.17, 1.17, 1.17, 1.17, 1.16, 1.16, 1.17, 1.18, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.18, 1.17, 1.17, 1.18, 1.17, 1.19, 1.18, 1.17, 1.19, 1.18, 1.18, 1.17, 1.18, 1.17, 1.17, 1.19, 1.19, 1.19, 1.19, 1.19, 1.18, 1.18, 1.19, 1.19, 1.19, 1.19, 1.20, 1.19, 1.19, 1.19, 1.19, 1.19, 1.19, 1.19, 1.18, 1.19, 1.19, 1.19, 1.20, 1.19, 1.19, 1.19, 1.19, 1.20, 1.19, 1.20, 1.19, 1.19, 1.18, 1.19, 1.20, 1.21, 1.19, 1.20, 1.21, 1.20, 1.21, 1.20, 1.19, 1.19, 1.21, 1.21, 1.20, 1.20, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.19, 1.21, 1.22, 1.21, 1.20, 1.22, 1.21, 1.19, 1.21, 1.21, 1.21, 1.19, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.20, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.20, 1.21, 1.21, 1.22, 1.21, 1.22, 1.21, 1.22, 1.21, 1.22, 1.20, 1.22, 1.21, 1.23, 1.23, 1.21, 1.22, 1.23, 1.23, 1.22, 1.23, 1.23, 1.23, 1.22, 1.24, 1.23, 1.23, 1.22, 1.23, 1.23, 1.24, 1.23, 1.23, 1.23, 1.23, 1.23, 1.23, 1.24, 1.23, 1.23, 1.23, 1.23, 1.23, 1.24, 1.24, 1.23, 1.23, 1.23, 1.23, 1.23, 1.22, 1.22, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.19, 1.19, 1.18, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.16, 1.17))

# ----------------------------------------------------
# Helper: Load Tektronix CSV (TIME, CH1, CH2)
# ----------------------------------------------------
def load_tektronix_csv(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()

	# Locate the "TIME,CH1,CH2" header row
	for i, line in enumerate(lines):
		if line.strip().startswith("TIME"):
			header_index = i
			break
	else:
		raise ValueError("Could not find TIME,CH1,CH2 header in " + filename)

	# Load numerical data
	data = np.loadtxt(filename, delimiter=",", skiprows=header_index+1)

	# Extract into arrays
	t = data[:, 0]
	ch1 = data[:, 1]
	ch2 = data[:, 2]

	return t, ch1, ch2

def effective_sample_size(x):
	x = np.asarray(x, float)
	x = x - np.mean(x)
	n = len(x)
	# autocorrelation via FFT
	f = np.fft.rfft(x, n=2*n)
	acf = np.fft.irfft(f*np.conj(f))[:n]
	acf /= acf[0]
	# integrated autocorrelation time (truncate when acf < 0)
	positive = acf[1:] > 0
	if not np.any(positive):
		return n
	m = np.argmax(~positive) + 1 if np.any(~positive) else n
	tau_int = 1 + 2*np.sum(acf[1:m])
	print("effective size = "+str(n/tau_int))
	return n / tau_int

def mean_ratio_with_bg_uncertainty(ch1_arr, ch2_arr, bg1, bg1_err, bg2, bg2_err,
	n_mc=5000, seed=0, guard_den=1e-12,
	bootstrap=False):
	"""
	T = mean( (|ch1|-bg1) / (|ch2|-bg2) )
	Uncertainty includes background mean uncertainties via Monte Carlo.
	If bootstrap=True, also includes finite-sample waveform uncertainty.
	"""
	rng = np.random.default_rng(seed)

	x = np.abs(ch1_arr).astype(float)
	y = np.abs(ch2_arr).astype(float)
	N = len(x)
	idx = np.arange(N)

	# nominal
	denom0 = y - bg2
	m0 = np.abs(denom0) > guard_den
	r0 = (x[m0] - bg1) / denom0[m0]
	T_nom = np.mean(r0)
	se_stat = np.std(r0, ddof=1) / np.sqrt(effective_sample_size(r0))

	# MC over backgrounds (+ optional bootstrap)
	bg1_s = rng.normal(bg1, bg1_err, size=n_mc)
	bg2_s = rng.normal(bg2, bg2_err, size=n_mc)

	T_samp = np.empty(n_mc, float)
	for m in range(n_mc):
		if bootstrap:
			ii = rng.choice(idx, size=N, replace=True)
			xx = x[ii]
			yy = y[ii]
		else:
			xx = x
			yy = y

		denom = yy - bg2_s[m]
		mask = np.abs(denom) > guard_den
		rr = (xx[mask] - bg1_s[m]) / denom[mask]
		T_samp[m] = np.mean(rr)
		#print("loading "+str(100*m/n_mc)+"%")

	T_hat = np.mean(T_samp)
	#se_total = np.std(T_samp, ddof=1)
	print("-----")
	se_bg = np.std(T_samp, ddof=1)
	se_tot = np.sqrt(se_stat**2 + se_bg**2)
	return T_hat, se_tot, se_stat#, se_bg

def block_mean(a, block_size):
    a = np.asarray(a, float)
    n = len(a) // block_size
    a = a[:n*block_size]
    return a.reshape(n, block_size).mean(axis=1)

def transmission_mc_fast(ch1_arr, ch2_arr, bg1, bg1_err, bg2, bg2_err,
                         block_size=500, n_mc=1000, seed=0, guard_frac=0.01):
    rng = np.random.default_rng(seed)

    x = block_mean(np.abs(ch1_arr), block_size)
    y = block_mean(np.abs(ch2_arr), block_size)

    denom0 = y - bg2
    scale = np.median(np.abs(denom0))
    guard = max(1e-12, guard_frac * scale)
    mask0 = np.abs(denom0) > guard

    r0 = (x[mask0] - bg1) / (y[mask0] - bg2)
    T = np.mean(r0)
    se_stat = np.std(r0, ddof=1) / np.sqrt(len(r0))  # blocks ~independent

    # MC over backgrounds (cheap now: len(x) ~ few hundred)
    bg1_s = rng.normal(bg1, bg1_err, size=n_mc)
    bg2_s = rng.normal(bg2, bg2_err, size=n_mc)

    # vectorised: shape (n_mc, n_blocks)
    denom = (y[mask0][None, :] - bg2_s[:, None])
    numer = (x[mask0][None, :] - bg1_s[:, None])

    # avoid rare near-zero denom in draws
    good = np.abs(denom) > guard
    ratio = np.where(good, numer / denom, np.nan)
    T_samp = np.nanmean(ratio, axis=1)

    se_bg = np.nanstd(T_samp, ddof=1)
    se_tot = np.sqrt(se_stat**2 + se_bg**2)

    return T, se_tot, se_stat, se_bg, len(r0)
import glob
#14-22
for k in range(-1,0):
	print(k)
	first = k

	if first == 0:
		folder = "SilverSpecFirst/"
	elif first == 1:
		folder = "SilverSpecSecond/"
	elif first == 2:
		folder = "VoltageTime/"
	elif first == 3:
		folder = "TEEMP/"
	elif first == 4:
		folder = "SilverSpecThird/"
	elif first == 5:
		folder = "WeakProbeFirst/"
	elif first == 6:
		folder = "SILVERRWPQ/M1/"
	elif first == 7:
		folder = "SILVERRWPQ/M2/"
	elif first == 8:
		folder = "SILVERRWPQ/M3/"
	elif first == 9:
		folder = "SILVERRWPQ/M4/"
	elif first == 10:
		folder = "SILVERRWPQ/M5/"
	elif first == 11:
		folder = "SILVERRWPQ/M6/"
	elif first == 12:
		folder = "SILVERRWPQ/M7/"
	elif first == 13:
		folder = "SILVERRWPQ/M8/"
	elif first == 14:
		folder = "SILVERWEAKPROBENEW/M1/"
	elif first == 15:
		folder = "SILVERWEAKPROBENEW/M2/"
	elif first == 16:
		folder = "SILVERWEAKPROBENEW/M3/"
	elif first == 17:
		folder = "SILVERWEAKPROBENEW/M4/"
	elif first == 18:
		folder = "SILVERWEAKPROBENEW/M5/"
	elif first == 19:
		folder = "SILVERWEAKPROBENEW/M6/"
	elif first == 20:
		folder = "SILVERWEAKPROBENEW/M7/"
	elif first == 21:
		folder = "SILVERWEAKPROBENEW/M8/"
	elif first == 22:
		folder = "SILVERWEAKPROBENEW/M9/"
	elif first == -1:
		folder = "SPEC30MICROWATT/8A/"
	elif first == -2:
		folder = "WPWM/"

	base_path = "Photodiode_Data/" + folder

	files = sorted(glob.glob(base_path + "tek*ALL.csv"))

	print("Found files:", len(files))
	#print(files)

	power = 1.2e-6  # 1.2 µW in W

	averages1 = []
	averages2 = []
	background1 = 0.0
	background2 = 0.0

	# ----------------------------------------------------
	# Load all datasets
	# ----------------------------------------------------
	bg_index = len(files)-1
	file_csvbg = base_path + "tek" + str(bg_index).zfill(4) + "ALL.csv"
	t_, ch1_arr_, ch2_arr_ = load_tektronix_csv(file_csvbg)
	
	bg1 = np.mean(np.abs(ch1_arr_))
	bg2 = np.mean(np.abs(ch2_arr_))
	bg1_error = np.std(np.abs(ch1_arr_), ddof=1) / np.sqrt(effective_sample_size(ch1_arr_))
	bg2_error = np.std(np.abs(ch2_arr_), ddof=1) / np.sqrt(effective_sample_size(ch2_arr_))

	def chunked_sem(x, n_chunks=20):
		x = np.asarray(x, float)
		chunks = np.array_split(x, n_chunks)
		means = np.array([np.mean(c) for c in chunks])
		return np.std(means, ddof=1)  # this is a drift-like scale

	#bg1_error = chunked_sem(np.abs(ch1_arr_))
	#bg2_error = chunked_sem(np.abs(ch2_arr_))


	for i in range(0, len(files)-1):
		print("file "+str(i+1)+"/"+str(len(files)-1))
		# Single Tektronix CSV containing TIME, CH1, CH2
		file_csv = base_path + "tek" + str(i).zfill(4) + "ALL.csv"

		# Load data
		t, ch1_arr, ch2_arr = load_tektronix_csv(file_csv)

		T, T_err, T_err_stat, T_err_bg, nblocks = transmission_mc_fast(
			ch1_arr, ch2_arr,
			bg1, bg1_error,
			bg2, bg2_error,
			block_size=500,
			n_mc=1000,
			seed=123+i
		)

		#if i != len(files)-1:
		averages1.append((T, T_err))
		#else: print("done")

		#ch3_arr = (np.abs(ch1_arr)-bg1)/(np.abs(ch2_arr)-bg2)

		#avg = np.mean(ch3_arr)
		#avg_error = np.std(ch3_arr)/np.sqrt(len(ch3_arr))

		# Compute averages
		#avg1 = np.mean(ch1_arr)
		#avg2 = np.mean(ch2_arr)
		
		#avg1err = np.std(ch1_arr)/np.sqrt(len(ch1_arr))
		#avg2err = np.std(ch2_arr)/np.sqrt(len(ch2_arr))

		#

		#if i != len(files)-1:
		#	averages1.append((avg, avg_error))
			#averages1.append((avg1, avg1err))
			#averages2.append((avg2, avg2err))
		#else:
			#plt.plot(t,ch3_arr)
			#plt.show()
			#background1 = ( avg, avg_error)
			#background1 = (avg1, avg1err)
			#background2 = (avg2, avg2err)
		#	print("doneeee")

	averages1_means = np.array([m for (m, e) in averages1])
	averages1_errs  = np.array([e for (m, e) in averages1])

	#averages2_means = np.array([m for (m, e) in averages2])
	#averages2_errs  = np.array([e for (m, e) in averages2])

	#background1_mean, background1_err = background1
	#background2_mean, background2_err = background2

	# ----------------------------------------------------
	# Convert to arrays and subtract background
	# ----------------------------------------------------

	if first < 5:

		import pandas as pd
		frequencies = pd.read_csv("frequencies1.csv")
		frequencies2 = pd.read_csv("frequencies2.csv")
		frequencies3 = pd.read_csv("frequencies3.csv")
		frequencies4 = pd.read_csv("times.csv")["freq4"]
		times = np.array(pd.read_csv("times.csv")["times"])/60
		frequencies5 = pd.read_csv("frequencies5.csv")

		xs1 = np.linspace(1, 4, len(files)-1)
		xs2 = np.linspace(0, len(files)-2, len(files)-1)

		print(len(frequencies5))

		if first == 0:
			xs = -np.array(frequencies)*2 + (c / (328.1629601))# - 633
		elif first == 1:
			xs = -np.array(frequencies2)*2 + (c / (328.1629601))# - 633
		elif first == 2:
			xs = xs2
		elif first == 3:
			xs = times
		elif first == 4:
			xs = -np.array(frequencies3)*2 + (c / (328.1629601))# - 633
		elif first == -1:
			xs = -np.array(frequencies5)*2 + (c / (328.1629601))
		elif first == -2:
			Pnew = ( 2.339-0.182, 2.339-0.182, 2.339-0.182,
			1.165-0.183, 1.165-0.183, 1.165-0.183,
			0.645-0.179, 0.645-0.179, 0.645-0.179,
			4.74-0.180, 4.74-0.180, 4.74-0.180,
			8.55-0.179, 8.55-0.179, 8.55-0.179,
			10.34-0.180, 10.34-0.180, 10.34-0.180,
			14.15-0.178, 14.15-0.178, 14.15-0.178,
			21.64-0.175, 21.64-0.175, 21.64-0.175,
			27.08-0.173, 27.08-0.173, 27.08-0.173,
			33.9-0.179, 33.9-0.179, 33.9-0.179 )
			xs = Pnew

	y1 = np.abs(np.abs(averages1_means))# - np.abs(background1_mean))
	#y2 = np.abs(np.abs(averages2_means) - np.abs(background2_mean))

	y1_err = np.sqrt(averages1_errs**2)# + background1_err**2)# + )
	#y2_err = np.sqrt(averages2_errs**2 + background2_err**2)# + )

	powers = ((238.1-0.179),
			(522-0.237),
			(119.8-0.231),
			(26.01-0.232),
			(1.273-0.225))

	powers1 = ((238.1-0.179),
			(238.1-0.179),
			(522-0.237),
			(522-0.237),
			(119.8-0.231),
			(119.8-0.231),
			(26.01-0.232),
			(26.01-0.232),
			(1.273-0.225),
			(1.273-0.225))

	#angle_unc = 0.165/100

	#y2_err = np.sqrt(y2_err**2 + (angle_unc * y2)**2)

	# --------------------------------------------------------
	# Add % uncertainty due to beam power fluctuations
	# --------------------------------------------------------
	power_frac = 0.00   # 0 percent

	#y1_err = np.sqrt(y1_err**2 + (power_frac * y1)**2)
	#y2_err = np.sqrt(y2_err**2 + (power_frac * y2)**2)

	#print(y1,y2)

	if first < 5:

		# ----------------------------------------------------
		# Remove region for fitting
		# ----------------------------------------------------
		#exclude = (xs1 > 2.0) & (xs1 < 3.5)
		#mask = ~exclude

		# Linear fit to CH1
		#coeffs1 = np.polyfit(xs1[mask], y1[mask], 1)
		#m1, c1 = coeffs1
		#fit_line1 = np.polyval(coeffs1, xs1)

		# ----------------------------------------------------
		# Plot original + fit
		# ----------------------------------------------------

		"""
		print(y1)

		if first == 2:
			plt.errorbar(xs2, y1, yerr = y1_err, marker="o", label="CH1 data")
			plt.xlabel("Time (Mins)")
			plt.ylabel("CH1 Signal (V)")
		else:
			plt.plot(xs1, fit_line1, '--', label=f"CH1 fit\n y = {m1:.3g}x + {c1:.3g}")
			plt.errorbar(xs1, y1, yerr = y1_err, marker="o", label="CH1 data")
			plt.errorbar(xs1, y2, yerr = y2_err, marker="o", label="CH2 data")
			plt.legend()
			plt.xlabel("Voltage (GHz)")
			plt.ylabel("Signal (V)")

		if first == 0:
			plt.savefig("Photodiode_Plot", dpi=300, bbox_inches='tight')
		elif first == 1:
			plt.savefig("Photodiode_Plot2", dpi=300, bbox_inches='tight')
		plt.show()
		"""

	# ----------------------------------------------------
	# Transmission and uncertainty propagation
	# ----------------------------------------------------
	transmission = y1# / y2

	transmission_err = y1_err
	#np.abs(transmission)*np.sqrt((y1_err/y1)**2+(y2_err/y2)**2)

	#plt.errorbar( powers1, transmission, transmission_err, marker = ".", linestyle = "", label = "wf{}, Ch1".format(i))
	#plt.plot( powers1, y2, marker = ".", linestyle = "", label = "wf{}, Ch2".format(i))

	#plt.legend()
	#plt.show()

	######## save to csv

	if first == 0:
		df = pd.DataFrame({
			"Transmission1": transmission,
			"Transmission1err": transmission_err,
		})
		df.to_csv("transmission1.csv", index=False)
	elif first == 1:
		df = pd.DataFrame({
			"Transmission2": transmission,
			"Transmission2err": transmission_err,
		})
		df.to_csv("transmission2.csv", index=False)
	elif first == 4:
		df = pd.DataFrame({
		"Transmission3": transmission,
		"Transmission3err": transmission_err,
		})
		df.to_csv("transmission3.csv", index=False)
	elif first == 5:
		df = pd.DataFrame({
		"Transmission": transmission,
		"Transmissionerr": transmission_err,
		})
		df.to_csv("WeakProbeTransmissions.csv", index=False)
	elif first >= 6:
		df = pd.DataFrame({
		"Transmission": transmission,
		"Transmissionerr": transmission_err,
		})
		df.to_csv("WeakProbeTransmissions{}.csv".format(first-4), index=False)
	elif first == -1:
		df = pd.DataFrame({
		"Transmission": transmission,
		"Transmissionerr": transmission_err,
		})
		df.to_csv("Spec30MicroWatts.csv", index=False)
	elif first == -2:
		df = pd.DataFrame({
		"Transmission": transmission,
		"Transmissionerr": transmission_err,
		"Powers": np.array(Pnew)
		})
		df.to_csv("WeakProbe2.csv", index=False)
	######## save to csv

	print("saved to csv")

	print(len(transmission))#,len(xs))

	if first == 1 or first == 4 or first == -1 or first == -2:
		#print(np.max(transmission))
		#print(np.min(transmission))
		plt.errorbar(xs, transmission, yerr=np.abs(transmission_err),fmt='.')#/np.max(transmission))
		
		#plt.ylim(0,1.1)
		#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		plt.xlabel("Detuning (GHz)")
		plt.ylabel("Transmission")
	elif first == 3:
		print(np.max(transmission))
		print(np.min(transmission))
		plt.errorbar(xs, transmission/np.max(transmission),
					yerr=np.abs(transmission_err/np.max(transmission)),
					marker='o')
		
		plt.ylim(0,1.1)
		plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		plt.xlabel("Time (Minutes)")
		plt.ylabel("Transmission")
	elif first >= 5:
		print(np.max(transmission))
		print(np.min(transmission))
		#plt.errorbar(xs, transmission/np.max(transmission),yerr=np.abs(transmission_err/np.max(transmission)),marker='o')
		
		#plt.ylim(0,1.1)
		#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		#plt.xlabel("Time (Minutes)")
		#plt.ylabel("Transmission")
	else:
		transmission = transmission/0.3301348605312241
		transmission_err = transmission_err/0.3301348605312241

		plt.errorbar(xs, transmission - np.min(transmission) + (0.08973872980696299/0.3301348605312241),
					yerr=np.abs(transmission_err),
					marker='o')
		#plt.ylim(0,1.1)
		#plt.yticks([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
		plt.xlabel("Time (Minutes)")
		plt.ylabel("On resonance Transmission")


	if first < 5:
		plt.tight_layout()

		if first == 0:
			plt.savefig("Photodiode_Transmission", dpi=300, bbox_inches='tight')
		elif first == 1:
			plt.savefig("Photodiode_Transmission2", dpi=300, bbox_inches='tight')
		elif first == 2:
			plt.savefig("TransmissionTime", dpi=300, bbox_inches='tight')

		#if first < 2:
			#plt.ylim([0, 1.1])
			#plt.xlim([-8.5,8.5])

		plt.show()