import os
import time
import csv
from pathlib import Path
from datetime import datetime
import ctypes
import math

import numpy as np
import pyvisa


# -----------------------------
# User settings
# -----------------------------
REPO_DIR = Path(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
TYPEFOLDERNAME = "TestRun"

SCOPE_ADDR = "USB0::0x0699::0x0421::C020493::INSTR"
AFG_ADDR   = "USB0::0x0699::0x0343::C023586::INSTR"

WAVEMETER_PRESENT = True  # set False if not connected / not needed

offset_start = 0.0
offset_stop  = 5.0
offset_step  = 1  # 10 mV (set to 1.0 if you want quick testing)

measurement_delay = 2   # seconds

# -----------------------------
# Folder setup (always under REPO_DIR)
# -----------------------------
data_dir = REPO_DIR / "Photodiode_Data" / TYPEFOLDERNAME
data_dir.mkdir(parents=True, exist_ok=True)

run_folder = data_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder.mkdir(parents=True, exist_ok=False)

print("Saving data to:", run_folder)


# -----------------------------
# VISA setup
# -----------------------------
rm = pyvisa.ResourceManager("@ivi")

scope = rm.open_resource(SCOPE_ADDR)
afg   = rm.open_resource(AFG_ADDR)

# More forgiving comms settings
scope.timeout = 30000  # 30 s (ASCII waveform transfer can be slow/large)
afg.timeout   = 5000

# Increase chunk size so large reads don't stall
scope.chunk_size = 1024 * 1024  # 1 MB

scope.read_termination = "\n"
scope.write_termination = "\n"

# Clear any old I/O state
try:
	scope.clear()
except Exception:
	pass

import math

wavemeter = None

if WAVEMETER_PRESENT:

    wavemeter_path = r"C:\Program Files (x86)\Bristol Wavelength Meter V2_31b\CLDevIFace.dll"

    try:
        wavemeter = ctypes.CDLL(wavemeter_path)
        print("Bristol wavemeter DLL loaded")
    except Exception as e:
        print("Wavemeter DLL not found, disabling wavemeter:", e)
        WAVEMETER_PRESENT = False

def get_wavemeter_frequency_hz():

	if not WAVEMETER_PRESENT:
		return math.nan

	try:
		freq = ctypes.c_double()

		# Typical Bristol DLL call
		wavemeter.CLGetLambdaReading(ctypes.byref(freq))

		return freq.value * 1e12  # convert THz → Hz if needed

	except Exception as e:
		print("Wavemeter read failed:", e)
		return math.nan

def safe_wavemeter_frequency_hz() -> float:
	if not WAVEMETER_PRESENT:
		return math.nan
	try:
		return float(get_wavemeter_frequency_hz())
	except Exception as e:
		print(f"[WAVEMETER] read failed: {e}")
		return math.nan

def tek_name(index: int) -> str:
	return f"tek{index:04d}ALL.csv"

def get_waveform(channel: int):
	"""
	Download a waveform from the scope using ASCII encoding (robust).
	Returns (t, volts).
	"""
	scope.write(f"DATA:SOU CH{channel}")
	scope.write("DATA:WIDTH 1")
	scope.write("DATA:ENC ASCii")

	# Scaling (Tektronix WFMPRE)
	ymult = float(scope.query("WFMPRE:YMULT?"))
	yzero = float(scope.query("WFMPRE:YZERO?"))
	yoff  = float(scope.query("WFMPRE:YOFF?"))
	xincr = float(scope.query("WFMPRE:XINCR?"))
	xzero = float(scope.query("WFMPRE:XZERO?"))

	# Stop acquisition for consistent data
	scope.write("ACQ:STATE STOP")
	raw = scope.query("CURVE?").strip()
	scope.write("ACQ:STATE RUN")

	# Parse ASCII points
	y = np.fromstring(raw, sep=",", dtype=float)

	volts = (y - yoff) * ymult + yzero
	t = xzero + np.arange(volts.size) * xincr
	return t, volts


# -----------------------------
# Sweep loop with progress / ETA
# -----------------------------
# -----------------------------
# Sweep loop with Tek naming + progress
# -----------------------------
num_measurements = int(round((offset_stop - offset_start) / offset_step)) + 1
print(f"Total measurements: {num_measurements}")

start_time = time.time()

for idx in range(num_measurements):
	offset = offset_start + idx * offset_step
	loop_start = time.time()

	file_index = idx
	filename = run_folder / tek_name(file_index)

	print(f"\nMeasurement {idx+1}/{num_measurements} | offset = {offset:.3f} V -> {filename.name}")

	# Set AFG offset
	afg.write(f"SOUR1:VOLT:OFFS {offset}")

	time.sleep(measurement_delay)

	# Capture both channels
	t1, ch1 = get_waveform(1)
	t2, ch2 = get_waveform(2)

	n = min(len(t1), len(t2), len(ch1), len(ch2))
	t = t1[:n]
	ch1 = ch1[:n]
	ch2 = ch2[:n]

	# Save CSV
	freq_hz = safe_wavemeter_frequency_hz()
	print(f"Wavemeter frequency: {freq_hz:.6e} Hz")

	with open(filename, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["time_s", "CH1_V", "CH2_V", "wavemeter_freq_Hz"])
		# repeat the scalar frequency on each row (easy for downstream code)
		w.writerows((ti, v1, v2, freq_hz) for ti, v1, v2 in zip(t, ch1, ch2))

	print(f"Saved: {filename.name}")

	time.sleep(measurement_delay)

	loop_time = time.time() - loop_start
	elapsed = time.time() - start_time
	remaining = (num_measurements - (idx + 1)) * loop_time
	print(f"Time/measurement: {loop_time:.2f} s | Elapsed: {elapsed/60:.2f} min | ETA: {remaining/60:.2f} min")

print("\nSweep complete.")

# -----------------------------
# Background capture (same naming convention)
# -----------------------------
input("Press ENTER to take background measurement...")

bg_index = num_measurements
bg_file = run_folder / tek_name(bg_index)
print(f"Taking background -> {bg_file.name}")

time.sleep(measurement_delay)  # optional: settle before background

t1, ch1 = get_waveform(1)
t2, ch2 = get_waveform(2)

n = min(len(t1), len(t2), len(ch1), len(ch2))
t = t1[:n]
ch1 = ch1[:n]
ch2 = ch2[:n]

freq_hz = safe_wavemeter_frequency_hz()

with open(bg_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "CH1_V", "CH2_V", "wavemeter_freq_Hz"])
    w.writerows((ti, v1, v2, freq_hz) for ti, v1, v2 in zip(t, ch1, ch2))

print(f"Background saved: {bg_file.name}")
print("\nMeasurement run complete.")