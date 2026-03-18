import os
import time
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
import pyvisa
import subprocess
import math

# -----------------------------
# User settings
# -----------------------------
REPO_DIR = Path(r"C:\Users\Alienware\OneDrive - Durham University\Level_4_Project\Lvl_4\Repo")
TYPEFOLDERNAME = "SubDoppler_8mA_4_Lamp_Flipped"

SCOPE_ADDR = "USB0::0x0699::0x0421::C020493::INSTR"
AFG_ADDR   = "USB0::0x0699::0x0343::C023586::INSTR"

measurement_delay = 4  # seconds

# --- Channel recording switches ---
RECORD_CH3 = False   # Set to False to save only CH1 and CH2

# --- Desired scope scales ---
TIME_PER_DIV_S = 0.1      # 100 ms/div
CH1_SCALE_VDIV = 0.02     # 20 mV/div
CH2_SCALE_VDIV = 0.02     # 20 mV/div
CH3_SCALE_VDIV = 0.02     # 20 mV/div

# --- Bristol wavemeter (use 32-bit helper only) ---
WAVEMETER_PRESENT = True
BRISTOL_HELPER = REPO_DIR / "bristol_read.py"
BRISTOL_HELPER_PY = ["py", "-3.14-32"]
BRISTOL_TIMEOUT_S = 2.0

def read_bristol_lambda_nm_via_32bit(timeout_s: float = BRISTOL_TIMEOUT_S) -> float:
	"""
	Calls bristol_read.py using 32-bit Python and returns wavelength in nm.
	bristol_read.py MUST print only a single number (nm) or 'nan'.
	"""
	try:
		out = subprocess.check_output(
			BRISTOL_HELPER_PY + [str(BRISTOL_HELPER)],
			text=True,
			timeout=timeout_s
		).strip()

		if not out or out.lower() == "nan":
			return math.nan
		return float(out)
	except Exception:
		return math.nan


def read_bristol_freq_hz_string() -> str:
	lam_nm = read_bristol_lambda_nm_via_32bit()
	if not (lam_nm > 0.0) or math.isnan(lam_nm):
		return "nan"
	c = 299_792_458.0
	freq = c / (lam_nm * 1e-9)
	return f"{freq:.6f}"


# -----------------------------
# Tek helpers
# -----------------------------
def tek_name(index: int) -> str:
	return f"tek{index:04d}ALL.csv"


def build_offsets() -> np.ndarray:
	seg1 = np.round(np.arange(0.00, 1.80 + 1e-12, 0.10), 2)
	seg2 = np.round(np.arange(1.81, 3.80 + 1e-12, 0.01), 2)
	seg3 = np.round(np.arange(3.90, 5.00 + 1e-12, 0.10), 2)

	offsets = np.concatenate([seg1, seg2, seg3])
	return offsets


def configure_scope(scope):
	"""
	Set scope horizontal + vertical scales.
	"""
	scope.write(f"HOR:MAIN:SCA {TIME_PER_DIV_S}")

	scope.write(f"CH1:SCA {CH1_SCALE_VDIV}")
	scope.write(f"CH2:SCA {CH2_SCALE_VDIV}")

	scope.write("SEL:CH1 ON")
	scope.write("SEL:CH2 ON")

	if RECORD_CH3:
		scope.write(f"CH3:SCA {CH3_SCALE_VDIV}")
		scope.write("SEL:CH3 ON")
	else:
		scope.write("SEL:CH3 OFF")


# -----------------------------
# Main
# -----------------------------
def main():
	data_dir = REPO_DIR / "Photodiode_Data" / TYPEFOLDERNAME
	data_dir.mkdir(parents=True, exist_ok=True)

	run_folder = data_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
	run_folder.mkdir(parents=True, exist_ok=False)
	print("Saving data to:", run_folder)

	rm = pyvisa.ResourceManager("@ivi")
	scope = rm.open_resource(SCOPE_ADDR)
	afg   = rm.open_resource(AFG_ADDR)

	scope.timeout = 30000
	afg.timeout   = 5000

	scope.chunk_size = 1024 * 1024
	scope.read_termination = "\n"
	scope.write_termination = "\n"

	try:
		scope.clear()
	except Exception:
		pass

	configure_scope(scope)

	def get_waveform(channel: int):
		"""
		Export the full current waveform record using the scope's existing settings.
		"""
		scope.write(f"DATA:SOU CH{channel}")
		scope.write("DATA:WIDTH 1")
		scope.write("DATA:ENC ASCii")

		rec_len = int(float(scope.query("HOR:RECO?")))
		scope.write("DATA:STAR 1")
		scope.write(f"DATA:STOP {rec_len}")

		n_send = int(float(scope.query("WFMOUTPRE:NR_PT?")))

		ymult = float(scope.query("WFMOUTPRE:YMULT?"))
		yzero = float(scope.query("WFMOUTPRE:YZERO?"))
		yoff  = float(scope.query("WFMOUTPRE:YOFF?"))
		xincr = float(scope.query("WFMOUTPRE:XINCR?"))
		xzero = float(scope.query("WFMOUTPRE:XZERO?"))

		raw = scope.query("CURVE?").strip()
		y = np.fromstring(raw, sep=",", dtype=float)

		print(f"CH{channel}: HOR:RECO={rec_len}, WFMOUTPRE:NR_PT={n_send}, received={y.size}")

		volts = (y - yoff) * ymult + yzero
		t = xzero + np.arange(volts.size) * xincr
		return t, volts

	try:
		if WAVEMETER_PRESENT:
			test_freq = read_bristol_freq_hz_string()
			print(f"Wavemeter test (Hz): {test_freq}")

		offsets = build_offsets()
		num_measurements = len(offsets)
		print(f"Total measurements: {num_measurements}")

		start_time = time.time()

		for idx, offset in enumerate(offsets):
			loop_start = time.time()

			filename = run_folder / tek_name(idx)
			print(f"\nMeasurement {idx+1}/{num_measurements} | offset = {offset:.3f} V -> {filename.name}")

			afg.write(f"SOUR1:VOLT:OFFS {offset}")
			time.sleep(measurement_delay)

			t1, ch1 = get_waveform(1)
			t2, ch2 = get_waveform(2)

			if RECORD_CH3:
				t3, ch3 = get_waveform(3)
				n = min(len(t1), len(t2), len(t3), len(ch1), len(ch2), len(ch3))
				t = t1[:n]
				ch1 = ch1[:n]
				ch2 = ch2[:n]
				ch3 = ch3[:n]
			else:
				n = min(len(t1), len(t2), len(ch1), len(ch2))
				t = t1[:n]
				ch1 = ch1[:n]
				ch2 = ch2[:n]

			freq_str = read_bristol_freq_hz_string()
			print(f"Wavemeter frequency (Hz): {freq_str}")

			with open(filename, "w", newline="") as f:
				w = csv.writer(f)

				if RECORD_CH3:
					w.writerow(["TIME", "CH1", "CH2", "CH3", "FREQ"])
					w.writerows((ti, v1, v2, v3, freq_str) for ti, v1, v2, v3 in zip(t, ch1, ch2, ch3))
				else:
					w.writerow(["TIME", "CH1", "CH2", "FREQ"])
					w.writerows((ti, v1, v2, freq_str) for ti, v1, v2 in zip(t, ch1, ch2))

			print(f"Saved: {filename.name}")

			loop_time = time.time() - loop_start
			elapsed = time.time() - start_time
			remaining = (num_measurements - (idx + 1)) * loop_time
			print(f"Time/measurement: {loop_time:.2f} s | Elapsed: {elapsed/60:.2f} min | ETA: {remaining/60:.2f} min")

		print("\nSweep complete.")

		input("Press ENTER to take background measurement...")

		bg_index = num_measurements
		bg_file = run_folder / tek_name(bg_index)
		print(f"Taking background -> {bg_file.name}")

		time.sleep(measurement_delay)

		t1, ch1 = get_waveform(1)
		t2, ch2 = get_waveform(2)

		if RECORD_CH3:
			t3, ch3 = get_waveform(3)
			n = min(len(t1), len(t2), len(t3), len(ch1), len(ch2), len(ch3))
			t = t1[:n]
			ch1 = ch1[:n]
			ch2 = ch2[:n]
			ch3 = ch3[:n]
		else:
			n = min(len(t1), len(t2), len(ch1), len(ch2))
			t = t1[:n]
			ch1 = ch1[:n]
			ch2 = ch2[:n]

		freq_str = read_bristol_freq_hz_string()
		print(f"Wavemeter frequency (Hz): {freq_str}")

		with open(bg_file, "w", newline="") as f:
			w = csv.writer(f)

			if RECORD_CH3:
				w.writerow(["TIME", "CH1", "CH2", "CH3", "FREQ"])
				w.writerows((ti, v1, v2, v3, freq_str) for ti, v1, v2, v3 in zip(t, ch1, ch2, ch3))
			else:
				w.writerow(["TIME", "CH1", "CH2", "FREQ"])
				w.writerows((ti, v1, v2, freq_str) for ti, v1, v2 in zip(t, ch1, ch2))

		print(f"Background saved: {bg_file.name}")
		print("\nMeasurement run complete.")

	finally:
		try:
			scope.close()
		except Exception:
			pass
		try:
			afg.close()
		except Exception:
			pass
		try:
			rm.close()
		except Exception:
			pass


if __name__ == "__main__":
	main()