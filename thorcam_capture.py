import os
import sys
from pathlib import Path

import numpy as np
import tifffile

from PIL import Image

CAMERA_EXPOSURE_US = 100008   # 100.008 ms
CAMERA_GAIN_DB = 27.3
CAMERA_BLACK_LEVEL = 0
CAMERA_BINX = 1
CAMERA_BINY = 1
CAMERA_ROI = (0, 0, 1439, 1079)

PNG_MIN_INTENSITY = 321
PNG_MAX_INTENSITY = 658

ENABLE_FRAME_RATE_CONTROL = False
CAMERA_FRAME_RATE_FPS = 34.815

# -------------------------------------------------
# Adjust these paths to match your installation
# -------------------------------------------------
SDK_PACKAGE_PARENT = Path(
	r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
	r"\Scientific Camera Interfaces\SDK\Python Toolkit\thorlabs_tsi_sdk-0.0.8"
)

SDK_DLL_DIR_64 = Path(
	r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
	r"\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib"
)

SDK_DLL_DIR_32 = Path(
	r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
	r"\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\32_lib"
)

def save_preview_png(image_u16: np.ndarray, png_path: Path, min_intensity=PNG_MIN_INTENSITY, max_intensity=PNG_MAX_INTENSITY):
    clipped = np.clip(image_u16, min_intensity, max_intensity)
    scaled = (clipped - min_intensity) / (max_intensity - min_intensity)
    scaled = np.clip(255 * scaled, 0, 255).astype(np.uint8)
    Image.fromarray(scaled).save(str(png_path))

def configure_thorlabs_paths():
	is_64bits = sys.maxsize > 2**32
	dll_dir = SDK_DLL_DIR_64 if is_64bits else SDK_DLL_DIR_32

	if not SDK_PACKAGE_PARENT.exists():
		raise FileNotFoundError(f"Thorlabs SDK package folder not found: {SDK_PACKAGE_PARENT}")
	if not dll_dir.exists():
		raise FileNotFoundError(f"Thorlabs DLL folder not found: {dll_dir}")

	if str(SDK_PACKAGE_PARENT) not in sys.path:
		sys.path.insert(0, str(SDK_PACKAGE_PARENT))

	os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
	try:
		os.add_dll_directory(str(dll_dir))
	except AttributeError:
		pass


def capture_one_image(save_path: Path, exposure_time_us: int = CAMERA_EXPOSURE_US, poll_timeout_ms: int = 2000):
	configure_thorlabs_paths()

	from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	with TLCameraSDK() as sdk:
		cameras = sdk.discover_available_cameras()
		if len(cameras) == 0:
			raise RuntimeError("No Thorlabs cameras detected")

		with sdk.open_camera(cameras[0]) as camera:
			print(f"Opened camera: {cameras[0]}")
			print("Model:", camera.model)
			print("Serial:", camera.serial_number)

			# Match ThorCam acquisition settings
			camera.exposure_time_us = exposure_time_us
			camera.binx = CAMERA_BINX
			camera.biny = CAMERA_BINY
			camera.roi = CAMERA_ROI
			camera.black_level = CAMERA_BLACK_LEVEL

			# Gain in the SDK is usually set by index, so convert from dB
			if camera.gain_range.max > 0:
				gain_index = camera.convert_decibels_to_gain(CAMERA_GAIN_DB)
				camera.gain = gain_index
				print(f"Requested gain: {CAMERA_GAIN_DB} dB")
				print(f"Applied gain: {camera.convert_gain_to_decibels(camera.gain)} dB")
			else:
				print("Gain not supported on this camera.")

			# Frame-rate control, if supported
			if ENABLE_FRAME_RATE_CONTROL:
				try:
					camera.is_frame_rate_control_enabled = True
					camera.frame_rate_control_value = CAMERA_FRAME_RATE_FPS
					print(f"Frame-rate control set to: {camera.frame_rate_control_value} FPS")
				except Exception as e:
					print(f"Could not set frame-rate control: {e}")

			camera.image_poll_timeout_ms = poll_timeout_ms

			# For one saved image, arm and take one software-triggered frame
			camera.frames_per_trigger_zero_for_unlimited = 1

			print("Final settings:")
			print("  exposure_time_us =", camera.exposure_time_us)
			print("  gain =", camera.gain)
			print("  black_level =", camera.black_level)
			print("  binx =", camera.binx)
			print("  biny =", camera.biny)
			print("  roi =", camera.roi)

			camera.arm(2)
			try:
				camera.issue_software_trigger()

				frame = camera.get_pending_frame_or_null()
				if frame is None:
					raise TimeoutError("Timeout reached while waiting for ThorCam frame")

				image_buffer_copy = np.copy(frame.image_buffer)

				print("Image stats:")
				print("  min =", int(image_buffer_copy.min()))
				print("  max =", int(image_buffer_copy.max()))
				print("  mean =", float(image_buffer_copy.mean()))

				# Save raw TIFF
				tifffile.imwrite(str(save_path), image_buffer_copy)

				# Save viewable PNG preview
				png_path = save_path.with_suffix(".png")
				save_preview_png(image_buffer_copy, png_path, min_intensity=321, max_intensity=658)

				print(f"Saved TIFF: {save_path}")
				print(f"Saved PNG:  {png_path}")
				print(f"Frame #{frame.frame_count}, shape={image_buffer_copy.shape}, dtype={image_buffer_copy.dtype}")

			finally:
				camera.disarm()


def main():
	if len(sys.argv) < 2:
		print("Usage: py thorcam_capture.py <output_path> [exposure_time_us]")
		sys.exit(1)

	save_path = Path(sys.argv[1])
	exposure_time_us = int(sys.argv[2]) if len(sys.argv) >= 3 else CAMERA_EXPOSURE_US

	capture_one_image(save_path, exposure_time_us=exposure_time_us)


if __name__ == "__main__":
	main()