# bristol_read.py (run with 32-bit python)
import ctypes
from ctypes import c_int, c_double, POINTER, byref
import os
import math
import time

DLL_PATH = r"C:\Program Files (x86)\Bristol Wavelength Meter V2_31b\CLDevIFace.dll"
COM_PORT = 3

# From header: enum { LAMBDA_UNIT_NM, LAMBDA_UNIT_GH, LAMBDA_UNIT_CM};
LAMBDA_UNIT_NM = 0

READ_RETRIES = 5
RETRY_DELAY_S = 0.05

def main():
    if not os.path.exists(DLL_PATH):
        print("nan")
        return

    dll = ctypes.CDLL(DLL_PATH)  # __cdecl

    dll.CLOpenUSBSerialDevice.argtypes = [c_int]
    dll.CLOpenUSBSerialDevice.restype  = c_int

    if hasattr(dll, "CLCloseDevice"):
        dll.CLCloseDevice.argtypes = [c_int]
        dll.CLCloseDevice.restype  = c_int

    has_lambda1 = hasattr(dll, "CLGetLambdaReading")
    if has_lambda1:
        # double __cdecl CLGetLambdaReading(int DevHandle);
        dll.CLGetLambdaReading.argtypes = [c_int]
        dll.CLGetLambdaReading.restype  = c_double

    has_lambda2 = hasattr(dll, "CLGetLambdaReading2")
    if has_lambda2:
        # int __cdecl CLGetLambdaReading2(int DevHandle, double *data);
        dll.CLGetLambdaReading2.argtypes = [c_int, POINTER(c_double)]
        dll.CLGetLambdaReading2.restype  = c_int

    # Optional but nice: force units to nm if available
    if hasattr(dll, "CLSetLambdaUnits"):
        dll.CLSetLambdaUnits.argtypes = [c_int, c_int]
        dll.CLSetLambdaUnits.restype  = c_int

    handle = dll.CLOpenUSBSerialDevice(COM_PORT)
    if handle < 0:
        print("nan")
        return

    try:
        if hasattr(dll, "CLSetLambdaUnits"):
            try:
                dll.CLSetLambdaUnits(handle, LAMBDA_UNIT_NM)
            except Exception:
                pass

        lam_nm = math.nan

        for _ in range(READ_RETRIES):
            # Try return-double API first
            if has_lambda1:
                try:
                    v = float(dll.CLGetLambdaReading(handle))
                    if v > 0.0 and math.isfinite(v):
                        lam_nm = v
                        break
                except Exception:
                    pass

            # Fallback pointer API
            if has_lambda2:
                try:
                    lam = c_double(0.0)
                    r = int(dll.CLGetLambdaReading2(handle, byref(lam)))
                    if r == 0 and lam.value > 0.0 and math.isfinite(lam.value):
                        lam_nm = float(lam.value)
                        break
                except Exception:
                    pass

            time.sleep(RETRY_DELAY_S)

        if lam_nm > 0.0:
            print(f"{lam_nm:.12f}")
        else:
            print("nan")

    finally:
        if hasattr(dll, "CLCloseDevice"):
            try:
                dll.CLCloseDevice(handle)
            except Exception:
                pass

if __name__ == "__main__":
    main()