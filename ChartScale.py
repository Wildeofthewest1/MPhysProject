import numpy as np

numbers = np.array((481.941,650.535,657.487,538.737,690.102,696.437))

D2Line = 328.1629601
c = 2.99792458e8

centroid = c/D2Line

def ConvertToFrequency(array):

    zero = 456.577
    ref1 = 546.495

    arrayerr = 0.1

    dist = ref1 - zero
    disterr = np.sqrt(2 * arrayerr**2)

    scale = 1000/dist
    scaleerr = (1000/dist**2)*disterr
    
    absolutearr = array - zero
    absolutearrerr = np.sqrt((2 * (arrayerr**2)))

    arraynew = scale * absolutearr
    arraynewerr = arraynew * np.sqrt((scaleerr/scale)**2+(absolutearrerr/absolutearr)**2)

    return np.array(arraynew),np.array(arraynewerr)

splittings = ConvertToFrequency(numbers)[0]
freqError = ConvertToFrequency(numbers)[1]

#1.115946880928723 #MHz

F01Shift = splittings[3] - splittings[0]
F11Shift = splittings[4] - splittings[1]
F12Shift = splittings[5] - splittings[2]

F01ShiftErr = np.sqrt(freqError[3]**2 + freqError[0]**2)
F11ShiftErr = np.sqrt(freqError[4]**2 + freqError[1]**2)
F12ShiftErr = np.sqrt(freqError[5]**2 + freqError[2]**2)

print(
    f"{F01Shift:.2f} +/- {F01ShiftErr:.2f} MHz",
    f"{F11Shift:.2f} +/- {F11ShiftErr:.2f} MHz",
    f"{F12Shift:.2f} +/- {F12ShiftErr:.2f} MHz",
    sep="\n"
)

# collect isotope shifts and errors
shifts = np.array([F01Shift, F11Shift, F12Shift])
shiftErrs = np.array([F01ShiftErr, F11ShiftErr, F12ShiftErr])

# unweighted mean and its uncertainty
meanShift = np.mean(shifts)
meanShiftErr = np.sqrt(np.sum(shiftErrs**2)) / len(shifts)

print(f"Average isotope shift = {meanShift:.2f} +/- {meanShiftErr:.2f} MHz")
