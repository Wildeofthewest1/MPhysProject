import numpy as np

numbers = np.array([481.941,650.535,657.487,538.737,690.102,696.423,(2 * 0.1**2)+456.577])

D2Line = 328.1629601
c = 2.99792458e8

centroid = c/D2Line

def ConvertToFrequency(array):

    zero = 456.577
    ref1 = 546.495
    ref2 = 591.300
    arrayerr = 0.1

    dist = ref2 - ref1
    disterr = np.sqrt(2 * 0.1**2)

    scale = 500/dist
    scaleerr = 500/disterr

    absolutearr = array - zero
    absolutearrerr = np.sqrt(2*arrayerr**2)
    
    arraynew = scale * absolutearr
    arraynewerr = arraynew * np.sqrt((scaleerr/scale)**2+(absolutearrerr/absolutearr)**2)

    return np.array(arraynew),np.array(arraynewerr)

splittings = ConvertToFrequency(numbers)[0]

freqError = ConvertToFrequency(numbers)[1]
print(freqError)

#1.115946880928723 #MHz

F01Shift = splittings[3] - splittings[0]
F11Shift = splittings[4] - splittings[1]
F012hift = splittings[5] - splittings[2]

print(F012hift,F11Shift,F012hift)

shiftError = 2 * freqError**2

