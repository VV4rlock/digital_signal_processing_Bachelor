import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.fft import *
from scipy.signal import *

#x = np.array([1, 2, -2, 8, -4, 3, 5, -2, 1])
#print(x)
#fc = fft(x)
#print(fc)
#ifc = ifft(fc)
#print(ifc)
#print(np.abs(ifc))
#print(np.real(ifc))

#x = np.random.randn(1000)
#fc = fft(x)
#plt.figure()
#plt.plot(np.abs(fc))
#plt.show()


#fs = 1000
#F = 200
#t = np.linspace(0, 1, fs)
#x = np.sin(2*np.pi*F*t)
#fc = fft(x)
#plt.figure()
#plt.plot(np.abs(fc[:len(fc)//2]))
#plt.show()

#x2 = np.sin(2*np.pi*2*F*t)
#x = (x + x2) / 2
#fc = fft(x)
#plt.figure()
#plt.plot(np.abs(fc[:len(fc)//2]))
#plt.show()

N = 1000
h = hann(N)
plt.figure()
plt.plot(h)
plt.show()

fch = fft(h)
plt.figure()
plt.plot(np.abs(fch))
plt.show()
