import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.fft import *
from scipy.signal import *
import scipy.io.wavfile as sw

'''
x = np.array([1, 2, -2, 8, -4, 3, 5, -2, 1])
print(x)
fc = fft(x)
print(fc)
ifc = ifft(fc)
print(ifc)
print(np.abs(ifc))
print(np.real(ifc))

x = np.random.randn(1000)
fc = fft(x)
plt.figure()
plt.plot(np.abs(fc))
plt.show()
'''
'''

fs = 1000
F = 200
t = np.linspace(0, 1, fs)
x = np.sin(2*np.pi*F*t)
fc = fft(x)
plt.figure()
plt.plot(np.abs(fc[:len(fc)//2])) #[:len(fc)//2]
plt.show()

x2 = np.sin(2*np.pi*2*F*t)
x = (x + x2) / 2
fc = fft(x)
plt.figure()
plt.plot(np.abs(fc[:len(fc)//2]))
plt.show()
'''

'''
N = 1000
h = hann(N)
plt.figure()
plt.subplot(2,1,1)
plt.plot(h)

fch = fft(h)
plt.subplot(2,1,2)
plt.plot(np.abs(fch))
plt.show()'''

def index_of_max(arr):
    m,x=-1,-1
    for i in range(len(arr)):
        if arr[i]>m:
            m,x=arr[i],i
    return x

with open('Speech/voice.wav',"rb") as f:
    Fs,s=sw.read(f) #28700 56400

silence_begin=10500
voice_begin=40800

time_length=int(Fs/1000*20) #мс 10500 40800
silence=s[silence_begin:silence_begin+time_length]
voice=s[voice_begin:voice_begin+time_length]
plt.figure()
plt.subplot(2,2,1)
plt.plot(silence)
plt.subplot(2,2,2)
plt.plot(voice)
plt.subplot(2,2,3)
fft_silence=np.abs(fft(silence*hann(len(silence))))
fft_silence=fft_silence[:len(fft_silence)//2]
plt.plot(fft_silence)
plt.subplot(2,2,4)
fft_voice=np.abs(fft(voice*hann(len(voice))))
fft_voice=fft_voice[:len(fft_voice)//2]
print("Voice F0=",index_of_max(fft_voice)*Fs/(time_length-1))
plt.plot(fft_voice)
plt.show()
