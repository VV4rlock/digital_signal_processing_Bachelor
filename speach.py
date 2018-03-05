import scipy.io.wavfile as sw
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

#fr,s=sw.read('Speech/voice.wav')
'''
with open('Speech/voice.wav',"rb") as f:
    fr,s=sw.read(f)

print(fr)
print(s)
print(s.dtype)
print(max(s),np.max(abs(s)))
s2=s*3
sw.write("Speech/out.wav",fr//3,s2)
plt.figure()
plt.plot(s2)
plt.show()

r=np.random.randn(2000)
b=np.cumsum(r)
#l[n]=x[n]-y[n]
'''
F=24000
fs=16000
#t=np.linspace(0,5,5*fs)
#t=np.random.randn(100000)
#print(t)
#S=np.sin(2*np.pi*F*t)
S=np.random.randn(2000)
#S=np.random.uniform(-1,1,100000)
S=np.cumsum(S)
S=np.float128(S/max(abs(S)))#(-1;1)

def func(S,B):
    print(B, "bit:")
    M = 2 ** (B - 1) - 1
    Y = np.around(S * M) / M
    E = S - Y
    print("\tsn teor: ", 6 * B - 7.2, "sn prac: ", 10 * sp.log10(np.var(S) / np.var(E)))
    return Y

plt.figure()
plt.subplot(3,1,1)
plt.plot(S)
plt.plot(func(S,8))
plt.subplot(3,1,2)
plt.plot(S)
plt.plot(func(S,16))
plt.subplot(3,1,3)
plt.plot(S)
plt.plot(func(S,32))
plt.show()


#sw.write("A.wav",fs,S)
