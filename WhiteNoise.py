import numpy as np
import matplotlib.pyplot as plt
'''
N=10000000
#noise=np.random.randn(N)
noise=np.random.rand(N)
plt.figure()

#~~~~~~~~~~
#plt.plot(noise)
#print("mean:",np.mean(noise))
#print("var:",np.var(noise))
#plt.show()
#~~~~~~~~~~~~~~~
#plt.hist(noise,1000)
#plt.show()
#~~~~~~~~~~~~~~~~~

h,bins=np.histogram(noise,100)
#plt.plot(bins[:-1],h)
plt.bar(bins[:-1],h)
plt.show()
'''

N=10000
t=np.linspace(0,1,N)
F1=2#frequent
F2=10
S1=np.sin(2*np.pi*F1*t)
S2=np.sin(2*np.pi*F2*t)
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,S1,'-r',label='Freq=2')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Freq=2")
plt.subplot(2,3,2)
plt.plot(t,S2,'-g',label='Freq=10')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Freq=10")
plt.legend()

plt.show()
