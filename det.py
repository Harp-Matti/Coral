import os
import pathlib

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

N_samples = 1024
device = SDR(N_samples)

#apply settings
device.setSampleRate(3.2e6)
device.setBandwidth(8.0e6)
device.setFrequency(1.0e9)

eps = 1.0e-10        
        
def kurt(x):
    s = x[0,:]+1j*x[1,:]
    s = s-np.mean(s)
    cs = np.conj(s)    
    v = np.mean(s*cs)    
    s20 = np.absolute(np.mean(np.power(s,2))/(v+eps))
    s22 = np.absolute(np.mean(np.power(s,2)*np.power(cs,2))/(np.power(v,2)+eps))    
    return (s22-2-np.power(s20,2))/(1+np.power(s20,2)/2)

N_classifications = 11
#receive some samples
for i in range(N_classifications):
    if device.receive() < N_samples:
        print('Receive failed')
    freq = 1.0e9+i*10.0e6    
    device.setFrequency(freq)    
    print('Kurtosis ' + str(kurt(np.asarray(device.read()).reshape((2,N_samples)))) + ' at frequency ' + str(round(freq/1e6)) + ' MHz')

device.deactivateStream()
