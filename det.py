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
    return np.mean(np.power(s,4))/np.mean(np.power(s,2))

N_classifications = 11
#receive some samples
for i in range(N_classifications):
    if device.receive() < N_samples:
        print('Receive failed')
    freq = 1.0e9+(i+1)*10.0e6    
    device.setFrequency(freq)    
    print('Kurtosis ' + str(kurt(np.asarray(device.read()).reshape((2,N_samples)))) + ' at frequency ' + str(round(freq/1e6)) + ' MHz')
