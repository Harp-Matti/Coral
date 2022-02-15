import os
import time
import pathlib

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

N_samples = 4096
device = SDR(N_samples)

#apply settings
f0 = 1.0e9
device.setSampleRate(6e6)
device.setBandwidth(8.0e6)
device.setFrequency(f0)
freq = f0

eps = 1.0e-10        
        
def my_sleep(duration, get_now=time.perf_counter):
    # More precise version of time.sleep()
    now = get_now()
    end = now + duration
    while now < end:
        now = get_now()        
        
def kurt(x):    
    s = x[0,:]+1j*x[1,:] 
    s = s-np.mean(s)
    cs = np.conj(s)    
    v = np.absolute(np.mean(s*cs))    
    s20 = np.absolute(np.mean(np.power(s,2))/(v+eps))
    s22 = np.absolute(np.mean(np.power(s,2)*np.power(cs,2))/(np.power(v,2)+eps))    
    return (s22-2.0-np.power(s20,2))/(1.0+np.power(s20,2)/2)

N_classifications = 11
#receive some samples
for i in range(N_classifications):
    if device.receive() < N_samples:
        print('Receive failed')
    print('Kurtosis ' + str(kurt(np.nan_to_num(np.asarray(device.read()).reshape((2,N_samples))))) + ' at frequency ' + str(round(freq/1e6)) + ' MHz')    
    freq = f0+i*10.0e6    
    device.setFrequency(freq)
    #my_sleep(0.1)    

device.deactivateStream()
