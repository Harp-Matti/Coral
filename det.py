import os
import time
import pathlib

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

import sys, getopt

N_samples = 1024
f0 = 1.0e9
bw = 8.0e6
rate = 6e6
step = 10e6
N_steps = 11

try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:s:r:b:N:n:")
except getopt.GetoptError:
        print('det.py -f <frequency> -s <frequency_step> -b <bandwidth> -r <sample_rate> -N <n_samples> -n <n_steps>')
        sys.exit(2)
for opt, arg in opts:
        if opt == '-h':
                print('det.py -f <frequency> -s <frequency_step> -b <bandwidth> -r <sample_rate> -N <n_samples> -n <n_steps>')
                sys.exit()
        elif opt == "-f":
                f0 = float(arg)
        elif opt == "-s":
                step = float(arg)
        elif opt == "-b":
                bw = float(arg)
        elif opt == "-r":
                rate = float(arg)
        elif opt == "-N":
                N_samples = int(arg)
        elif opt == "-n":
                N_steps = int(arg)        

device = SDR(N_samples)
                
#apply settings
device.setSampleRate(rate)
device.setBandwidth(bw)
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
    #s = x[0,:]+1j*x[1,:] 
    #s = s-np.mean(s)
    #cs = np.conj(s)    
    #v = np.absolute(np.mean(s*cs))    
    #s20 = np.absolute(np.mean(np.power(s,2))/(v+eps))
    #s22 = np.absolute(np.mean(np.power(s,2)*np.power(cs,2))/(np.power(v,2)+eps))    
    #return (s22-2.0-np.power(s20,2))/(1.0+np.power(s20,2)/2)
    s = x[0,:]
    s -= np.mean(s)
    return np.mean(np.power(s,4))/np.power(np.mean(np.power(s,2)),2)

def stft(x):
        NFFT = 128
        overlap = 96
        s = x[0,:]+1j*x[1,:]
        S = np.zeros((int(np.floor((len(s)-NFFT)/(NFFT-overlap))),NFFT))
        for i in range(S.shape[0]):
                si = i*(NFFT-overlap)
                S[i,:] = s[si:si+NFFT]
        S = np.fft.fft(S)
        m = np.median(S)
        for i in range(S.shape[0]):
                st = ""
                for j in range(NFFT):
                        if S[i,j] > 3*m/2:
                                st += 'O'
                        else if S[i,j] > m:
                                st += 'o'
                        else if S[i,j] > m/2:
                                st += '.'
                        else:
                                st += ' '
                print(st)                
        

#receive some samples
for i in range(N_steps):
    start = timer()
    if device.receive() < N_samples:
        print('Receive failed')
    my_sleep(N_samples/rate-(timer()-start))
    print('Kurtosis ' + str(kurt(np.nan_to_num(np.asarray(device.read()).reshape((2,N_samples))))) + ' at frequency ' + str(round(freq/1e6)) + ' MHz')    
    stft(np.nan_to_num(np.asarray(device.read()).reshape((2,N_samples))))
    freq += step    
    device.setFrequency(freq)
    #my_sleep(0.1)    

device.deactivateStream()
