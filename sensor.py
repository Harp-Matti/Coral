import os
import pathlib
import platform

machine = platform.uname().machine
if machine == 'aarch64':
    from coralclassifier import Classifier
elif machine == 'r': # TODO: find machine for raspberry
    from tfclassifier import Classifier
else: 
    error('Unknown machine')

import socket
import sys
from time import sleep
import random
from struct import pack

from sdr.sdr import SDR
import numpy as np

from timeit import default_timer as timer

from comms.comms import *

eps = 1.0e-10

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_hfradio_resnet_quant_edgetpu.tflite')

def normalize(x):
    x -= np.mean(x,1,keepdims=True)
    x /= np.sqrt(np.mean(np.sum(np.power(x,2))))+eps
    return x

def pwelch(x,n):
    psd = np.zeros((n))
    w = np.hanning(n)
    N = int(np.floor((x.size/2-n)/(n/2))+1)
    for i in range(N):
        psd = psd + abs(np.fft.fft((x[0,0,int(i*n/2):int(i*n/2+n),0] + 1j*x[0,1,int(i*n/2):int(i*n/2+n),0])* w))**2/N
    #ma = np.amax(psd)
    #psd = psd/ma
    return psd.tolist()

class Sensor:
    def __init__(self, comms):
        self.comms = comms
        
        self.N_samples = 2*1024
        self.device = SDR(self.N_samples)
        
        #apply settings
        self.device.setSampleRate(3.2e6)
        self.device.setBandwidth(8.0e6)
        self.device.setFrequency(1.0e9)

        self.classifier = Classifier(model_file)
        
    def run(self):
        if self.device.receive() < self.N_samples:
            print('Receive failed')
            self.comms.send(Failure())
        else:
            x = normalize(np.asarray(self.device.read()).reshape((2,self.N_samples))).reshape(1,2,self.N_samples,1)
            class_result = self.classifier.run(x)
            spectrum = pwelch(x,128)
            self.comms.send(Result(class_result,spectrum))
    
    def get_parameter(self,parameter):
        if parameter == 'frequency':
            return self.device.getFrequency()
        elif parameter == 'bandwidth':
            return self.device.getBandwidth()
        elif parameter == 'sample_rate':
            return self.device.getSampleRate()
        else:
            error('Unknown parameter for method get_parameter')
    
    def set_parameter(self,parameter,value):
        if parameter == 'frequency':
            self.device.setFrequency(value)
        elif parameter == 'bandwidth':
            self.device.setBandwidth(value)
        elif parameter == 'sample_rate':
            self.device.setSampleRate(value)
        else:
            error('Unknown parameter for method set_parameter')
            
        return self.get_parameter(parameter) == value 
        
    def wait(self):
        while True:
            message = self.comms.receive()
            message_type = type(message)
            if message_type == Run:
                self.run()
            elif message_type == Get:
                value = self.get_parameter(message.parameter)
                self.comms.send(Return(message.parameter,value))
            elif message_type == Set:
                success = self.set_parameter(message.parameter,message.value)
                if success:
                    self.comms.send(Success())
                else:
                    self.comms.send(Failure())
            elif message_type == Exit:
                exit()
            else:
                error('Unknown message type')
               
def main():
    comms = Client('192.168.3.113', 65000)
    sensor = Sensor(comms)
    sensor.wait()

if __name__ == "__main__":
    main() 
