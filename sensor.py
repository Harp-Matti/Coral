#!/usr/bin/env python3

import os
import pathlib
import platform
import socket
import time

script_dir = pathlib.Path(__file__).parent.absolute()
from pycoral.utils import edgetpu

if len(edgetpu.list_edge_tpus()) > 0:
    print("Edge TPU detected")
    from coralclassifier import *
    model_file = os.path.join(script_dir, 'model_hfradio_resnet_maxnorm_qaware_quant_edgetpu.tflite')
else:
    from tfclassifier import *
    model_file = os.path.join(script_dir, 'model_hfradio_resnet_maxnorm_qaware_quant.tflite')

import sys
from time import sleep
import random
from struct import pack

from sdr.sdr import SDR
import numpy as np

from timeit import default_timer as timer

from comms.comms import *

eps = 1.0e-10

rf_file = os.path.join(script_dir, 'model_adaboost.joblib')
svm_file = os.path.join(script_dir, 'model_hfradio_sgd_pwelch_norm_map_128.joblib')

def normalize(x):
    #x -= np.mean(x,1,keepdims=True)
    #x /= np.sqrt(np.mean(np.sum(np.power(x,2),axis=0,keepdims=True),axis=1,keepdims=False))+eps
    x /= np.sqrt(np.amax(np.sum(np.power(x,2),axis=0,keepdims=False)))+eps
    return x

def pwelch(x,n):
    psd = np.zeros((n))
    w = np.hanning(n)
    N = int(np.floor((x.size/2-n)/(n/2))+1)
    for i in range(N):
        psd = psd + abs(np.fft.fft((x[0,0,int(i*n/2):int(i*n/2+n),0] + 1j*x[0,1,int(i*n/2):int(i*n/2+n),0])* w))**2/(N*n)
    #ma = np.amax(psd)
    #psd = psd/ma
    return psd.tolist()

class Sensor:
    def __init__(self, comms):
        self.comms = comms
        
        self.N_classifications = 3
        self.N_samples = 2*1024
        self.sample_length = self.N_classifications*self.N_samples
        self.device = SDR(self.sample_length)
        r = self.device.getRates()
        self.rates = []
        for i in range(int(len(r)/2)):
            self.rates.append((r[2*i],r[2*i+1]))
        b = self.device.getWidths()
        self.widths = []
        for i in range(int(len(b)/2)):
            self.widths.append((b[2*i],b[2*i+1]))        
        self.gains = self.device.getGains()
        print('SDR set')
           
        self.timeout = 10
        self.params = ['frequency','bandwidth','sample_rate','gain']
        self.values = []
        for par in self.params:
            self.values.append(self.get_parameter(par))
        self.classifiers = []
        self.classifiers.append(NeuralNet(model_file))
        self.classifiers.append(RandomForest(rf_file))
        self.classifiers.append(SVM(svm_file))
        print('Classifiers loaded')
        
    def reset(self):
        del self.device
        self.device = SDR(self.sample_length)
        self.set_parameter('frequency',self.values[0])
        self.set_parameter('bandwidth',self.values[1])
        self.set_parameter('sample_rate',self.values[2])
        self.set_parameter('gain',self.values[2])
        print('Device reset')
        
    def run(self,index): 
        i = 0
        while self.device.receive() < self.sample_length and i < self.timeout:
            self.reset()
            for j in range(len(self.params)):
                self.set_parameter(self.params[j],self.values[j])
            i += 1
            
        if i < self.timeout:
            s = np.asarray(self.device.read()).reshape((self.sample_length,2)).T
            x_a = []
            for j in range(self.N_classifications):
                x_a.append(normalize(s[:,j*self.N_samples:(j+1)*self.N_samples]).reshape(1,2,self.N_samples,1))
            result = []
            start = time.perf_counter()
            for j in range(self.N_classifications):
                result.append(self.classifiers[index].run(x_a[j]))
            rate = self.N_classifications/(time.perf_counter()-start)
            class_result = max(set(result), key = result.count) 
            spectrum = pwelch(s.reshape(1,2,self.sample_length,1),128)
            self.comms.send(Result(class_result,rate,spectrum))
            print('Result sent')
        else:
            self.comms.send(Failure())
    
    def valid_rate(self,rate):
        dist = float('inf')
        match = -1.0
        for low, high in self.rates:
            if rate >= low and rate <= high:
                return rate
            if abs(rate-low) < dist:
                dist = abs(rate-low)
                match = low
            if abs(rate-high) < dist:
                dist = abs(rate-high)
                match = high
        return match
        
    def valid_width(self,width):
        dist = float('inf')
        match = -1.0
        for low, high in self.widths:
            if width >= low and width <= high:
                return width
            if abs(width-low) < dist:
                dist = abs(width-low)
                match = low
            if abs(width-high) < dist:
                dist = abs(width-high)
                match = high
        return match
        
    def valid_gain(self,gain):
        low = self.gains[0]
        high = self.gains[1]
        if gain >= low and gain <= high:
            return gain
        elif gain < low:
            return low
        elif gain > high:
            return high
    
    def get_parameter(self,parameter):
        if parameter == 'frequency':
            return self.device.getFrequency()
        elif parameter == 'bandwidth':
            return self.device.getBandwidth()
        elif parameter == 'sample_rate':
            return self.device.getSampleRate()
        elif parameter == 'gain':
            return self.device.getGain()
        else:
            raise Exception('Unknown parameter for method get_parameter')
    
    def set_parameter(self,parameter,value):
        if parameter == 'frequency':
            self.device.setFrequency(value)
            new_value = self.get_parameter(parameter)
            self.values[0] = new_value
        elif parameter == 'bandwidth':
            self.device.setBandwidth(self.valid_width(value))
            new_value = self.get_parameter(parameter)
            self.values[1] = new_value
        elif parameter == 'sample_rate':
            self.device.setSampleRate(self.valid_rate(value))
            new_value = self.get_parameter(parameter)
            self.values[2] = new_value
        elif parameter == 'gain':
            self.device.setGain(self.valid_gain(value))
            new_value = self.get_parameter(parameter)
            self.values[3] = new_value
        else:
            raise Exception('Unknown parameter for method set_parameter')
            
        print(parameter + ' set')
        return new_value
        
    def wait(self):
        while True:
            print('Waiting for instructions')
            message = self.comms.receive()
            message_type = type(message)
            if message_type == Run:
                if message.index < len(self.classifiers):
                    self.run(message.index)
                else:
                    raise Exception('Run error: invalid classifier index')
            elif message_type == Get:
                value = self.get_parameter(message.parameter)
                self.comms.send(Return(message.parameter,value))
            elif message_type == Set:
                value = self.set_parameter(message.parameter,message.value)
                self.comms.send(Return(message.parameter,value))
            elif message_type == Exit:
                print('Exiting')
                exit()
            else:
                raise Exception('Unknown message type')
               
def main():
    try:
        comms = Client('192.168.3.118', 65000)
    except:
        comms = Client('192.168.3.113', 65000)
    sensor = Sensor(comms)
    sensor.wait()

if __name__ == "__main__":
    main() 
