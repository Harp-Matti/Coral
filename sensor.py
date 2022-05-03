import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
#import tflite_runtime.interpreter as tflite

import socket
import sys
from time import sleep
import random
from struct import pack

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

from comms.comms import Comms, Failure, Success, Run, Result, Get, Set

class Sensor:
  def __init__(self, comms):
    self.comms = comms
    
    self.N_samples = 2*1024
    self.device = SDR(self.N_samples)

    self.eps = 1.0e-10
    
    #apply settings
    self.device.setSampleRate(3.2e6)
    self.device.setBandwidth(8.0e6)
    self.device.setFrequency(1.0e9)

    # Specify the TensorFlow model, labels, and image
    script_dir = pathlib.Path(__file__).parent.absolute()
    #model_file = os.path.join(script_dir, 'model_augmod_quant_edgetpu.tflite')
    model_file = os.path.join(script_dir, 'model_hfradio_resnet_quant_edgetpu.tflite')
    
    #self.label_file = os.path.join(script_dir, 'classes_hfradio.txt')
    #labels = dataset.read_label_file(label_file)

    # Initialize the TF interpreter
    self.interpreter = edgetpu.make_interpreter(model_file)
    self.interpreter.allocate_tensors()

  def normalize(x):
    x -= np.mean(x,1,keepdims=True)
    x /= np.sqrt(np.mean(np.sum(np.power(x,2))))+eps
    return x
  
  def pwelch(x,n):
    psd = np.zeros((n))
    w = np.hanning(n)
    N = int(np.floor((len(x)-n)/(n/2))+1)
    for i in range(N):
        psd = psd + abs(np.fft.fft(x[int(i*n/2):int(i*n/2+n)] * w))**2/N
    #ma = np.amax(psd)
    #psd = psd/ma
    return psd.tolist()
  
  def run(self):
      if self.device.receive() < self.N_samples:
        print('Receive failed')
        self.comms.send(Failure())
      else:
        x = normalize(np.asarray(self.device.read()).reshape((2,self.N_samples))).reshape(1,2,self.N_samples,1)
        common.set_input(self.interpreter, x)
        self.interpreter.invoke()
        class_result = classify.get_classes(self.interpreter, top_k=1)
        self.comms.send(Result(class_result,pwelch(x,128)))
        
  def wait(self):
    while True:
      message = self.comms.receive()
      message_type = type(message)
      if message_type == Run:
        self.run()
               
def main():
  comms = Comms('192.168.3.112', 65000)
  sensor = Sensor(comms)
  sensor.wait()

if __name__ == "__main__":
  main() 
