import os
import pathlib
import platform
import time

import dill

script_dir = pathlib.Path(__file__).parent.absolute()
try:
    from pycoral.utils import edgetpu
    edge = len(edgetpu.list_edge_tpus()) > 0
except:
    edge = False

model_files = [None]*5
if edge:
    print("Edge TPU detected")
    from coralclassifier import *    
    model_files[0] = os.path.join(script_dir, 'model_hfradio_resnet_maxnorm_qaware_quant_edgetpu.tflite')
    model_files[1] = os.path.join(script_dir, 'model_hfradio_deepcnn_maxnorm_qaware_quant_edgetpu.tflite')
    model_files[2] = os.path.join(script_dir, 'model_hfradio_resnetos_maxnorm_qaware_quant_edgetpu.tflite')
else:
    from tfclassifier import *
    model_files[0] = os.path.join(script_dir, 'model_hfradio_resnet_maxnorm_qaware_quant.tflite')
    model_files[1] = os.path.join(script_dir, 'model_hfradio_deepcnn_maxnorm_qaware_quant.tflite')
    model_files[2] = os.path.join(script_dir, 'model_hfradio_resnetos_maxnorm_qaware_quant.tflite')
    
import sys
from time import sleep
import random
from struct import pack
import numpy as np
from numpy.random import default_rng

from timeit import default_timer as timer

eps = 1.0e-10

model_files[3] = os.path.join(script_dir, 'model_hfradio_sgd_pwelch_norm_map_128.joblib')
model_files[4] = os.path.join(script_dir, 'model_adaboost_hfradio_11.joblib')

def normalize(x):
    x /= np.sqrt(np.amax(np.sum(np.power(x,2),axis=0,keepdims=False)))+eps
    return x
  
class Sensor:
    def __init__(self): 
        self.N_classifications = 10
        self.N_samples = 2*1024
        self.sample_length = self.N_classifications*self.N_samples
            
        self.rng = default_rng() 
          
        self.classifiers = []
        self.classifiers.append(NeuralNet(model_files[0]))
        self.classifiers.append(NeuralNet(model_files[1]))
        self.classifiers.append(NeuralNet(model_files[2]))
        self.classifiers.append(SVM(model_files[3]))
        self.classifiers.append(RandomForest(model_files[4]))
        print('Classifiers loaded')
        
    def run(self,index): 
        s = normalize(self.rng.standard_normal((2,self.N_samples))).reshape((1,2,self.N_samples,1))
        start = time.perf_counter()
        for j in range(self.N_classifications):
            self.classifiers[index].run(s)
        rate = self.N_classifications/(time.perf_counter()-start)
        print(f'Classification rate for classifier index {index}: {rate}')
        
    def test(self):
        for i in range(len(self.classifiers)):
            self.run(i)
               
def main():
    sensor = Sensor()
    sensor.test()

if __name__ == "__main__":
    main() 
