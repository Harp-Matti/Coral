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

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

host, port = '192.168.3.113', 65000
server_address = (host, port)

N_samples = 2*1024
device = SDR(N_samples)

#apply settings
device.setSampleRate(3.2e6)
device.setBandwidth(8.0e6)
device.setFrequency(1.0e9)

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
#model_file = os.path.join(script_dir, 'model_augmod_quant_edgetpu.tflite')
model_file = os.path.join(script_dir, 'model_hfradio_resnet_quant_edgetpu.tflite')
label_file = os.path.join(script_dir, 'classes_hfradio.txt')
labels = dataset.read_label_file(label_file)

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

def runClassifier(interpreter,labels,x):
    common.set_input(interpreter, x)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
        
        # Pack three 32-bit floats into message and send
        message = pack('3f', x, y, z)
        sock.sendto(message, server_address)

eps = 1.0e-10        
        
def normalize(x):
    x -= np.mean(x,1,keepdims=True)
    x /= np.sqrt(np.mean(np.sum(np.power(x,2))))+eps
    return x

N_classifications = 11
start = timer()
#receive some samples
for i in range(N_classifications):
    if device.receive() < N_samples:
        print('Receive failed')
    #device.setFrequency(1.0e9+(i+1)*10.0e6)    
    #device.read()
    runClassifier(interpreter,labels,normalize(np.asarray(device.read()).reshape((2,N_samples))).reshape(1,2,N_samples,1))

end = timer()
print(f'Average inference time over {N_classifications} samples: {(end-start)/N_classifications} seconds')
