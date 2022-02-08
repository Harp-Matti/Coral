import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
#import simplesoapy as sp
import numpy as np #use numpy for buffers

from timeit import default_timer as time

#enumerate devices
results = SoapySDR.Device.enumerate()
for result in results: print(result)
#print(sp.detect_devices(as_string=True))

#create device instance
#args can be user defined or from the enumeration result
args = dict(driver="sdrplay")
sdr = SoapySDR.Device(args)
#sdr = sp.SoapyDevice('driver=sdrplay')

#query device info
print(sdr.listAntennas(SOAPY_SDR_RX, 0))
print(sdr.listGains(SOAPY_SDR_RX, 0))
freqs = sdr.getFrequencyRange(SOAPY_SDR_RX, 0)
for freqRange in freqs: print(freqRange)

#apply settings
sdr.setSampleRate(SOAPY_SDR_RX, 0, 1e6)
sdr.setFrequency(SOAPY_SDR_RX, 0, 912.3e6)
#sdr.sample_rate = 2.56e6
#sdr.freq = 88e6

#setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream) #start streaming
#sdr.start_stream()

#create a re-usable buffer for rx samples
buff = np.array([0]*1024, numpy.complex64)

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_augmod_mean.tflite')
label_file = os.path.join(script_dir, 'classes.txt')
labels = dataset.read_label_file(label_file)

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

def runClassifier(interpreter,labels,x):
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

def normalize(x):
    x -= np.mean(x)
    x /= np.sqrt(np.mean(np.absolute(x)))
    IQ = np.array((2,1024), dtype=numpy.float32)
    IQ[0,:] = np.real(x)
    IQ[1,:] = np.imag(x)
    return IQ

N_classifications = 10
start = timer()
#receive some samples
for i in range(N_classifications):
    sr = sdr.readStream(rxStream, [buff], len(buff))
    print(sr.ret) #num samples or error code
    print(sr.flags) #flags set by receive operation
    print(sr.timeNs) #timestamp for receive buffer
    #sdr.read_stream_into_buffer(buff)
    runClassifier(interpreter,labels,normalize(buff))

end = timer()
print(f'Average inference time over {N_classifications} samples: {(end-start)/N_classifications} seconds')

#shutdown the stream
sdr.deactivateStream(rxStream) #stop streaming
sdr.closeStream(rxStream)
#sdr.stop_stream()
