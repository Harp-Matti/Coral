import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

device = SDR()

#apply settings
device.setSampleRate(1e6)
device.setFrequency(912.3e6)

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_augmod_quant_edgetpu.tflite')
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
    x.reshape((2,1024))
    x -= np.mean(x,1,keepdims=True)
    x /= np.sqrt(np.mean(np.sum(np.power(x,2))))
    return x

N_classifications = 10
start = timer()
#receive some samples
for i in range(N_classifications):
    device.receive()
    runClassifier(interpreter,labels,normalize(np.array(device.read())))

end = timer()
print(f'Average inference time over {N_classifications} samples: {(end-start)/N_classifications} seconds')

#shutdown the stream
sdr.deactivateStream(rxStream) #stop streaming
sdr.closeStream(rxStream)
#sdr.stop_stream()
