import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

import numpy as np #use numpy for buffers

from timeit import default_timer as time

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
    IQ = np.array((2,1024), dtype=np.float32)
    IQ[0,:] = np.real(x)
    IQ[1,:] = np.imag(x)
    return IQ

N_classifications = 10
start = timer()
#receive some samples
for i in range(N_classifications):
    runClassifier(interpreter,labels,normalize(buff))

end = timer()
print(f'Average inference time over {N_classifications} samples: {(end-start)/N_classifications} seconds')
