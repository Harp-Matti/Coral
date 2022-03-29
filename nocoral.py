import os
import pathlib
import tflite_runtime.interpreter as tflite

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

N_samples = 2*1024
device = SDR(N_samples)

#apply settings
device.setSampleRate(3.2e6)
print('sample rate set')
device.setBandwidth(8.0e6)
print('bandwidth set')
device.setFrequency(1.0e9)
print('frequency set')

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_hfradio_resnet_quant.tflite')
label_file = os.path.join(script_dir, 'classes_hfradio.txt')

# Initialize the TF interpreter
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = load_labels(label_file)

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def runClassifier(interpreter,labels,x):
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))

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
