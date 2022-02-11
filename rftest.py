import os
import pathlib

from pycoral.utils import dataset

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

from sdr.sdr import SDR
import numpy as np #use numpy for buffers

from timeit import default_timer as timer

from joblib import load

device = SDR()

#apply settings
device.setSampleRate(10.0e6)
device.setBandwidth(8.0e6)
device.setFrequency(1.0e9)

model = load('model_adaboost.joblib')
label_file = os.path.join(script_dir, 'classes.txt')
labels = dataset.read_label_file(label_file)

def runClassifier(model,labels,x):
    common.set_input(interpreter, x)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

eps = 1.0e-10        
        
def normalize(x):
    x -= np.mean(x,1,keepdims=True)
    x /= np.sqrt(np.mean(np.sum(np.power(x,2))))+eps
    return x

def features(x):
    s = x[0,:]+1j*x[1,:]
    cs = np.conj(s)
    s2 = np.power(s,2)
    cs2 = np.conj(s2)
    M20 = np.reduce_sum(s2)
    M21 = np.reduce_sum(s*cs)
    C40 = np.reduce_sum(np.power(s2,2))-3*M20*M20
    C41 = np.reduce_sum(s2*s*cs)-3*M21*M20
    C42 = np.reduce_sum(s2*cs2)-np.absolute(M20*M20)-2*M21*M21
    C63 = np.reduce_sum(s2*s*cs2*cs)-9*C42*M21-6*C21*C21*C21
    return [np.absolute(C40)/np.absolute(C42),np.absolute(C41)/np.absolute(C42),np.power(np.absolute(C63),2)/np.power(np.absolute(C42)+eps,3)]
  
N_classifications = 11
start = timer()
#receive some samples
for i in range(N_classifications):
    if device.receive() < 1024:
        print('Receive failed')
    #device.setFrequency(1.0e9+(i+1)*10.0e6)    
    runClassifier(model,labels,features(normalize(np.asarray(device.read()).reshape((2,1024)))))

end = timer()
print(f'Average inference time over {N_classifications} samples: {(end-start)/N_classifications} seconds')