from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

import numpy as np

from joblib import load

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class Classifier:
    pass
    
class NeuralNet(Classifier):
    def __init__(self, model_file):
        # TODO: different functionality for nn, svm or rf based on filename
        self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        
    def run(self,x):
        common.set_input(self.interpreter, x)
        self.interpreter.invoke()
        class_result = classify.get_classes(self.interpreter, top_k=1)
        return class_result[0].id
        
class RandomForest(Classifier):
    def __init__(self, model_file):
        self.model = load(model_file)
        self.eps = 1.0e-10
        
    def features(self,x):
        s = np.squeeze(x[0,0,:,0]+1j*x[0,1,:,0])
        cs = np.conj(s)
        s2 = np.power(s,2)
        cs2 = np.conj(s2)
        M20 = np.sum(s2)
        M21 = np.sum(s*cs)
        C40 = np.sum(np.power(s2,2))-3*M20*M20
        C41 = np.sum(s2*s*cs)-3*M21*M20
        C42 = np.sum(s2*cs2)-np.absolute(M20*M20)-2*M21*M21
        C63 = np.sum(s2*s*cs2*cs)-9*C42*M21-6*M21*M21*M21
        return np.asarray([np.absolute(C40)/(np.absolute(C42)+self.eps),np.absolute(C41)/(np.absolute(C42)+self.eps),np.power(np.absolute(C63),2)/(np.power(np.absolute(C42),3)+self.eps)])
        
    def run(self,x):
        return self.model.predict(self.features(x))