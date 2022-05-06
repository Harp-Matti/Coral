import tflite_runtime.interpreter as tflite

import numpy as np

from joblib import load

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class Classifier:
    pass
    
class NeuralNet(Classifier):
    def __init__(self, model_file):
        self.interpreter = tflite.interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self,x):
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = np.squeeze(output_data)
        return results.argsort()[-1]

class RandomForest(Classifier):
    def __init__(self, model_file):
        self.model = load(model_file)

    def features(x):
        s = x[0,:]+1j*x[1,:]
        cs = np.conj(s)
        s2 = np.power(s,2)
        cs2 = np.conj(s2)
        M20 = np.sum(s2)
        M21 = np.sum(s*cs)
        C40 = np.sum(np.power(s2,2))-3*M20*M20
        C41 = np.sum(s2*s*cs)-3*M21*M20
        C42 = np.sum(s2*cs2)-np.absolute(M20*M20)-2*M21*M21
        C63 = np.sum(s2*s*cs2*cs)-9*C42*M21-6*M21*M21*M21
        return np.asarray([np.absolute(C40)/(np.absolute(C42)+eps),np.absolute(C41)/(np.absolute(C42)+eps),np.power(np.absolute(C63),2)/(np.power(np.absolute(C42),3)+eps)])

    def run(self,x):
        return self.model.predict(features(x))