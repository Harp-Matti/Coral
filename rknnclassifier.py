from rknn.api import RKNN

import numpy as np

from joblib import load

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

class Classifier:
    pass
    
class NeuralNet(Classifier):
    def __init__(self, model_file):
        self.interpreter = RKNN()
        #self.interpreter.load_rknn(path=model_file)
        #self.interpreter.init_runtime()
        
        #params = common.input_details(self.interpreter, 'quantization_parameters')
        #self.scale = params['scales']
        #self.zero_point = params['zero_points']
        
    def __del__(self):
        self.interpreter.release()
        
    def run(self,x):
        #common.set_input(self.interpreter, np.clip(x/self.scale+self.zero_point,0,255).astype(np.uint8))
        class_result = self.interpreter.inference(inputs=[x])[0].reshape(-1)
        print(class_result)
        return np.argmax(class_result[0])
        
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
        return self.model.predict(self.features(x).reshape(1,-1))[0]
        
class SVM(Classifier):
    def __init__(self, model_file):
        model_and_map = load(model_file)
        self.model = model_and_map[0]
        self.feature_map = model_and_map[1]
        
    def pwelch(self,x,n):
        psd = np.zeros((n))
        w = np.hanning(n)
        N = int(np.floor((len(x)-n)/(n/2))+1)
        for i in range(N):
            psd = psd + abs(np.fft.fft(x[int(i*n/2):int(i*n/2+n)] * w))**2/N
        ma = np.amax(psd)    
        return np.fft.fftshift(psd)/ma
        
    def features(self,x):
        y = self.pwelch(np.squeeze(x[0,0,:,0]+1j*x[0,1,:,0]),128)
        return self.feature_map.transform([y])
        
    def run(self,x):
        return self.model.predict(self.features(x).reshape(1,-1))[0]
