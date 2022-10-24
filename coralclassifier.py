from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

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
        self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        
        params = common.input_details(self.interpreter, 'quantization_parameters')
        self.scale = params['scales']
        self.zero_point = params['zero_points']
        
    def run(self,x):
        common.set_input(self.interpreter, np.clip(x/self.scale+self.zero_point,0,255).astype(np.uint8))
        self.interpreter.invoke()
        class_result = classify.get_classes(self.interpreter, top_k=1)
        #print(class_result[0].id, class_result[0].score)
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
        s4 = s2*s2
        cs4 = cs2*cs2
        
        M20 = np.mean(s2)
        M21 = np.mean(s*cs)
        M22 = np.mean(cs2)
        M40 = np.mean(s4)
        M41 = np.mean(s*s2*cs)
        M42 = np.mean(s2*cs2)
        M43 = np.mean(s*cs*cs2)
        M44 = np.mean(cs4)
        M60 = np.mean(s4*s2)
        M61 = np.mean(s4*s*cs)
        M62 = np.mean(s4*cs2)
        M63 = np.mean(s2*s*cs2*cs)

        C20 = M20
        C21 = M21
        C40 = M40-3*np.power(M20,2)
        C41 = M41-3*M20*M21
        C42 = M42-np.power(np.absolute(M20),2)-2*np.power(M21,2)
        C60 = M60-15*M20*M40+30*np.power(M20,3)
        C61 = M61-5*M21*M40-10*M20*M41+30*np.power(M20,2)*M21
        C62 = M62-6*M20*M42-8*M21*M41-M22*M40+6*np.power(M20,2)*M22+24*np.power(M21,2)*M20
        C63 = M63-9*M21*M42+12*np.power(M21,3)-3*M20*M43-3*M22*M41+18*M20*M21*M22
        
        A = np.mean(np.absolute(s/np.sqrt(M21)))
        P = np.unwrap(np.angle(s))
        F = np.gradient(P)
        SP = np.std(P,1)
        SF = np.std(F,1)
        
        return np.asarray([np.absolute(C20/(C21+self.eps)), np.absolute(np.sqrt(C40)/(C21+self.eps)), np.absolute(np.sqrt(C41)/(C21+self.eps)), np.absolute(np.sqrt(C42)/(C21+self.eps)), np.absolute(np.cbrt(C60)/(C21+self.eps)), np.absolute(np.cbrt(C61)/(C21+self.eps)), np.absolute(np.cbrt(C62)/(C21+self.eps)), np.absolute(np.cbrt(C63)/(C21+self.eps)), A, SP, SF])
        
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
