from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

import numpy as np

class Classifier:
    def __init__(self, model_file):
        # TODO: different functionality for nn, svm or rf based on filename
        self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        
    def run(self,x):
        common.set_input(self.interpreter, x)
        self.interpreter.invoke()
        class_result = classify.get_classes(self.interpreter, top_k=1)
        return class_result[0].id
        