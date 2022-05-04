import tflite_runtime.interpreter as tflite

import numpy as np

class Classifier:
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