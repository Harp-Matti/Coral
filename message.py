class Message:
    pass
    
class Failure(Message):    
    pass
    
class Success(Message):
    pass
    
class Run(Message):
    pass
    
class Result(Message):
    def __init__(self,class_result,spectrum_result):
        self.class_result = class_result
        self.spectrum_result = spectrum_result
        
class Get(Message):
    def __init__(self,parameter):
        self.parameter = parameter
        
class Set(Message):
    def __init__(self,parameter,value):
        self.parameter = parameter
        self.value = value
        
class Return(Message):
    def __init__(self,parameter,value):
        self.parameter = parameter
        self.value = value