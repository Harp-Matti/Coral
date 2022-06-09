import socket
import sys
import time
from struct import pack, unpack, unpack_from

inds_to_parameters = ['frequency','bandwidth','sample_rate','gain']
parameters_to_inds = {
    'frequency': 0,
    'bandwidth': 1,
    'sample_rate': 2,
    'gain': 3
}

class Message:
    pass
    
class Exit(Message):
    pass
    
class Failure(Message):    
    pass
    
class Success(Message):
    pass
    
class Run(Message):
    def __init__(self,index):
        self.index = index
    
class Result(Message):
    def __init__(self,class_result,rate_result,spectrum_result):
        self.class_result = class_result
        self.rate_result = rate_result
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

class Comms:
    def __init__(self,port):   
        self.sock = socket.socket()
        raise Exception('The base class Comms should not be used: use either Server or Client')

    def receive(self):
        # Wait for message
        message = self.sock.recv(4096)
        if len(message) < 4:
            return Exit()
        else:
            message_type = unpack_from('i', message)[0]
            if message_type == 0: 
                return Failure()
            elif message_type == 1:
                return Success()
            elif message_type == 2:
                _ ,index = unpack_from('ii',message)
                return Run(index)
            elif message_type == 3:
                _ ,class_result = unpack_from('ii',message)
                rate_and_spectrum_result = list(unpack_from(int((len(message)-8)/4)*'f',message,8))
                return Result(class_result,rate_and_spectrum_result[0], rate_and_spectrum_result[1:])
            elif message_type == 4:
                _ ,ind = unpack_from('ii',message)
                return Get(inds_to_parameters[ind])
            elif message_type == 5:
                _ ,ind, value = unpack_from('iid',message)
                return Set(inds_to_parameters[ind], value)
            elif message_type == 6:
                _ ,ind, value = unpack_from('iid',message)
                return Return(inds_to_parameters[ind], value)
            else:
                raise Exception('Receive error: Unknown message type')
            
    def send(self, message):
        if isinstance(message,Message):
            message_type = type(message)
            if message_type == Failure:
                self.sock.send(pack('i',0))
            elif message_type == Success:
                self.sock.send(pack('i',1))
            elif message_type == Run:
                self.sock.send(pack('ii', 2, message.index))
            elif message_type == Result:
                N = len(message.spectrum_result)
                self.sock.send(pack('iif%sf' % N, 3, message.class_result, message.rate_result, *message.spectrum_result))
            elif message_type == Get:
                ind = parameters_to_inds[message.parameter]
                self.sock.send(pack('ii', 4, ind))
            elif message_type == Set:
                ind = parameters_to_inds[message.parameter]
                self.sock.send(pack('iid', 5, ind, message.value))
            elif message_type == Return:
                ind = parameters_to_inds[message.parameter]
                self.sock.send(pack('iid', 6, ind, message.value))
            else:
                raise Exception('Send error: Unknown message type')
        else:
            raise Exception('Send error: Invalid message')

class Server(Comms):
    def __init__(self,port):   
        # Create a UDP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        print(f'Starting UDP server on port {port}')
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind(('',port))
            self.bound = True
            self.server_socket.listen(5)
            self.sock, client_address = self.server_socket.accept()
        except:
            self.bound = false
            raise Exception('Failed to connect')

    def __del__(self):  
        if self.bound:  
            self.sock.close()
        self.server_socket.close()
        
class Client(Comms):
    def __init__(self,host,port):   
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        self.server_address = (host, port)
        self.sock.connect(self.server_address)
        
    def __del__(self):
        self.sock.close()
