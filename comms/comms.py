import socket
import sys
import time
from struct import pack, unpack, unpack_from

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

class Comms:
    pass

class Server(Comms):
    def __init__(self,host,port):   
        # Create a UDP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        self.server_address = (host, port)
        print(f'Starting UDP server on {host} port {port}')
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(5)
        self.sock, client_address = self.server_socket.accept()
        #self.sock.connect(self.server_address)
        
    def receive(self):
        # Wait for message
        message, address = self.sock.recv(4096)
        message_type = unpack_from('i', message)[0]
        if message_type == 0: 
            return Failure()
        elif message_type == 1:
            return Success()
        elif message_type == 2:
            return Run()
        elif message_type == 3:
            class_result = unpack_from('i',4,message)[0]
            spectrum_result = list(unpack_from((len(message)-8)*'f',8,message))
            return Result(class_result, spectrum_result)
        elif message_type == 4:
            return Get(unpack_from('i',4,message)[0])
        elif message_type == 5:
            parameter, value = unpack_from('id',4,message)
            return Set(parameter, value)
        elif message_type == 6:
            parameter, value = unpack_from('id',4,message)
            return Return(parameter, value)
        else:
            error('Receive error: Unknown message type')
            
    def send(self, message):
        if isinstance(message,Message):
            message_type = type(message)
            if message_type == Failure:
                self.sock.send(pack('i',0))
            elif message_type == Success:
                self.sock.send(pack('i',1))
            elif message_type == Run:
                self.sock.send(pack('i',2))
            elif message_type == Result:
                N = len(message.spectrum_result)
                self.sock.send(pack('ii%sf' % N, 3, message.class_result, *message.spectrum_result))
            elif message_type == Get:
                self.sock.send(pack('ii', 4, message.parameter))
            elif message_type == Set:
                self.sock.send(pack('iid', 5, message.parameter, message.value))
            elif message_type == Return:
                self.sock.send(pack('iid', 6, message.parameter, message.value))
            else:
                error('Send error: Unknown message type')
        else:
            error('Send error: Invalid message')

class Client(Comms):
    def __init__(self,host,port):   
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        self.server_address = (host, port)
        self.sock.connect(self.server_address)
        
    def receive(self):
        # Wait for message
        message, address = self.sock.recv(4096)
        message_type = unpack_from('i', message)[0]
        if message_type == 0: 
            return Failure()
        elif message_type == 1:
            return Success()
        elif message_type == 2:
            return Run()
        elif message_type == 3:
            class_result = unpack_from('i',4,message)[0]
            spectrum_result = list(unpack_from((len(message)-8)*'f',8,message))
            return Result(class_result, spectrum_result)
        elif message_type == 4:
            return Get(unpack_from('i',4,message)[0])
        elif message_type == 5:
            parameter, value = unpack_from('id',4,message)
            return Set(parameter, value)
        elif message_type == 6:
            parameter, value = unpack_from('id',4,message)
            return Return(parameter, value)
        else:
            error('Receive error: Unknown message type')
            
    def send(self, message):
        if isinstance(message,Message):
            message_type = type(message)
            if message_type == Failure:
                self.sock.send(pack('i',0))
            elif message_type == Success:
                self.sock.send(pack('i',1))
            elif message_type == Run:
                self.sock.send(pack('i',2))
            elif message_type == Result:
                N = len(message.spectrum_result)
                self.sock.send(pack('ii%sf' % N, 3, message.class_result, *message.spectrum_result))
            elif message_type == Get:
                self.sock.send(pack('ii', 4, message.parameter))
            elif message_type == Set:
                self.sock.send(pack('iid', 5, message.parameter, message.value))
            elif message_type == Return:
                self.sock.send(pack('iid', 6, message.parameter, message.value))
            else:
                error('Send error: Unknown message type')
        else:
            error('Send error: Invalid message')