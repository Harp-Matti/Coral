import socket
import sys
import time
from struct import pack, unpack, unpack_from
from message import Message, Failure, Success, Run, Result, Get, Set

class Comms:
    def __init__(self,host,port):   
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind the socket to the port
        self.server_address = (host, port)
        print(f'Starting UDP server on {host} port {port}')
        self.sock.bind(self.server_address)
        
    def receive(self):
        # Wait for message
        message, address = self.sock.recvfrom(4096)
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
                self.sock.sendto(pack('i'),0)
            elif message_type == Success:
                self.sock.sendto(pack('i'),1)
            elif message_type == Run:
                self.sock.sendto(pack('i'),2)
            elif message_type == Result:
                N = len(message.spectrum_result)
                self.sock.sendto(pack('ii%sf') % N, 3, message.class_result, *message.spectrum_result)
            elif message_type == Get:
                self.sock.sendto(pack('ii'), 4, message.parameter)
            elif message_type == Set:
                self.sock.sendto(pack('iid'), 5, message.parameter, message.value)
            elif message_type == Return:
                self.sock.sendto(pack('iid'), 5, message.parameter, message.value)
            else:
                error('Send error: Unknown message type')
        else:
            error('Send error: Invalid message')