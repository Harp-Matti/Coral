#!/bin/bash

swig -c++ -python -py3 sdr.i
g++ -c -fPIC sdr_wrap.cxx -I/usr/include/python3.7m -lSoapySDR
g++ -shared sdr_wrap.o -o _sdr.so -lSoapySDR
