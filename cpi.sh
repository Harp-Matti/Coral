#!/bin/sh
clear
cd sdr
sudo swig -c++ -python -py3 sdr.i
sudo g++ -c -fPIC sdr_wrap.cxx -I/usr/include/python3.9 -lSoapySDR
sudo g++ -shared sdr_wrap.o -o _sdr.so -lSoapySDR
cd ..
