swig -c++ -python -py3 sdr.i
g++ -c sdr_wrap.cxx -I/usr/include/python3.7m
g++ -shared sdr_wrap.o -o _sdr.so