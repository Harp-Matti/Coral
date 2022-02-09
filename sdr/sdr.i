%module sdr
%include std_vector.i

namespace std {
   %template(ComplexVector) vector<complex<float>>;
}

%{
#include "sdr.h"
%}

%include "sdr.h"
