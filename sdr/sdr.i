%module sdr
%include std_vector.i
%include std_string.i

namespace std {
   %template(FloatVector) vector<float>;
}

%{
#include "sdr.h"
%}

%include "sdr.h"
