%module sdr
%include std_vector.i

namespace std {
   %template(FloatVector) vector<float>;
}

%{
#include "sdr.h"
%}

%include "sdr.h"
