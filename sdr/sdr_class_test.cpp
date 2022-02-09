#include "sdr.h"

int main()
{
	double freq = 1e9, rate = 1e6;

	SDR S = SDR();
	S.setFrequency(freq);
    S.setSampleRate(rate);	
	std::cout << S.getFrequency() << std::endl;
	std::cout << S.getSampleRate() << std::endl;
	std::cout << S.receiveSamples() << std::endl;
	~S();
	return 0;
}