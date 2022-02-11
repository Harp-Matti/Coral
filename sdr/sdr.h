#include <cstdio>
#include <cstdlib>

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>

#include <string>
#include <vector>
#include <map>

#include <iostream>

class SDR {
public:
	SDR() {
		SoapySDR::KwargsList results = SoapySDR::Device::enumerate();
		if (results.size() == 0) {
			fprintf(stderr, "No SDR found\n");
		}
		
		SoapySDR::Kwargs args = results[0];
		sdr = SoapySDR::Device::make(args);
		if( sdr == NULL ){
			fprintf(stderr, "SoapySDR::Device::make failed\n");
		}
		
		ranges = sdr->getFrequencyRange( SOAPY_SDR_RX, 0);
		//setBufferSize(1024);
		
		rx_stream = sdr->setupStream( SOAPY_SDR_RX, SOAPY_SDR_CF32);
		if(rx_stream == NULL)
		{
			fprintf( stderr, "Failed\n");
			SoapySDR::Device::unmake(sdr);
		}
		sdr->activateStream(rx_stream, 0, 0, 0);
	}
	
	~SDR() {
		sdr->deactivateStream(rx_stream, 0, 0);
		sdr->closeStream(rx_stream);
		SoapySDR::Device::unmake(sdr);
	}
	
	int receive() {
		void *buffs[] = {buff};
		int flags;
		long long time_ns;
		int ret = sdr->readStream(rx_stream, buffs, 1024, flags, time_ns, 1e5);
		//printf("ret = %d, flags = %d, time_ns = %lld\n", ret, flags, time_ns);
		return ret;
	}
	
	std::vector<float> read() {
		std::vector<float> output;
    		for (int i=0; i<1024; i++){
			output.push_back(buff[i].real());
			output.push_back(buff[i].imag());
		}
		return output;	
	}
	
	double getSampleRate(){
		return sdr->getSampleRate(SOAPY_SDR_RX, 0);
	}
	
	void setSampleRate(double rate){
		sdr->setSampleRate(SOAPY_SDR_RX, 0, rate);
	}
	
	double getBandwidth(){
		return sdr->getBandwidth(SOAPY_SDR_RX, 0);
	}
	
	void setBandwidth(double bw){
		sdr->setBandwidth(SOAPY_SDR_RX, 0, bw);
	}
	
	double getFrequency(){
		return sdr->getFrequency(SOAPY_SDR_RX, 0);
	}
	
	void setFrequency(double freq){
		if (freq >= ranges[0].minimum() && freq <= ranges[0].maximum()){
			sdr->setFrequency(SOAPY_SDR_RX, 0, freq);
		} else {
			fprintf(stderr, "Frequency out of bounds\n");
		}
	}
	
	SoapySDR::Device *sdr;
	SoapySDR::RangeList ranges;
	SoapySDR::Stream *rx_stream;
	std::complex<float> buff[1024];
};