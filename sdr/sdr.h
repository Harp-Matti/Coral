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
	SDR(int N) {
		SoapySDR::KwargsList results = SoapySDR::Device::enumerate();
		if (results.size() == 0) {
			fprintf(stderr, "No SDR found\n");
		}
		
		SoapySDR::Kwargs args = results[0];
		std::string s = SoapySDR::KwargsToString(args);
		name = (s.substr(0,s.find(','))).substr(s.find('=')+1,std::string::npos);
		
		sdr = SoapySDR::Device::make(args);
		if( sdr == NULL ){
			fprintf(stderr, "SoapySDR::Device::make failed\n");
		}
		
		ranges = sdr->getFrequencyRange( SOAPY_SDR_RX, 0);
		rates = sdr->getSampleRateRange( SOAPY_SDR_RX, 0);
		bandwidths = sdr->getBandwidthRange( SOAPY_SDR_RX, 0);
		
		N_samples = N;
		std::complex<float> b[N];
		buff = b;
		
		rx_stream = sdr->setupStream( SOAPY_SDR_RX, SOAPY_SDR_CF32);
		if(rx_stream == NULL)
		{
			fprintf( stderr, "Failed\n");
			SoapySDR::Device::unmake(sdr);
		}
		streamActive = false;
		
		this->setGainMode(false);
	}
	
	~SDR() {
		if (streamActive){
			deactivateStream();	
		}
		sdr->closeStream(rx_stream);
		SoapySDR::Device::unmake(sdr);
	}
	
	int receive() {
		void *buffs[] = {buff};
		int flags;
		long long time_ns;
		if (!streamActive) {
			activateStream();
		}
		int ret = sdr->readStream(rx_stream, buffs, N_samples, flags, time_ns, 1e5);
		//printf("ret = %d, flags = %d, time_ns = %lld\n", ret, flags, time_ns);
		return ret;
	}
	
	std::vector<float> read() {
		std::vector<float> output;
    		for (int i=0; i<N_samples; i++){
			output.push_back(buff[i].real());
			output.push_back(buff[i].imag());
		}
		return output;	
	}
	
	std::string getName() {
		return name;	
	}
	
	void activateStream() {
		sdr->activateStream(rx_stream, 0, 0, 0);
		streamActive = true;
	}
	
	void deactivateStream() {
		sdr->deactivateStream(rx_stream, 0, 0);
		streamActive = false;
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
		if (freq >= ranges.front().minimum() && freq <= ranges.back().maximum()){
			sdr->setFrequency(SOAPY_SDR_RX, 0, freq);
		} else {
			fprintf(stderr, "Frequency out of bounds\n");
		}
	}
	
	bool getGainMode(){
		return sdr->getGainMode(SOAPY_SDR_RX, 0);
	}
	
	void setGainMode(bool automatic){
		sdr->setGainMode(SOAPY_SDR_RX, 0, automatic);
	}
	
	void listRates(){
		for (int i=0; i < rates.size(); i++){
			std::cout << rates[i].minimum() << ' ' << rates[i].maximum() << std::endl;
		}
	}
	
	std::vector<float> getRates() {
		std::vector<float> output;
		for (int i=0; i < rates.size(); i++){
			output.push_back(rates[i].minimum());
			output.push_back(rates[i].maximum());
		}
		return output;	
	}
	
	std::vector<float> getWidths() {
		std::vector<float> output;
		for (int i=0; i < bandwidths.size(); i++){
			output.push_back(bandwidths[i].minimum());
			output.push_back(bandwidths[i].maximum());
		}
		return output;	
	}
	
	SoapySDR::Device *sdr;
	SoapySDR::RangeList bandwidths;
	SoapySDR::RangeList ranges;
	SoapySDR::RangeList rates;
	SoapySDR::Stream *rx_stream;
	std::complex<float> *buff;
	std::string name;
	bool streamActive;
	int N_samples;
};
