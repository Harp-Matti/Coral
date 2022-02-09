#!/bin/sh
# v0.1 - first version
# v0.2 - fix read command
# v0.3 - remove read command
# Script to download, build and install SoapySDR framework
# Then to the same for SoapySDRPlay and finally test everything works

# clear the terminal
clear
# say what we are doing
echo "SDRplay script to download and build SoapySDR - v0.3"
# make the ~/Dev folder (but only if it doesn't already exist)
mkdir -p ~/Dev
# change to the Dev folder
cd ~/Dev
# remove any previous SoapySDR folder
rm -rf SoapySDR
# we need compiler, git, cmake and swig to build SoapySDR
sudo apt install build-essential git cmake swig g++ libpython3-dev python3-numpy python3-dev -y
# download SoapySDR code
git clone https://github.com/pothosware/SoapySDR
# change to the SoapySDR folder
cd SoapySDR
# create build folder
mkdir build
# change to the build folder
cd build
# use cmake to create the build files
cmake ..
# start the compiler
make
# install SoapySDR into the system folders
sudo make install
# refresh the library path variable so we can use the library
sudo ldconfig
echo "SoapySDR build finished. Continuing to build SoapySDRPlay..."
# change to the Dev folder again
cd ~/Dev
# remove any previous SoapySDRPlay folder
rm -rf SoapySDRPlay3
# download SoapySDRPlay (for API 3)
git clone https://github.com/SDRplay/SoapySDRPlay3
# change to the SoapySDRPlay folder
cd SoapySDRPlay3
# create the build folder
mkdir build
# change to the build folder
cd build
# use cmake to create the build files
cmake ..
# start the compiler
make
# install SoapySDRPlay library into the SoapySDR framework
sudo make install
# refresh the library path variable so we can use the library
sudo ldconfig
# ask the user if they want to test the SoapySDR framework now
echo "SoapySDR and SoapySDRPlay should now be built."
echo -n "If an RSP is connected, this can be tested - ok to test? (y/n) :"
old_stty_cfg=$(stty -g)
stty raw -echo
answer=$( while ! head -c 1 | grep -i '[ny]' ;do true; done )
stty $old_stty_cfg
if echo "$answer" | grep -iq "^y" ;then
	echo " "
	SoapySDRUtil --probe="driver=sdrplay"
else
        echo " "
        echo "The SoapySDR framework can be tested by using the SoapySDRUtil command"
	echo "Some examples of tests you can run are..."
	echo "SoapySDRUtil --find=\"driver=sdrplay\""
	echo "SoapySDRUtil --probe=\"driver=sdrplay\""
	echo "SoapySDRUtil --args=\"driver=sdrplay\" --rate=2e6 --direction=RX"
fi
# display useful information
echo " "
echo "For more information on SoapySDR and SoapySDRPlay, visit the following pages"
echo " "
echo "SoapySDR - https://github.com/pothosware/SoapySDR/wiki"
echo "SoapySDRPlay - https://github.com/pothosware/SoapySDRPlay3/wiki"
echo " "
echo "For both projects, click on Issues at the top of the page for help/support"
