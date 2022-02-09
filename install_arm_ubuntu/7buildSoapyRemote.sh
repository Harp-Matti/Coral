#!/bin/sh
# v0.1 - first version
# Script to download, build and install SoapyRemote

# clear the terminal
clear
# say what we are doing
echo "SDRplay script to download and build SoapyRemote - v0.1"
# make the ~/Dev folder (but only if it doesn't already exist)
mkdir -p ~/Dev
# change to the Dev folder
cd ~/Dev
# remove any previous SoapyRemote folder
rm -rf SoapyRemote
# we need compiler, git, cmake and avahi to build SoapyRemote
sudo apt install build-essential git cmake avahi-daemon libavahi-client-dev -y
# download SoapyRemote code
git clone https://github.com/pothosware/SoapyRemote
# change to the SoapyRemote folder
cd SoapyRemote
# create build folder
mkdir build
# change to the build folder
cd build
# use cmake to create the build files
cmake ..
# start the compiler
make
# install SoapyRemote into the system folders
sudo make install
# refresh the library path variable so we can use the library
sudo ldconfig
echo " "
echo "Finished. Visit https://github.com/pothosware/SoapyRemote/wiki for more info"
