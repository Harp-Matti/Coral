#!/bin/sh
# v0.1 - first version
# Script to download, build and install RSP TCP Server

# clear the terminal
clear
# say what we are doing
echo "SDRplay script to download and build RSPTCPServer - v0.1"
# make the ~/Dev folder (but only if it doesn't already exist)
mkdir -p ~/Dev
# change to the Dev folder
cd ~/Dev
# remove any previous RSPTCPServer folder
rm -rf RSPTCPServer
# we need git, cmake and build-essential to build RSPTCPServer
sudo apt install git cmake build-essential -y
# download RSPTCPServer code
git clone https://github.com/SDRplay/RSPTCPServer
# change to the RSPTCPServer folder
cd RSPTCPServer
# create build folder
mkdir build
# change to the build folder
cd build
# use cmake to create the build files
cmake ..
# start the compiler
make
# install RSPTCPServer into the system folders
sudo make install
# refresh the library path variable so we can use the software
sudo ldconfig
echo " "
echo "Finished. Current user guide can be found here..."
echo " "
echo "https://www.sdrplay.com/docs/SDRplay_RSP_TCP_Server_Guide.pdf"
