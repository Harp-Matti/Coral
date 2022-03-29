#!/bin/sh
# v0.1 - first version
# Script to download, build and install liquid-dsp

# clear the terminal
clear
# say what we are doing
echo "SDRplay script to download and build liquid-dsp - v0.1"
# make the ~/Dev folder (but only if it doesn't already exist)
mkdir -p ~/Dev
# change to the Dev folder
cd ~/Dev
# remove any previous liquid-dsp folder
rm -rf liquid-dsp
# download autoconf which is required for liquid-dsp
sudo apt install autoconf -y
# download liquid-dsp code
git clone https://github.com/jgaeddert/liquid-dsp
# change to the liquid-dsp folder
cd liquid-dsp
# start the build process
./bootstrap.sh
CFLAGS="-march=native -O3" ./configure --enable-fftoverride
# start the compiler
make
# install liquid-dsp into the system folders
sudo make install
# refresh the library path variable so we can use the library
sudo ldconfig
