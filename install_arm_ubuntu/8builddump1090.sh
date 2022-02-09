#!/bin/sh
# v0.1 - first version
# Script to download, build and install dump1090

# clear the terminal
clear
# say what we are doing
echo "SDRplay script to download and build dump1090 - v0.1"
# make the ~/Dev folder (but only if it doesn't already exist)
mkdir -p ~/Dev
# change to the Dev folder
cd ~/Dev
# remove any previous dump1090 folder
rm -rf dump1090
# we need git, rtl-sdr, cmake and build-essential to build dump1090
sudo apt install git cmake build-essential librtlsdr-dev -y
# download dump1090 code
git clone https://github.com/SDRplay/dump1090 -b api3
# change to the dump1090 folder
cd dump1090
# start the compiler
make
# install dump1090 into the system folders
sudo cp dump1090 /usr/local/bin/dump1090
sudo chmod 755 /usr/local/bin/dump1090
# refresh the library path variable so we can use the software
sudo ldconfig
echo " "
echo "Finished. Current User Guide can be found here..."
echo " "
echo "https://www.sdrplay.com/docs/SDRplay_ADS-B_User_Guide.pdf"
