#!/bin/sh
# v0.1 - first version
# v0.2 - update widgets version to 3.1.4
# Script to download, build and install the Widgets for CubicSDR

# clear the terminal
clear
# say what we are doing
echo "SDRplay script to download and build wxWidgets - v0.2"
# make the ~/Dev folder (but only if it doesn't already exist)
mkdir -p ~/Dev
# change to the Dev folder
cd ~/Dev
# get some dependencies
sudo apt install freeglut3 freeglut3-dev -y
sudo apt install libpulse-dev libgtk-3-dev -y
# clear these folders in case we've been building previously
rm -rf wxWidgets-staticlib
rm -rf wxWidgets-3.1.4
# download the widgets code
wget https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.4/wxWidgets-3.1.4.tar.bz2
# extract files from the compressed archive
tar -xvjf wxWidgets-3.1.4.tar.bz2
# change to the widgets folder
cd wxWidgets-3.1.4
# make the wxWidgets static library folder
mkdir -p ~/Dev/wxWidgets-staticlib
# run the build scripts
./autogen.sh
./configure --with-opengl --disable-shared --enable-monolithic --with-libjpeg --with-libtiff --with-libpng --with-zlib --disable-sdltest --enable-unicode --enable-display --enable-propgrid --disable-webkit --disable-webview --disable-webviewwebkit --prefix=`echo ~/Dev/wxWidgets-staticlib` CXXFLAGS="-std=c++0x"
# start the compiler
make
# install the library
make install
