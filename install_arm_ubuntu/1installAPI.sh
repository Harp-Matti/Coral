#!/bin/sh
# v0.1 - first version
# v0.2 - fix read command
# v0.3 - replace read command with option to not reboot the system
# Script to download and run the SDRplay RSP API installer
# Then reboot afterwards to make sure the service has started

# clear the terminal
clear
sudo apt-get update
sudo apt-get dist-upgrade
chmod +x 2buildSoapy.sh
# say what we are doing
echo "SDRplay script to download and run the RSP API Installer - v0.3"
# copy restart script
current_path=`dirname $0`
sudo cp ${current_path}/restartService.sh /usr/local/bin/restartSDRplay
sudo chmod 755 /usr/local/bin/restartSDRplay
# change to the Downloads folder
cd ~/Downloads
# download the API from the SDRplay website
wget https://www.sdrplay.com/software/SDRplay_RSP_API-ARM64-3.07.1.run
# change permission so the run file is executable
chmod 755 ./SDRplay_RSP_API-ARM64-3.07.1.run
# execute the API installer (follow the prompts)
./SDRplay_RSP_API-ARM64-3.07.1.run
sudo cp restartService.sh /usr/local/bin/restartSDRplay
sudo chmod 755 /usr/local/bin/restartSDRplay
# the system should be rebooted before the API is used
# ask the user if they want to reboot now or later
echo " "
echo "API installer finished. Once the API is installed, you can use"
echo "the command 'restartSDRplay' if you need to restart the API service."
echo " "
echo "The system should be rebooted before the API is used."
echo -n "Do you want to reboot now? (y/n) :"
old_stty_cfg=$(stty -g)
stty raw -echo
answer=$( while ! head -c 1 | grep -i '[ny]' ;do true; done )
stty $old_stty_cfg
if echo "$answer" | grep -iq "^y" ;then
        echo " "
        echo "Rebooting in 5 seconds..."
        sleep 5
        sudo reboot
else
        echo " "
        echo "Please make sure the system is rebooted before the API is used."
fi
