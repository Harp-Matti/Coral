#!/bin/sh
# v0.1 - first version
# Script to restart the RSP API service

sudo systemctl stop sdrplay
sudo pkill sdrplay_apiService
sudo rm -f /dev/shm/Glbl\\sdrSrv*
sudo systemctl start sdrplay
