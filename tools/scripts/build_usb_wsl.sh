#!/bin/bash

# Build WSL2 kernel with usb-storage support
# menuconfig -> Device Drivers -> USB support -> USB Mass Storage support

echo "Installing dependencies"
sudo apt install build-essential flex bison libglib2.0-dev libpixman-1-dev libncurses-dev autoconf libudev-dev libtool zip unzip v4l-utils libssl-dev python3-pip cmake git iputils-ping net-tools dwarves bc libelf-dev

cd /tmp || exit
sudo git clone https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel || exit

sudo cp /proc/config.gz .
sudo gunzip config.gz
sudo mv config .config

sudo make menuconfig
sudo make -j"$(nproc)"
sudo make modules_install -j"$(nproc)"
sudo make install -j"$(nproc)"

sudo mkdir /mnt/c/sources/
sudo cp -rf vmlinux /mnt/c/sources/
