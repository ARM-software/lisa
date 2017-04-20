#!/usr/bin/env bash

# Script to install the depenencies for LISA on an Ubuntu-like system.

# This is intended to be used for setting up containers and virtual machines to
# run LISA (e.g. for CI infrastructure or for Vagrant installation). However for
# the brave, it could be used to set up LISA directly on a real machine.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

usage() {
    echo Usage: "$0" [--install-android-sdk]
}

set -eu

install_android_sdk=n

for arg in "$@"; do
    if [ "$arg" == "--install-android-sdk" ]; then
        install_android_sdk=y
    else
        echo "Unrecognised argument: $arg"
        usage
        exit 1
    fi
done

apt-get update

apt-get -y remove ipython ipython-notebook

apt-get -y install build-essential autoconf automake libtool pkg-config \
    trace-cmd sshpass kernelshark nmap net-tools tree python-matplotlib \
    python-numpy libfreetype6-dev libpng12-dev python-nose python-pip \
    python-dev iputils-ping git wget expect

# Upgrade pip so we can use wheel packages instead of compiling stuff, this is
# much faster.
pip install --upgrade pip

# Incantation to fix broken pip packages
/usr/local/bin/pip install --upgrade packaging appdirs

# Use IPython 5.x because 6.0+ only supports Python 3.3
/usr/local/bin/pip install --upgrade "ipython<6.0.0" Cython trappy bart-py devlib psutil wrapt jupyter

if [ "$install_android_sdk" == y ]; then
    apt-get -y install openjdk-7-jre openjdk-7-jdk
    ANDROID_SDK_URL="https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz"
    mkdir -p "$SCRIPT_DIR"/tools
    if [ ! -e "$SCRIPT_DIR"/tools/android-sdk-linux ]; then
        echo "Downloading Android SDK [$ANDROID_SDK_URL]..."
        wget -qO- $ANDROID_SDK_URL | tar xz -C $SCRIPT_DIR/tools/
        expect -c "
            set timeout -1;
            spawn $SCRIPT_DIR/tools/android-sdk-linux/tools/android update sdk --no-ui
            expect {
                \"Do you accept the license\" { exp_send \"y\r\" ; exp_continue }
                eof
            }
        "
    fi
fi
