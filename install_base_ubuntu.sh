#!/usr/bin/env bash

# Script to install the depenencies for LISA on an Ubuntu-like system.

# This is intended to be used for setting up containers and virtual machines to
# run LISA (e.g. for CI infrastructure or for Vagrant installation).
# This can also work for a fresh LISA install on a workstation.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

usage() {
    echo Usage: "$0" [--install-android-sdk]
}

install_sdk() {
    apt-get -y install openjdk-8-jre openjdk-8-jdk
    mkdir -p "$SCRIPT_DIR"/tools
    if [ ! -e "$SCRIPT_DIR"/tools/android-sdk-linux ]; then
	ANDROID_SDK_URL="https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz"
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

    ANDROID_EXPORT="export ANDROID_HOME=$(realpath $SCRIPT_DIR/tools/android-sdk-linux)"
    if [ "$(cat ~/.bashrc | grep ANDROID_HOME | wc -l)" -eq 0 ]; then
	echo "$ANDROID_EXPORT" >> ~/.bashrc
    fi

    PLATFORM_TOOLS_EXPORT="export PATH=\$ANDROID_HOME/platform-tools"
    if [ -z "$(cat ~/.bashrc | grep "$PLATFORM_TOOLS_EXPORT")" ]; then
	echo "$PLATFORM_TOOLS_EXPORT:\$PATH" >> ~/.bashrc
    fi

    TOOLS_EXPORT="export PATH=\$ANDROID_HOME/tools"
    if [ -z "$(cat ~/.bashrc | grep "$TOOLS_EXPORT")" ]; then
	echo "$TOOLS_EXPORT:\$PATH" >> ~/.bashrc
    fi
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

# venv is not installed by default on Ubuntu, even though it is part of the
# Python standard library
apt-get -y install build-essential git wget expect kernelshark \
	python3 python3-pip python3-venv python3-tk

# Upgrade pip so we can use wheel packages instead of compiling stuff, this is
# much faster.
python3 -m pip install --upgrade pip

python3 -m pip install -e .[notebook,doc]

if [ "$install_android_sdk" == y ]; then
    install_sdk
fi
