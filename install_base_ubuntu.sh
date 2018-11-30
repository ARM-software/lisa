#!/usr/bin/env bash

# Script to install the depenencies for LISA on an Ubuntu-like system.

# This is intended to be used for setting up containers and virtual machines to
# run LISA (e.g. for CI infrastructure or for Vagrant installation).
# This can also work for a fresh LISA install on a workstation.

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPT_DIR"

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
	    spawn $SCRIPT_DIR/tools/android-sdk-linux/tools/android \
		    update sdk --no-ui -t tools,platform-tools
	    expect {
		\"Do you accept the license\" { exp_send \"y\r\" ; exp_continue }
		eof
	    }
	"
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
	python3 python3-pip python3-venv python3-tk gobject-introspection \
	libcairo2-dev libgirepository1.0-dev gir1.2-gtk-3.0

if [ "$install_android_sdk" == y ]; then
    install_sdk
fi
