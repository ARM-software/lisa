# This Dockerfile creates an image for use with Workload Automation
# and/or devlib.
#
# To build this Docker image, please run the following command from
# this directory:
#
#   docker build -t wa .
#
# This will create an image called wa, which is preconfigured to
# run WA and devlib. Please note that the build process automatically
# accepts the licenses for the Android SDK, so please be sure that you
# are willing to accept these prior to building and running the image
# in a container.
#
# To run the container, please run the following command from the
# directory you wish to work from:
#
#   docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --volume ${PWD}:/workspace --workdir /workspace wa
#
# The above command starts the container in privileged mode, with
# access to USB devices. The current directory is mounted into the
# image, allowing you to work from there. Any files written to this
# directory are directly written to the host. Additional "volumes",
# such as required assets, can be mounted into the container using a
# second --volume command.
#
# If you require access to a TTY from the Docker container, please
# also mount this into the container in the same style as is used to
# mount USB devices. For example:
#
#   docker run -it --privileged -v /dev/ttyUSB0:/dev/ttyUSB0 -v /dev/bus/usb:/dev/bus/usb --volume ${PWD}:/workspace --workdir /workspace wa
#
# When you are finished, please run `exit` to leave the container.
#
# NOTE: Please make sure that the ADB server is NOT running on the
# host. If in doubt, run `adb kill-server` before running the docker
# container.
#

# We want to make sure to base this on a recent ubuntu release
FROM ubuntu:18.10

# Please update the references below to use different versions of
# devlib, WA or the Android SDK
ARG DEVLIB_REF=v1.1.2
ARG WA_REF=v3.1.4
ARG ANDROID_SDK_URL=https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip

RUN apt-get update
RUN apt-get install -y python3 python3-pip git wget zip openjdk-8-jre-headless vim emacs nano curl sshpass ssh usbutils locales
RUN pip3 install pandas

# Ensure we're using utf-8 as our default encoding
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Let's get the two repos we need, and install them
RUN git clone -v https://github.com/ARM-software/devlib.git /tmp/devlib && cd /tmp/devlib && git checkout $DEVLIB_REF && python3 setup.py install && pip3 install .[full]
RUN git clone -v https://github.com/ARM-software/workload-automation.git /tmp/wa && cd /tmp/wa && git checkout $WA_REF && python3 setup.py install && pip3 install .[all]

# Clean-up
RUN rm -R /tmp/devlib /tmp/wa

# Create and switch to the wa user
RUN useradd -m -G plugdev,dialout wa
USER wa

# Let's set up the Android SDK for the user
RUN mkdir -p /home/wa/.android
RUN mkdir -p /home/wa/AndroidSDK && cd /home/wa/AndroidSDK && wget $ANDROID_SDK_URL -O sdk.zip && unzip sdk.zip
RUN cd /home/wa/AndroidSDK/tools/bin && yes | ./sdkmanager --licenses && ./sdkmanager platform-tools && ./sdkmanager 'build-tools;27.0.3'

# Update the path
RUN echo 'export PATH=/home/wa/AndroidSDK/platform-tools:${PATH}' >> /home/wa/.bashrc
RUN echo 'export PATH=/home/wa/AndroidSDK/build-tools:${PATH}' >> /home/wa/.bashrc
RUN echo 'export ANDROID_HOME=/home/wa/AndroidSDK' >> /home/wa/.bashrc

# Generate some ADB keys. These will change each time the image is build but will otherwise persist.
RUN /home/wa/AndroidSDK/platform-tools/adb keygen /home/wa/.android/adbkey

# We need to make sure to add the remote assets too
RUN wa --version && echo 'remote_assets_url: https://raw.githubusercontent.com/ARM-software/workload-automation-assets/master/dependencies' >> /home/wa/.workload_automation/config.yaml

