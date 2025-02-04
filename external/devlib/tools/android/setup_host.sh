#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to install Android SDK tools for LISA & devlib on an Ubuntu-like
# system and creates Android virtual devices.

# shellcheck disable=SC2317

if [[ -z ${ANDROID_HOME:-} ]]; then
    ANDROID_HOME="$(dirname "${BASH_SOURCE[0]}")/android-sdk-linux"
    export ANDROID_HOME
fi
export ANDROID_USER_HOME="${ANDROID_HOME}/.android"

ANDROID_CMDLINE_VERSION=${ANDROID_CMDLINE_VERSION:-"11076708"}

# Android SDK is picky on Java version, so we need to set JAVA_HOME manually.
# In most distributions, Java is installed under /usr/lib/jvm so use that.
# according to the distribution
ANDROID_SDK_JAVA_VERSION=17


# Read standard /etc/os-release file and extract the needed field lsb_release
# binary is not installed on all distro, but that file is found pretty much
# anywhere.
read_os_release() {
    local field_name=${1}
    # shellcheck source=/etc/os-release
    (source /etc/os-release &> /dev/null && printf "%s" "${!field_name}")
}

# Test the value of a field in /etc/os-release
test_os_release() {
    local field_name=${1}
    local value=${2}

    if [[ "$(read_os_release "${field_name}")" == "${value}" ]]; then
        return 0
    fi
    return 1
}

get_android_sdk_host_arch() {
    # Default to Google ABI type for Arm platforms
    local arch="arm64-v8a"
    local machine

    machine=$(uname -m)
    if [[ "${machine}" == "x86"* ]]; then
        arch=${machine}
    fi

    echo "${arch}"
}

# No need for the whole SDK for this one
install_android_platform_tools() {
    echo "Installing Android Platform Tools ..."

    local url="https://dl.google.com/android/repository/platform-tools-latest-linux.zip"

    echo "Downloading Android Platform Tools from: ${url}"
    wget -qO- "${url}" | bsdtar -xf- -C "${ANDROID_HOME}/"
}

cleanup_android_home() {
    echo "Cleaning up Android SDK: ${ANDROID_HOME}"
    rm -rf "${ANDROID_HOME}"
    mkdir -p "${ANDROID_HOME}/cmdline-tools"
}

install_android_sdk_manager() {
    echo "Installing Android SDK manager ..."

    # URL taken from "Command line tools only": https://developer.android.com/studio
    local url="https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_CMDLINE_VERSION}_latest.zip"

    echo "Downloading Android SDK manager from: $url"
    wget -qO- "${url}" | bsdtar -xf- -C "${ANDROID_HOME}/cmdline-tools"

    echo "Moving commandlinetools to cmdline-tools/latest..."
    # First, clear cmdline-tools/latest if it exists.
    rm -rf "${ANDROID_HOME}/cmdline-tools/latest"
    mv "${ANDROID_HOME}/cmdline-tools/cmdline-tools" "${ANDROID_HOME}/cmdline-tools/latest"

    chmod +x -R "${ANDROID_HOME}/cmdline-tools/latest/bin"

    yes | (call_android_sdkmanager --licenses || true)
}

find_java_home() {
    _JAVA_BIN=$(find -L /usr/lib/jvm -path "*${ANDROID_SDK_JAVA_VERSION}*/bin/java" -not -path '*/jre/bin/*' -print -quit)
    _JAVA_HOME=$(dirname "${_JAVA_BIN}")/../

    echo "Found JAVA_HOME=${_JAVA_HOME}"
}

call_android_sdk() {
    local tool="${ANDROID_HOME}/cmdline-tools/latest/bin/${1}"
    shift
    JAVA_HOME=${_JAVA_HOME} "${tool}" "$@"
}

call_android_sdkmanager() {
    call_android_sdk sdkmanager "$@"
}

call_android_avdmanager() {
    call_android_sdk avdmanager "$@"
}

install_build_tools() {
    yes | call_android_sdkmanager --verbose --channel=0 --install "build-tools;34.0.0"
}

install_platform_tools() {
    yes | call_android_sdkmanager --verbose --channel=0 --install "platform-tools"
}

install_platforms() {
    yes | call_android_sdkmanager --verbose --channel=0 --install "platforms;android-31"
    yes | call_android_sdkmanager --verbose --channel=0 --install "platforms;android-33"
    yes | call_android_sdkmanager --verbose --channel=0 --install "platforms;android-34"
}

install_system_images() {
    local android_sdk_host_arch
    android_sdk_host_arch=$(get_android_sdk_host_arch)

    yes | call_android_sdkmanager --verbose --channel=0 --install "system-images;android-31;google_apis;${android_sdk_host_arch}"
    yes | call_android_sdkmanager --verbose --channel=0 --install "system-images;android-33;android-desktop;${android_sdk_host_arch}"
    yes | call_android_sdkmanager --verbose --channel=0 --install "system-images;android-34;google_apis;${android_sdk_host_arch}"
}

create_android_vds() {
    local android_sdk_host_arch
    android_sdk_host_arch=$(get_android_sdk_host_arch)

    local vd_name
    vd_name="devlib-p6-12"
    echo "Creating virtual device \"${vd_name}\" (Pixel 6 - Android 12)..."
    echo no | call_android_avdmanager -s create avd -n "${vd_name}" -k "system-images;android-31;google_apis;${android_sdk_host_arch}" -b "${android_sdk_host_arch}" -f

    vd_name="devlib-p6-14"
    echo "Creating virtual device \"${vd_name}\" (Pixel 6 - Android 14)..."
    echo no | call_android_avdmanager -s create avd -n "${vd_name}" -k "system-images;android-34;google_apis;${android_sdk_host_arch}" -b "${android_sdk_host_arch}" -f

    vd_name="devlib-chromeos"
    echo "Creating virtual device \"${vd_name}\" (ChromeOS - Android 13, Pixel tablet)..."
    echo no | call_android_avdmanager -s create avd -n "${vd_name}" -k "system-images;android-33;android-desktop;${android_sdk_host_arch}" -b "${android_sdk_host_arch}" -f
}

install_apt() {
    echo "Installing apt packages..."
    local apt_cmd=(DEBIAN_FRONTEND=noninteractive apt-get)
    sudo "${apt_cmd[@]}" update
    if [[ $unsupported_distro == 1 ]]; then
        for package in "${apt_packages[@]}"; do
            if ! sudo "${apt_cmd[@]}" install -y "$package"; then
                echo "Failed to install $package on that distribution" >&2
            fi
        done
    else
        sudo "${apt_cmd[@]}" install -y "${apt_packages[@]}" || exit $?
    fi
}

install_pacman() {
    echo "Installing pacman packages..."
    sudo pacman -Sy --needed --noconfirm "${pacman_packages[@]}" || exit $?
}

# APT-based distributions like Ubuntu or Debian
apt_packages=(
    libarchive-tools
    qemu-user-static
    wget
)

# pacman-based distributions like Archlinux or its derivatives
pacman_packages=(
    coreutils
    libarchive
    qemu-user-static
    wget
)

# Detection based on the package-manager, so that it works on derivatives of
# distributions we expect. Matching on distro name would prevent that.
if which apt-get &>/dev/null; then
    install_functions+=(install_apt)
    package_manager='apt-get'
    expected_distro="Ubuntu"
elif which pacman &>/dev/null; then
    install_functions+=(install_pacman)
    package_manager="pacman"
    expected_distro="Arch Linux"
else
    echo "The package manager of distribution $(read_os_release NAME) is not supported, will only install distro-agnostic code"
fi

if [[ -n "${package_manager}" ]] && ! test_os_release NAME "${expected_distro}"; then
    unsupported_distro=1
    echo -e "\nINFO: the distribution seems based on ${package_manager} but is not ${expected_distro}, some package names might not be right\n"
else
    unsupported_distro=0
fi

usage() {
    echo "Usage: ${0} [--help] [--cleanup-android-sdk] [--install-android-tools]
    [--install-android-platform-tools] [--create-avds] [--install-all]"
    cat << EOF

Install distribution packages and other bits required by Android emulator.
Archlinux and Ubuntu are supported, although derivative distributions will
probably work as well.

--install-android-platform-tools is not needed when using
--install-android-tools, but has the advantage of not needing a Java
installation and is quicker to install.
EOF
}

# Defaults to --install-all if no option is given
if [[ -z "$*" ]]; then
    args=("--install-all")
else
    args=("$@")
fi

# Use conditional fall-through ;;& to all matching all branches with
# --install-all
for arg in "${args[@]}"; do
    # We need this flag since *) does not play well with fall-through ;;&
    handled=0
    case "$arg" in

    "--cleanup-android-sdk")
        install_functions+=(cleanup_android_home)
        handled=1
        ;;&

    # Not part of --install-all since that is already satisfied by
    # --install-android-tools The advantage of that method is that it does not
    # require the Java JDK/JRE to be installed, and is a bit quicker. However,
    # it will not provide the build-tools which are needed by devlib.
    "--install-android-platform-tools")
        install_functions+=(install_android_platform_tools)
        handled=1
        ;;&

    "--install-android-tools" | "--install-all")
        install_functions+=(
            find_java_home
            install_android_sdk_manager
            install_build_tools
            install_platform_tools
        )
        apt_packages+=(openjdk-"${ANDROID_SDK_JAVA_VERSION}"-jre openjdk-"${ANDROID_SDK_JAVA_VERSION}"-jdk)
        pacman_packages+=(jre"${ANDROID_SDK_JAVA_VERSION}"-openjdk jdk"${ANDROID_SDK_JAVA_VERSION}"-openjdk)
        handled=1
        ;;&

    "--create-avds" | "--install-all")
        install_functions+=(
            find_java_home
            install_android_sdk_manager
            install_platform_tools
            install_platforms
            install_system_images
            create_android_vds
        )
        handled=1
        ;;&

    "--help")
        usage
        exit 0
        ;;&

    *)
        if [[ ${handled} != 1 ]]; then
            echo "Unrecognised argument: ${arg}"
            usage
            exit 2
        fi
        ;;
    esac
done

# In order in which they will be executed if specified in command line
ordered_functions=(
    # Distro package managers before anything else, so all the basic
    # pre-requisites are there
    install_apt
    install_pacman

    find_java_home
    # cleanup must be done BEFORE installing
    cleanup_android_home
    install_android_sdk_manager
    install_android_platform_tools
    install_build_tools
    install_platform_tools
    install_platforms
    install_system_images
    create_android_vds
)

# Remove duplicates in the list
# shellcheck disable=SC2207
install_functions=($(echo "${install_functions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

mkdir -p "${ANDROID_HOME}/cmdline-tools"

# Call all the hooks in the order of available_functions
ret=0
for f in "${ordered_functions[@]}"; do
    for func in "${install_functions[@]}"; do
        if [[ ${func} == "${f}" ]]; then
            # If one hook returns non-zero, we keep going but return an overall failure code.
            ${func}; _ret=$?
            if [[ $_ret != 0 ]]; then
                ret=${_ret}
                echo "Stage ${func} failed with exit code ${ret}" >&2
            else
                echo "Stage ${func} succeeded" >&2
            fi
        fi
    done
done

exit $ret

# vim: set tabstop=4 shiftwidth=4 textwidth=80 expandtab:
