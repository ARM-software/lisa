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

# Script to install the depenencies for LISA on an Ubuntu-like system.

# This is intended to be used for setting up containers and virtual machines to
# run LISA (e.g. for CI infrastructure or for docker installation).
# This can also work for a fresh LISA install on a workstation.

# Read standard /etc/os-release file and extract the needed field lsb_release
# binary is not installed on all distro, but that file is found pretty much
# anywhere.
read_os_release() {
    local field_name=$1
    (source /etc/os-release &> /dev/null && printf "%s" "${!field_name}")
}

# Test the value of a field in /etc/os-release
test_os_release(){
    local field_name=$1
    local value=$2

    if [[ "$(read_os_release "$field_name")" == "$value" ]]; then
        # same as "true" command
        return 0
    else
        # same as "false" commnad
        return 1
    fi
}

lower_or_equal() {
    local x
    local y

    x=$(printf "%s" "$1" | sed 's/\.//2g')
    y=$(printf "%s" "$2" | sed 's/\.//2g')
    [[ "$(printf "%s\n%s\n" "$x" "$y" | sort -g | head -1)" == "$x" ]]
}

sudo() {
    # Run without sudo if we are already root
    if [[ $EUID = 0 ]]; then
        env - "$@"
    else
        command sudo "$@"
    fi
}

# Make it available in subshells
export -f sudo

LISA_HOME=${LISA_HOME:-$(dirname "${BASH_SOURCE[0]}")}
cd "$LISA_HOME" || (echo "LISA_HOME ($LISA_HOME) does not exists" && exit 1)

# Must be kept in sync with shell/lisa_shell
export ANDROID_HOME="$LISA_HOME/tools/android-sdk-linux/"

install_apt() {
    echo "Installing apt packages ..."
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
    echo "Installing pacman packages ..."
    sudo pacman -Sy --needed --noconfirm "${pacman_packages[@]}" || exit $?
}

devlib_setup_host() {
    if [ ${#devlib_params[@]} -ne 0 ]; then
        "${LISA_HOME}/external/devlib/tools/android/setup_host.sh" "${devlib_params[@]}"
    fi
}

# Extra Python pip requirements, to be installed by lisa-install
pip_extra_requirements=()

# APT-based distributions like Ubuntu or Debian
apt_packages=(
    build-essential
    coreutils
    git
    kernelshark
    openssh-client
    python3
    # venv is not installed by default on Ubuntu, even though it is part of the
    # Python standard library
    python3-pip
    python3-venv
    python3-tk
    python3-setuptools
    qemu-user-static
)

# pacman-based distributions like Archlinux or its derivatives
pacman_packages=(
    base-devel
    coreutils
    git
    kernelshark
    openssh
    python
    python-pip
    python-setuptools
    qemu-user-static
)

HOST_ARCH="$(uname -m)"

# ABI-specific packages
case $HOST_ARCH in
    aarch64)
        # Allows building C extensions from sources, when they do not ship a
        # prebuilt wheel for that arch
        apt_packages+=(python3-dev)
        # pacman_packages+=()
        ;;
esac


# More recent versions of Ubuntu ship firefox as a snap package already
# containing the geckodriver
if test_os_release NAME "Ubuntu" && lower_or_equal "$(read_os_release VERSION_ID)" "22"; then
    # In order to save plots using bokeh, we need a browser usable with
    # selenium, along with its selenium driver.
    #
    # Note: firefox seems to cope well without X11, unlike chromium.
    apt_packages+=(firefox-geckodriver)
fi

# Array of functions to call in order
install_functions=()

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

if [[ -n "$package_manager" ]] && ! test_os_release NAME "$expected_distro"; then
    unsupported_distro=1
    echo -e "\nINFO: the distribution seems based on $package_manager but is not $expected_distro, some package names might not be right\n"
else
    unsupported_distro=0
fi

usage() {
    echo "Usage: $0 [--help] [--cleanup-android-sdk] [--install-android-tools]
    [--install-android-platform-tools] [--install-doc-extras]
    [--install-tests-extra] [--install-toolchains] [--install-all]"
    cat << EOF

Install distribution packages and other bits that don't fit in the Python
venv managed by lisa-install. Archlinux and Ubuntu are supported, although
derivative distributions will probably work as well.

--install-android-platform-tools is not needed when using --install-android-tools,
but has the advantage of not needing a Java installation and is quicker to install.
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
devlib_params=()
for arg in "${args[@]}"; do
    # We need this flag since *) does not play well with fall-through ;;&
    handled=0
    case "$arg" in

    "--cleanup-android-sdk" | "--install-android-platform-tools" )
        devlib_params+=(${arg})
        install_functions+=(devlib_setup_host)
        handled=1
        ;;&

    "--install-android-tools" | "--install-all" )
        devlib_params+=("--install-android-tools")
        install_functions+=(devlib_setup_host)
        handled=1
        ;;&

    "--install-doc-extras" | "--install-all")
        apt_packages+=(plantuml graphviz pandoc)
        # plantuml can be installed from the AUR
        pacman_packages+=(graphviz pandoc)
        handled=1
        ;;&

    # Requirement for LISA's self tests (in tests/ folder), not the synthetic
    # kernel tests.
    "--install-tests-extras" | "--install-all")
        apt_packages+=(clang)
        pacman_packages+=(clang)
        handled=1
        ;;&

    "--install-build-env" | "--install-all")
        apt_packages+=(fuse-overlayfs)
        pacman_packages+=(fuse-overlayfs)
        handled=1
        ;;&

    "--install-toolchains" | "--install-build-env" | "--install-all")
        apt_packages+=(build-essential gcc-arm-linux-gnueabi gcc-aarch64-linux-gnu)
        # arm-linux-gnueabihf-gcc can be installed from the AUR
        pacman_packages+=(base-devel aarch64-linux-gnu-gcc flex)

        # Build dependencies of some assets
        apt_packages+=(autopoint autoconf libtool bison flex cmake)
        # gettext for autopoint
        pacman_packages+=(gettext autoconf libtool bison cmake)

        handled=1
        ;;&

    "--install-kernel-build-dependencies" | "--install-build-env" | "--install-all")
        apt_packages+=(build-essential gcc bc bison flex libssl-dev libncurses5-dev libelf-dev)

        if test_os_release NAME "Ubuntu"; then
            case $(read_os_release VERSION_ID) in
                # Ubuntu Jammy 22.04 does ship a pahole package
                "18.04" | "18.10" | "19.04" | "19.10" | "20.04" | "20.10" | "21.04" | "21.10" | "22.10" | "23.04" | "23.10")
                    apt_packages+=(dwarves)
                    ;;
                *)
                    apt_packages+=(pahole)
                    ;;
            esac
        fi

        handled=1
        ;;&

    "--help")
        usage
        exit 0
        ;;&

    *)
        if [[ $handled != 1 ]]; then
            echo "Unrecognised argument: $arg"
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
    devlib_setup_host
)

# Remove duplicates in the list
# shellcheck disable=SC2207
install_functions=($(echo "${install_functions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Call all the hooks in the order of available_functions
ret=0
for _func in "${ordered_functions[@]}"; do
    for func in "${install_functions[@]}"; do
        if [[ $func == "$_func" ]]; then
            # If one hook returns non-zero, we keep going but return an overall failure code.
            $func; _ret=$?
            if [[ $_ret != 0 ]]; then
                ret=$_ret
                echo "Stage $func failed with exit code $ret" >&2
            else
                echo "Stage $func succeeded" >&2
            fi
        fi
    done
done
exit ${ret}

# vim: set tabstop=4 shiftwidth=4 textwidth=80 expandtab:
