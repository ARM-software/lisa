#! /usr/bin/env bash

# Script to install the depenencies for LISA on an Ubuntu-like system.

# This is intended to be used for setting up containers and virtual machines to
# run LISA (e.g. for CI infrastructure or for Vagrant installation).
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

LISA_HOME=${LISA_HOME:-$(dirname "${BASH_SOURCE[0]}")}
cd "$LISA_HOME" || (echo "LISA_HOME ($LISA_HOME) does not exists" && exit 1)

# Must be kept in sync with shell/lisa_shell
ANDROID_HOME="$LISA_HOME/tools/android-sdk-linux/"
ANDROID_SDK_ROOT="$ANDROID_HOME"
mkdir -p "$ANDROID_HOME"

# No need for the whole SDK for this one
install_android_platform_tools() {
    echo "Installing Android platform tools ..."

    local url="https://dl.google.com/android/repository/platform-tools-latest-linux.zip"
    local archive="$ANDROID_HOME/android-platform-tools.zip"

    wget --no-verbose "$url" -O "$archive" &&
    echo "Extracting $archive ..." &&
    unzip -q -o "$archive" -d "$ANDROID_HOME"
}

cleanup_android_home() {
    echo "Cleaning up Android SDK: $ANDROID_HOME"
    rm -r "$ANDROID_HOME"
    mkdir -p "$ANDROID_HOME"
}

install_android_sdk_manager() {
    echo "Installing Android SDK manager ..."

    # URL taken from "Command line tools only": https://developer.android.com/studio
    # Used to be: https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip
    local url="https://dl.google.com/android/repository/commandlinetools-linux-6858069_latest.zip"
    local archive="$ANDROID_HOME/android-sdk-manager.zip"
    rm "$archive" &>/dev/null

    echo "Downloading Android SDK manager from: $url"
    wget --no-verbose "$url" -O "$archive" &&
    echo "Extracting $archive ..." &&
    unzip -q -o "$archive" -d "$ANDROID_HOME"

    yes | (call_android_sdkmanager --licenses || true)
    call_android_sdkmanager --list
}

# Android SDK is picky on Java version, so we need to set JAVA_HOME manually.
# In most distributions, Java is installed under /usr/lib/jvm so use that.
# according to the distribution
ANDROID_SDK_JAVA_VERSION=11
find_java_home() {
    _JAVA_BIN=$(find /usr/lib/jvm -path "*$ANDROID_SDK_JAVA_VERSION*/bin/java" -not -path '*/jre/bin/*' -print -quit)
    _JAVA_HOME=$(dirname "$_JAVA_BIN")/../

    echo "Found JAVA_HOME=$_JAVA_HOME"
}

call_android_sdk() {
    # Used to be:
    # local tool="$ANDROID_HOME/tools/bin/$1"
    local tool="$ANDROID_HOME/cmdline-tools/bin/$1"
    shift
    echo "Using JAVA_HOME=$_JAVA_HOME for Android SDK" >&2
    # Use grep to remove the progress bar, as there is no CLI option for the SDK
    # manager to do that
    JAVA_HOME=$_JAVA_HOME "$tool" "$@" | grep -v '\[='
}

call_android_sdkmanager() {
    call_android_sdk sdkmanager --sdk_root="$ANDROID_SDK_ROOT" "$@"
}

# Needs install_android_sdk_manager first
install_android_tools() {
    # We could use install_android_platform_tools here for platform-tools if the
    # SDK starts being annoying
    # Note: recent sdkmanager seem to be installing "platform-tools" by default,
    # so it's not necessary anymore to specify it on the command line
    yes | call_android_sdkmanager --verbose --channel=0 --install "build-tools;30.0.3"
}

# Install nodejs from snap packages instead of the main package manager
install_nodejs_snap() {
    # sanity check to make sure snap is up and running
    if ! which snap >/dev/null 2>&1; then
        echo 'Snap not installed on that system, not installing nodejs'
        return 1
    elif ! snap list >/dev/null 2>&1; then
        echo 'Snap not usable on that system, not installing nodejs'
        return 1
    else
        echo "Installing snap nodejs package ..."
        # Latest LTS version available on snap:
        # https://nodejs.org/fr/download/
        # https://snapcraft.io/node
        NODEJS_VERSION=12
        sudo snap install node --classic --channel=$NODEJS_VERSION
    fi
}

install_apt() {
    echo "Installing apt packages ..."
    sudo apt-get update &&
    sudo apt-get install -y "${apt_packages[@]}"
}

install_pacman() {
    echo "Installing pacman packages ..."
    sudo pacman -Sy --needed --noconfirm "${pacman_packages[@]}"
}

register_pip_extra_requirements() {
    local requirements="$LISA_HOME/extra_requirements.txt"
    local devmode_requirements="$LISA_HOME/devmode_extra_requirements.txt"

    echo "Registering extra Python pip requirements in $requirements:"
    local content=$(printf "%s\n" "${pip_extra_requirements[@]}")
    printf "%s\n\n" "$content" | tee "$requirements"

    # All the requirements containing "./" are prefixed with "-e " to install
    # them in editable mode
    printf "%s\n\n" "$(printf "%s" "$content" | sed '/.\//s/^/-e /')" > "$devmode_requirements"
}

# Extra Python pip requirements, to be installed by lisa-install
pip_extra_requirements=()

# APT-based distributions like Ubuntu or Debian
apt_packages=(
    coreutils
    build-essential
    git
    openssh-client
    sshpass
    wget
    unzip
    kernelshark
    python3
    python3-dev
    python3-pip
    # venv is not installed by default on Ubuntu, even though it is part of the
    # Python standard library
    python3-venv
    python3-setuptools
    python3-tk
)

# pacman-based distributions like Archlinux or its derivatives
pacman_packages=(
    coreutils
    git
    openssh
    sshpass
    base-devel
    wget
    unzip
    python
    python-pip
    python-setuptools

    # These packages can be installed from AUR
    # kernelshark
)

# ABI-specific packages
case $(uname -m) in
    aarch64)
        # Since scipy does not ship a prebuilt wheel on PyPI for arm64, it needs
        # to be built from source. That requires BLAS to be available, as well
        # as a fortran compiler.
        apt_packages+=(libblas-dev liblapack-dev gfortran libfreetype6)
        # This will not be enough as of August 2019, since these packages do not
        # include the dev headers, although they should according to the general
        # Archlinux policy.
        pacman_packages+=(lapack blas gcc-fortran freetype2)


        # For pyarrow
        apt_packages+=(cmake)
        pacman_packages+=(cmake)
        ;;
esac

# Array of functions to call in order
install_functions=(
    register_pip_extra_requirements
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

if [[ ! -z "$package_manager" ]] && ! test_os_release NAME "$expected_distro"; then
    echo
    echo "INFO: the distribution seems based on $package_manager but is not $expected_distro, some package names might not be right"
    echo
fi

usage() {
    echo "Usage: $0 [--help] [--cleanup-android-sdk] [--install-android-tools] [--install-android-platform-tools] [--install-doc-extras] [--install-nodejs] [--install-bisector-dbus] [--install-toolchains] [--install-vagrant] [--install-all]"
    cat << EOF

Install distribution packages and other bits that don't fit in the Python
venv managed by lisa-install. Archlinux and Ubuntu are supported, although
derivative distributions will probably work as well.

--install-android-platform-tools is not needed when using --install-android-tools,
but has the advantage of not needing a Java installation and is quicker to install.
EOF
}

# Use conditional fall-through ;;& to all matching all branches with
# --install-all
for arg in "$@"; do
    # We need this flag since *) does not play well with fall-through ;;&
    handled=0
    case "$arg" in

    "--cleanup-android-sdk")
        install_functions+=(cleanup_android_home)
        handled=1
        ;;&

    # TODO: remove --install-android-sdk, since it is only temporarily there to
    # give some time to migrate CI scripts
    "--install-android-sdk" | "--install-android-tools" | "--install-all")
        install_functions+=(
            find_java_home
            install_android_sdk_manager # Needed by install_android_build_tools
            install_android_tools
        )
        apt_packages+=(openjdk-$ANDROID_SDK_JAVA_VERSION-jre openjdk-$ANDROID_SDK_JAVA_VERSION-jdk)
        pacman_packages+=(jre$ANDROID_SDK_JAVA_VERSION-openjdk jdk$ANDROID_SDK_JAVA_VERSION-openjdk)
        handled=1;
        ;;&

    # Not part of --install-all since that is already satisfied by
    # --install-android-tools The advantage of that method is that it does not
    # require the Java JDK/JRE to be installed, and is a bit quicker. However,
    # it will not provide the build-tools which are needed by devlib.
    "--install-android-platform-tools")
        install_functions+=(install_android_platform_tools)
        handled=1;
        ;;&

    "--install-doc-extras" | "--install-all")
        apt_packages+=(plantuml graphviz pandoc)
        # plantuml can be installed from the AUR
        pacman_packages+=(graphviz pandoc)
        handled=1;
        ;;&

    "--install-nodejs" | "--install-all")
        # NodeJS v8+ is required, Ubuntu 16.04 LTS supports only an older version.
        # As a special case we can install it as a snap package
        if (
               test_os_release NAME Ubuntu && (
                   test_os_release VERSION_ID 16.04 ||
                    # On Ubuntu Bionic 18.04 LTS the nodejs package is broken,
                    # so use the one from snap instead:
                    # https://bugs.launchpad.net/ubuntu/+source/nodejs/+bug/1794589
                   test_os_release VERSION_ID 18.04 ||
                   # On Ubuntu Focal 20.04 LTS, nodejs version is too old (10.19)
                   test_os_release VERSION_ID 20.04
                )
        ); then
            apt_packages+=(snapd)
            install_functions+=(install_nodejs_snap)
        else
            apt_packages+=(nodejs npm)
            pacman_packages+=(nodejs npm)
        fi
        handled=1;
        ;;&

    "--install-toolchains" | "--install-all")
        apt_packages+=(build-essential gcc-arm-linux-gnueabi gcc-aarch64-linux-gnu)
        # arm-linux-gnueabihf-gcc can be installed from the AUR
        pacman_packages+=(base-devel aarch64-linux-gnu-gcc flex)

        # Build dependencies of some assets
        apt_packages+=(autopoint autoconf libtool bison flex cmake)
        # gettext for autopoint
        pacman_packages+=(gettext autoconf libtool bison cmake)

        handled=1;
        ;;&

    "--install-vagrant" | "--install-all")
        apt_packages+=(vagrant virtualbox)
        pacman_packages+=(vagrant virtualbox virtualbox-host-dkms)

        handled=1;
        ;;&

    "--install-bisector-dbus" | "--install-all")
        apt_packages+=(
            gobject-introspection
            # Some of that seems to only be needed on some version of Ubuntu.
            # GTK/Glib does not shine on packaging side, so ere on the side of
            # caution and install all the things that seem to avoid issues ...
            libcairo2-dev
            libgirepository1.0-dev
            gir1.2-gtk-3.0
        )
        # plantuml can be installed from the AUR
        pacman_packages+=(gobject-introspection)
        pip_extra_requirements+=(./tools/bisector[dbus])
        handled=1;
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
    install_nodejs_snap
    install_pacman

    find_java_home
    # cleanup must be done BEFORE installing
    cleanup_android_home
    install_android_sdk_manager # Needed by install_android_build_tools
    install_android_tools
    install_android_platform_tools

    register_pip_extra_requirements
)

# Remove duplicates in the list
install_functions=($(echo "${install_functions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Call all the hooks in the order of available_functions
ret=0
for _func in "${ordered_functions[@]}"; do
    for func in "${install_functions[@]}"; do
        if [[ $func == $_func ]]; then
            # If one hook returns non-zero, we keep going but return an overall failure
            # code
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

exit $ret

# vim: set tabstop=4 shiftwidth=4 textwidth=80 expandtab:
