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
#
# Forked from LISA/tools/lisa-buildroot-create-rootfs.
#

set -eu

ARCH="aarch64"
BUILDROOT_URI="git://git.busybox.net/buildroot"
KERNEL_IMAGE_NAME="Image"

function print_usage
{
	echo "Usage: ${0} [options]"
	echo "	options:"
	echo "		-a: set arch (default is aarch64, x86_64 is also supported)"
	echo "		-p: purge buildroot to force a fresh build"
	echo "		-h: print this help message"
}

function set_arch
{
	if [[ "${1}" == "aarch64" ]]; then
		return 0
	elif [[ "${1}" == "x86_64" ]]; then
		ARCH="x86_64"
		KERNEL_IMAGE_NAME="bzImage"
		return 0
	fi

	return 1
}

while getopts "ahp" opt; do
	case ${opt} in
	a)
		shift
		if ! set_arch "${1}"; then
			echo "Invalid arch \"${1}\"."
			exit 1
		fi
		;;
	p)
		rm -rf "${BUILDROOT_DIR}"
		exit 0
		;;
	h)
		print_usage
		exit 0
		;;
	*)
		print_usage
		exit 1
		;;
	esac
done

# Execute function for once
function do_once
{
	FILE="${BUILDROOT_DIR}/.devlib_${1}"
	if [ ! -e "${FILE}" ]; then
		eval "${1}"
		touch "${FILE}"
	fi
}

function br_clone
{
	git clone -b ${BUILDROOT_VERSION} -v ${BUILDROOT_URI} "${BUILDROOT_DIR}"
}

function br_apply_config
{
	pushd "${BUILDROOT_DIR}" >/dev/null

	mkdir -p "board/arm-power/${ARCH}/"
	cp -f "../configs/post-build.sh" "board/arm-power/"
	cp -f "../configs/${ARCH}/arm-power_${ARCH}_defconfig" "configs/"
	cp -f "../configs/${ARCH}/linux.config" "board/arm-power/${ARCH}/"

	make "arm-power_${ARCH}_defconfig"

	popd >/dev/null
}

function br_build
{
	pushd "${BUILDROOT_DIR}" >/dev/null
	make
	popd >/dev/null
}


BUILDROOT_VERSION=${BUILDROOT_VERSION:-"2023.11.1"}
BUILDROOT_DIR="$(dirname "$0")/buildroot-v${BUILDROOT_VERSION}-${ARCH}"

do_once br_clone

do_once br_apply_config

br_build

echo "Kernel image \"${BUILDROOT_DIR}/output/images/${KERNEL_IMAGE_NAME}\" is ready."
