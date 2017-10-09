#!/bin/bash

################################################################################
# Internal configurations
################################################################################
SCRIPT_DIR=$(dirname $(realpath -s $0))
BASE_DIR="$SCRIPT_DIR/.."
source "${BASE_DIR}/helpers"
source "${DEFINITIONS_PATH}"

DEFAULT_DTB="${KERNEL_SRC}/arch/${ARCH}/boot/dts/${KERNEL_DTB}"
DTB="${DTB:-$DEFAULT_DTB}"

DEFAULT_DTB_IMAGE="${ARTIFACTS_PATH}/${ANDROID_DTB_IMAGE}"
DTB_IMAGE="${DTB_IMAGE:-$DEFAULT_DTB_IMAGE}"

if [ ! -f ${DTB} ] ; then
	c_error "DTB not found: ${DTB}"
	exit $ENOENT
fi

################################################################################
# Report configuration
################################################################################
echo
c_info "Generate DTB image:"
c_info "   $DTB_IMAGE"
c_info "using this configuration :"
c_info "  DTB                    : $DTB"
c_info "  ANDROID_IMAGE_PAGESIZE : $ANDROID_IMAGE_PAGESIZE"

# Optional arguments
if [ "x${ANDROID_DTB_COMPRESSED}"=="xYES" ]; then
    c_info "- ANDROID_DTB_COMPRESSED : $ANDROID_DTB_COMPRESSED"
    ANDROID_DTB_COMPRESSED="--compress"
fi

################################################################################
# Generate BOOT image
################################################################################

# Ensure the output folder exists
mkdir -p $(dirname $DTB_IMAGE) &>/dev/null

set -x
"${ANDROID_SCRIPTS_PATH}"/mkdtimg \
    --dtb "${DTB}" \
    --pagesize "${ANDROID_IMAGE_PAGESIZE}" \
    $ANDROID_DTB_COMPRESSED \
    --output "${DTB_IMAGE}"
set +x

