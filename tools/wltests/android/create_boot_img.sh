#!/bin/bash

################################################################################
# Internal configurations
################################################################################
SCRIPT_DIR=$(dirname $(realpath -s $0))
BASE_DIR="$SCRIPT_DIR/.."
source "${BASE_DIR}/helpers"
source "${DEFINITIONS_PATH}"

DEFAULT_KERNEL="${KERNEL_SRC}/arch/${ARCH}/boot/${KERNEL_IMAGE}"
KERNEL="${KERNEL:-$DEFAULT_KERNEL}"

# Don't append a path if no ramdisk image is provided
[ -z ${RAMDISK_IMAGE} ] || DEFAULT_RAMDISK="${PLATFORM_OVERLAY_PATH}/${RAMDISK_IMAGE}"
RAMDISK="${RAMDISK:-$DEFAULT_RAMDISK}"

DEFAULT_BOOT_IMAGE="${ARTIFACTS_PATH}/${ANDROID_BOOT_IMAGE}"
BOOT_IMAGE="${BOOT_IMAGE:-$DEFAULT_BOOT_IMAGE}"

CMDLINE=${CMDLINE:-$KERNEL_CMDLINE}

if [ ! -f ${KERNEL} ] ; then
	c_error "KERNEL image not found: ${KERNEL}"
	exit $ENOENT
fi
if [ ! -z ${RAMDISK} ] && [ ! -f ${RAMDISK} ] ; then
	c_error "RAMDISK image not found: ${RAMDISK}"
	c_warning "A valid ramdisk image, which matches the device user-space"
	c_warning "must be deployed by the user under the required path."
	c_info "Please refer to the ISTALLATION INSTRUCTIONS"
	c_info "if you don'tknow how to provide such an image."
	echo
	exit $ENOENT
fi

################################################################################
# Report configuration
################################################################################
echo
c_info "Generate BOOT image:"
c_info "   $BOOT_IMAGE"
c_info "using this configuration :"
c_info "  KERNEL                 : $KERNEL"

if [ -z "{$RAMDISK}" ]; then
c_warning "  No RAMDISK provided, building image without ramdisk"
else
c_info "  RAMDISK                : $RAMDISK"
fi

c_info "  CMDLINE                : $CMDLINE"
c_info "  ANDROID_IMAGE_BASE     : $ANDROID_IMAGE_BASE"
c_info "  ANDROID_IMAGE_PAGESIZE : $ANDROID_IMAGE_PAGESIZE"
c_info "  ANDROID_OS_VERSION     : $ANDROID_OS_VERSION"
c_info "  ANDROID_OS_PATCH_LEVEL : $ANDROID_OS_PATCH_LEVEL"

# Optional arguments
if [ "${ANDROID_TAGS_OFFSET}" ]; then
    c_info "- ANDROID_TAGS_OFFSET    : ${ANDROID_TAGS_OFFSET}"
    ANDROID_TAGS_OFFSET="--tags_offset ${ANDROID_TAGS_OFFSET}"
fi

if [ "${ANDROID_KERNEL_OFFSET}" ]; then
    c_info "- ANDROID_KERNEL_OFFSET  : ${ANDROID_KERNEL_OFFSET}"
    ANDROID_KERNEL_OFFSET="--kernel_offset ${ANDROID_KERNEL_OFFSET}"
fi

if [ "${ANDROID_RAMDISK_OFFSET}" ]; then
    c_info "- ANDROID_RAMDISK_OFFSET : ${ANDROID_RAMDISK_OFFSET}"
    ANDROID_RAMDISK_OFFSET="--ramdisk_offset ${ANDROID_RAMDISK_OFFSET}"
fi

if [ ! -z "${RAMDISK}" ]; then
    RAMDISK_CMD="--ramdisk \"${RAMDISK}\""
fi

################################################################################
# Generate BOOT image
################################################################################

# Ensure the output folder exists
mkdir -p $(dirname $BOOT_IMAGE) &>/dev/null

set -x
"${ANDROID_SCRIPTS_PATH}/mkbootimg" \
	--kernel "${KERNEL}" \
	$RAMDISK_CMD \
	--cmdline "${CMDLINE}" \
	--base "${ANDROID_IMAGE_BASE}" \
	--pagesize "${ANDROID_IMAGE_PAGESIZE}" \
	--os_version "${ANDROID_OS_VERSION}" \
	--os_patch_level "${ANDROID_OS_PATCH_LEVEL}" \
	${ANDROID_TAGS_OFFSET} \
	${ANDROID_KERNEL_OFFSET} \
	${ANDROID_RAMDISK_OFFSET} \
	--output "${BOOT_IMAGE}"
set +x
