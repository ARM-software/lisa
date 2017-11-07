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

DEFAULT_DTB="${KERNEL_SRC}/arch/${ARCH}/boot/dts/${KERNEL_DTB}"
DTB="${DTB:-$DEFAULT_DTB}"

DEFAULT_RAMDISK="${PLATFORM_OVERLAY_PATH}/${RAMDISK_IMAGE}"
RAMDISK="${RAMDISK:-$DEFAULT_RAMDISK}"

DEFAULT_BOOT_IMAGE="${ARTIFACTS_PATH}/${ANDROID_BOOT_IMAGE}"
BOOT_IMAGE="${BOOT_IMAGE:-$DEFAULT_BOOT_IMAGE}"

CMDLINE=${CMDLINE:-$KERNEL_CMDLINE}

if [ ! -f ${KERNEL} ] ; then
	c_error "KERNEL image not found: ${KERNEL}"
	exit $ENOENT
fi

if [ ! -f ${DTB} ] ; then
	c_error "DTB not found: ${DTB}"
	exit $ENOENT
fi

if [ ! -f ${RAMDISK} ] ; then
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
c_info "  RAMDISK                : $RAMDISK"
c_info "  CMDLINE                : $CMDLINE"
c_info "  ANDROID_IMAGE_BASE     : $ANDROID_IMAGE_BASE"
c_info "  ANDROID_IMAGE_PAGESIZE : $ANDROID_IMAGE_PAGESIZE"
c_info "  ANDROID_OS_VERSION     : $ANDROID_OS_VERSION"
c_info "  ANDROID_OS_PATCH_LEVEL : $ANDROID_OS_PATCH_LEVEL"

# Optional arguments
if [ "${ANDROID_TAGS_OFFSET}" ]; then
    c_info "- ANDROID_TAGS_OFFSET    : ${ANDROID_TAGS_OFFSET}"
    ANDROID_TAGS_OFFSET="tagsaddr = ${ANDROID_TAGS_OFFSET}"
fi

if [ "${ANDROID_KERNEL_OFFSET}" ]; then
    c_info "- ANDROID_KERNEL_OFFSET  : ${ANDROID_KERNEL_OFFSET}"
    ANDROID_KERNEL_OFFSET="kerneladdr = ${ANDROID_KERNEL_OFFSET}"
fi

if [ "${ANDROID_BOOT_SIZE}" ]; then
    c_info "- ANDROID_BOOT_SIZE: ${ANDROID_BOOT_SIZE}"
    ANDROID_BOOT_SIZE="bootsize = ${ANDROID_BOOT_SIZE}"
fi

if [ "${ANDROID_PAGE_SIZE}" ]; then
    c_info "- ANDROID_PAGE_SIZE: ${ANDROID_PAGE_SIZE}"
    ANDROID_PAGE_SIZE="pagesize = ${ANDROID_PAGE_SIZE}"
fi

if [ "${ANDROID_SECOND_SIZE}" ]; then
    c_info "- ANDROID_SECOND_SIZE: ${ANDROID_SECOND_SIZE}"
    ANDROID_SECOND_SIZE="secondaddr = ${ANDROID_SECOND_SIZE}"
fi

if [ "${ANDROID_RAMDISK_OFFSET}" ]; then
    c_info "- ANDROID_RAMDISK_OFFSET : ${ANDROID_RAMDISK_OFFSET}"
    ANDROID_RAMDISK_OFFSET="ramdiskaddr = ${ANDROID_RAMDISK_OFFSET}"
fi

cat > "${ARTIFACTS_PATH}"/bootimg-960.cfg << EOL
${ANDROID_BOOT_SIZE}
${ANDROID_PAGE_SIZE}
${ANDROID_KERNEL_OFFSET}
${ANDROID_RAMDISK_OFFSET}
${ANDROID_SECOND_SIZE}
${ANDROID_TAGS_OFFSET}
name =
cmdline = ${CMDLINE}
EOL

################################################################################
# Generate BOOT image
################################################################################

# Ensure the output folder exists
mkdir -p $(dirname $BOOT_IMAGE) &>/dev/null

set -x
c_info "- IMAGE_DTB: $IMAGE_DTB"
IMAGE_DTB="$(dirname $KERNEL)/Image-dtb"
cat "${KERNEL}" "${DTB}" > "${IMAGE_DTB}"
abootimg --create "${BOOT_IMAGE}" -k "${IMAGE_DTB}" -r "${RAMDISK}" -f "${ARTIFACTS_PATH}"/bootimg-960.cfg
set +x
