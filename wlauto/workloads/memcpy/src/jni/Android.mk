LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_SRC_FILES := memcopy.c

LOCAL_LD_LIBS := -lrt

LOCAL_MODULE := memcpy


include $(BUILD_EXECUTABLE)
