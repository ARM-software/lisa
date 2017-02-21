LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_SRC_FILES:= dhrystone.c 
LOCAL_MODULE := dhrystone
LOCAL_MODULE_TAGS := optional
LOCAL_STATIC_LIBRARIES := libc
LOCAL_SHARED_LIBRARIES := liblog
LOCAL_LDLIBS := -llog
LOCAL_CFLAGS := -O2
include $(BUILD_EXECUTABLE)
