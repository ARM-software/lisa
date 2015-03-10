LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_SRC_FILES:= bbench_server.cpp
LOCAL_MODULE := bbench_server
LOCAL_MODULE_TAGS := optional
LOCAL_STATIC_LIBRARIES := libc
LOCAL_SHARED_LIBRARIES := 
include $(BUILD_EXECUTABLE)
