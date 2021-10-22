# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, ARM Limited and contributors.
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

ifneq ($(KERNELRELEASE),)
	MODULE_SRC ?= $(src)
	MODULE_OBJ := $(obj)
	KERNEL_SRC ?= $(srctree)
else
	MODULE_SRC ?= $(PWD)
	MODULE_OBJ := $(PWD)
	KERNEL_SRC ?= /lib/modules/`uname -r`/build
endif

VMLINUX_H = $(MODULE_OBJ)/vmlinux.h

# kbuild part of makefile. Only Kbuild-related targets should be used here to
# avoid any sort of clash.
ifneq ($(KERNELRELEASE),)

LISA_KMOD_NAME ?= sched_tp
obj-m := $(LISA_KMOD_NAME).o

# Since the module name is determined by the object file name, we need a
# matching source name. Unfortunately, there does not seem to be a more
# straightforward way of setting the module's name ...
MAIN_C_SRC := $(MODULE_SRC)/$(LISA_KMOD_NAME).c
$(MAIN_C_SRC): $(MODULE_SRC)/sched_tp.c
	rm $@ 2> /dev/null || true
	ln -s sched_tp.c $@

# -fno-stack-protector is needed to possibly undefined __stack_chk_guard symbol
ccflags-y = -I$(MODULE_SRC) -fno-stack-protector

VMLINUX_TXT = $(MODULE_SRC)/private_types.txt

# Can be either a kernel image built with DWARF debug info, or the BTF blob
# found at /sys/kernel/btf/vmlinux
_BTF_VMLINUX = $(MODULE_SRC)/vmlinux
_DWARF_VMLINUX = $(KERNEL_SRC)/vmlinux
ifneq ("$(wildcard $(_BTF_VMLINUX))","")
    VMLINUX := $(_BTF_VMLINUX)
else
    VMLINUX := $(_DWARF_VMLINUX)
endif

$(VMLINUX_H): $(VMLINUX_TXT) $(VMLINUX)
# --inner_anonymous and --skip_missing are not upstream yet, but they have been
# --submitted on the mailing list
	pahole --inner_anonymous -E --skip_missing -C file://$(VMLINUX_TXT) $(VMLINUX) > _header
# Create forward declaration of every type
	sed -r -n 's/.*(struct|union|enum) ([0-9a-zA-Z_]*) .*/\1 \2;/p' _header | sort -u > _fwd_decl
# Create TYPED_DEFINED_struct_foo macros for every type, so the client code can
# check whether a given type exists before making use of it
	sed -r -n 's/.*(struct|union|enum) ([0-9a-zA-Z_]*) .*/#define TYPE_DEFINED_\1_\2/p' _header | sort -u >> _fwd_decl
	cat _fwd_decl _header > $@
# cat $@

$(MODULE_OBJ)/$(LISA_KMOD_NAME).o: $(VMLINUX_H)

# Non-Kbuild part
else

.PHONY: all build install clean

all: install

build:
	$(MAKE) -C $(KERNEL_SRC) M=$(MODULE_SRC) modules

install: build
	$(MAKE) -C $(KERNEL_SRC) M=$(MODULE_SRC) modules_install

clean:
	rm -f $(VMLINUX_H)

endif
