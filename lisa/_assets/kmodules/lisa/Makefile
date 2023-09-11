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

# This would break the IN_TREE_BUILD detection
ifeq ($(KBUILD_ABS_SRCTREE),1)
$(error KBUILD_ABS_SRCTREE=1 is not handled by the lisa module
endif

ifeq ($(srctree),.)
	IN_TREE_BUILD := 0
else
	IN_TREE_BUILD := 1
endif

# This did not lie:
# https://lisa-linux-integrated-system-analysis.readthedocs.io/en/latest/setup.html#manual-module-setup-warning2
# Maintaining a patch to get in-tree build will be the least of your worries
# before long.
# If you remove this check, you won't be able to claim you have not been warned,
# and you will forego any sort of support. The world is your oyster.
ifeq ($(IN_TREE_BUILD),1)
# There is a comment 2 lines above you should read
$(error obscure error)
# There is a comment 4 lines above you should read, seriously
endif
# There is a comment ... by now you know the drill.
#
# Note: asking for this snippet to be removed will ultimately result in pulling
# the plug on the whole in-tree build support. Remove it on your local tree and
# maintain the patch if you are motivated enough to deal with that. Pull
# requests to fix issues related to in-tree build are welcome, *provided that
# they were tested in normal flow as well*.

# Kbuild
ifneq ($(KERNELRELEASE),)
	MODULE_OBJ := $(obj)
	KERNEL_SRC ?= $(srctree)
ifeq ($(IN_TREE_BUILD),1)
	MODULE_SRC ?= $(srctree)/$(src)
# Kbuild out-of-tree build
else
	MODULE_SRC ?= $(src)
endif
# non-Kbuild
else
	MODULE_SRC ?= $(PWD)
	MODULE_OBJ := $(PWD)
	KERNEL_SRC ?= /lib/modules/`uname -r`/build
endif

# kbuild part of makefile. Only Kbuild-related targets should be used here to
# avoid any sort of clash.
ifneq ($(KERNELRELEASE),)

LISA_KMOD_NAME ?= lisa
obj-m := $(LISA_KMOD_NAME).o
$(LISA_KMOD_NAME)-y := main.o tp.o wq.o features.o pixel6.o introspection_data.o

# -fno-stack-protector is needed to possibly undefined __stack_chk_guard symbol
ccflags-y := "-I$(MODULE_SRC)" -std=gnu11 -fno-stack-protector -Wno-declaration-after-statement -Wno-error

# Ensure we get appropriate debug info so that pahole can work on the object
# file regardless of the kernel config.
CFLAGS_introspection_data.o := -g3 -gno-split-dwarf -gdwarf -fno-eliminate-unused-debug-types

FEATURES_LDS := features.lds
GENERATED := $(MODULE_OBJ)/generated

$(GENERATED):
	mkdir -p "$@"

SYMBOL_NAMESPACES_H := $(GENERATED)/symbol_namespaces.h
MODULE_VERSION_H := $(GENERATED)/module_version.h
KALLSYMS := $(GENERATED)/kallsyms

# in-tree build
ifeq ($(IN_TREE_BUILD),1)

ccflags-y += -I$(srctree) -D_IN_TREE_BUILD
ldflags-y += -T $(srctree)/$(obj)/$(FEATURES_LDS)

# out-of-tree build
else

INTROSPECTION_DATA_H := $(GENERATED)/introspection_data.h

ldflags-y += -T $(M)/$(FEATURES_LDS)

clean-files := $(GENERATED)

PRIVATE_TYPES_TXT := $(MODULE_SRC)/private_types.txt

# Can be either a kernel image built with DWARF debug info, or the BTF blob
# found at /sys/kernel/btf/vmlinux
_BTF_VMLINUX := $(MODULE_SRC)/vmlinux
_REAL_VMLINUX := $(KERNEL_SRC)/vmlinux
_SYNTHETIC_VMLINUX := $(MODULE_OBJ)/introspection_data.o

ifeq ("$(wildcard $(_BTF_VMLINUX))","")
ifeq ($(IN_TREE_BUILD),1)
    VMLINUX := $(_SYNTHETIC_VMLINUX)
else
    VMLINUX := $(_REAL_VMLINUX)
endif
else
    VMLINUX := $(_BTF_VMLINUX)
endif

UNUSED_KERNEL_PRIVATE_TYPES_PREFIX := KERNEL_PRIVATE_

VENV := $(MODULE_OBJ)/venv
$(VENV):
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && python3 -m pip install --upgrade pip setuptools wheel && python3 -m pip install -r $(MODULE_SRC)/requirements.txt

$(INTROSPECTION_DATA_H): $(GENERATED) $(KALLSYMS) $(PRIVATE_TYPES_TXT) $(VMLINUX) $(VENV)
# Some options are not upstream (yet) but they have all be published on the
# dwarves mailing list
#
# Options:
# -F btf,dwarf: Use BTF first
# -E: Expand nested type definitions
# --suppress_force_paddings: Remove the "artificial" padding members pahole adds
#   to make padding more visible. They are not always valid C syntax and can
#   break build
# --skip_missing: Keep going if one of the types is not found
# --expand_types_once (non upstream): Only expand a given type once, to avoid type redefinition
#   (C does not care about nesting types, there is a single namespace).
#
# We then post-process the header to add a prefix to each type expanded by -E
# that was not explicitly asked for. This avoids conflicting with type
# definitions that would come from public kernel headers, while still allowing
# easy attribute access.
	pahole -F btf,dwarf -E --compile --suppress_force_paddings --show_only_data_members --skip_missing --expand_types_once -C "file://$(PRIVATE_TYPES_TXT)" "$(VMLINUX)" > $(MODULE_OBJ)/_header

# Create a header with all the types we can for introspection purposes. This
# will not be included in any of the module sources, so all we care about is
# that pycparser can parse it. Specifically, some types may be re-defined with
# incompatible definitions. This is expected as various drivers use the same
# struct names.
	pahole -F btf,dwarf -E --expand_types_once --suppress_force_paddings --suppress_aligned_attribute --suppress_packed --show_only_data_members --compile --fixup_silly_bitfields "$(VMLINUX)" > $(MODULE_OBJ)/_full_header

# Strip comments to avoid matching them with the sed regex.
	"$(CC)" -P -E - < $(MODULE_OBJ)/_header > $(MODULE_OBJ)/_header_no_comment
# Create forward declaration of every type to ensure the header can be parsed correctly.
	sed -r -n 's/.*(struct|union|enum) ([0-9a-zA-Z_]*) .*/\1 \2;/p' $(MODULE_OBJ)/_header_no_comment | sort -u > $(MODULE_OBJ)/_fwd_decl

# Rename all the kernel private types we are not directly interested in to avoid any clash
	cat $(MODULE_OBJ)/_fwd_decl $(MODULE_OBJ)/_header_no_comment > $(MODULE_OBJ)/_header
	. $(VENV)/bin/activate && python3 $(MODULE_SRC)/introspect_header.py --header $(MODULE_OBJ)/_header --type-prefix $(UNUSED_KERNEL_PRIVATE_TYPES_PREFIX) --non-renamed-types $(PRIVATE_TYPES_TXT) > $(MODULE_OBJ)/_renamed_header

# Create type introspection macros
	. $(VENV)/bin/activate && python3 $(MODULE_SRC)/introspect_header.py --header $(MODULE_OBJ)/_full_header --introspect --kallsyms $(KALLSYMS) --kernel-features $(MODULE_SRC)/kernel_features.json >> $(MODULE_OBJ)/_introspection_header

# Build the final header
	printf '#pragma once\n' > $@
	cat $(MODULE_OBJ)/_introspection_header $(MODULE_OBJ)/_renamed_header >> $@
# cat $@

# out-of-tree build
endif

# Some kernels require the use of MODULE_IMPORT_NS() before using symbols that are part of the given namespace:
# https://docs.kernel.org/core-api/symbol-namespaces.html
# Unfortunately, Android kernels seem to define their own namespaces for GKI, so
# in order to avoid issues and work on any kernel, we simply attempt to list all
# of the namespaces this kernel seems to rely on by looking at the sources.
# We could use Module.symvers file, but it can only be generated when building a
# kernel so it would be way to slow.
# There does not seem to be any other source for the info in e.g. sysfs or
# procfs, so we rely on that hack for now.
$(SYMBOL_NAMESPACES_H): $(GENERATED)
	printf '#pragma once\n' > $@
	find "$(KERNEL_SRC)" '(' -name '*.c' -o -name '*.h' ')' -print0 | xargs -0 sed -n 's/MODULE_IMPORT_NS([^;]*;/\0/p' | sort -u >> $@
	echo "MODULE_IMPORT_NS(VFS_internal_I_am_really_a_filesystem_and_am_NOT_a_driver);" >> $@

$(MODULE_VERSION_H): $(GENERATED)
	printf '#pragma once\n' > $@
	printf "#define LISA_MODULE_VERSION \"" >> $@
	export LC_ALL=C && (cd $(MODULE_SRC) && find -type f '(' -name '*.c' -or -name '*.h' -or -name '*.txt' -or -name '*.json' -or -name '*.py' -or -name '*.lds' -or -name 'Makefile' -or -name 'vmlinux' -or -name 'kallsyms' ')' -and -not -path './generated/*' | xargs sha1sum) | sort | sha1sum | cut -d' ' -f1 | tr -d '\n' >> $@
	printf "\"\n" >> $@


$(KALLSYMS): $(GENERATED)
	cat $(MODULE_SRC)/kallsyms > $@ || $(NM) $(_REAL_VMLINUX) > $@ || touch $@


# Make all object files depend on the generated sources
$(addprefix $(MODULE_OBJ)/,$($(LISA_KMOD_NAME)-y)): $(INTROSPECTION_DATA_H) $(SYMBOL_NAMESPACES_H) $(MODULE_VERSION_H)

# Non-Kbuild part
else

.PHONY: all build install clean

all: install

build:
	"$(MAKE)" -C "$(KERNEL_SRC)" "M=$(MODULE_SRC)" modules

install: build
	"$(MAKE)" -C "$(KERNEL_SRC)" "M=$(MODULE_SRC)" modules_install

clean:
	"$(MAKE)" -C "$(KERNEL_SRC)" "M=$(MODULE_SRC)" clean

endif
