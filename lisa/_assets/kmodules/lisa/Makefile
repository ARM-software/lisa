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

# Invoked by Kbuild
ifneq ($(KERNELRELEASE),)
	MODULE_OBJ := $(obj)
	MODULE_SRC ?= $(src)
	KERNEL_SRC ?= $(srctree)
else
	MODULE_OBJ := $(PWD)
	MODULE_SRC ?= $(PWD)
	KERNEL_SRC ?= /lib/modules/`uname -r`/build
endif

GENERATED := $(MODULE_OBJ)/generated
$(GENERATED):
	rm -r "$@" 2>&1 || true
	mkdir -p "$@"
clean-files := $(GENERATED)

RUST_VERSION ?= nightly
RUSTFLAGS := -Clinker=rust-lld -Clink-self-contained=y -Clto=n -Crelocation-model=static -Cno-redzone=y -Ctarget-cpu=generic -Cforce-frame-pointers=y -Ccodegen-units=1 -Zfunction-sections=y

LISA_EVAL_C := $(MODULE_SRC)/lisa-eval-c.sh

RUST_SRC := $(MODULE_SRC)/rust/lisakmod
RUST_LDS := $(RUST_SRC)/rust.lds
RUST_GENERATED := $(GENERATED)/rust
RUST_OBJECT := $(RUST_GENERATED)/rust.o
RUST_OBJECT_CMD := $(RUST_GENERATED)/.rust.o.cmd
RUST_CBINDGEN_H := $(RUST_SRC)/cbindgen.h
RUST_C_SHIMS_H := $(RUST_GENERATED)/rust_c_shims.h
RUST_C_SHIMS_C := $(RUST_GENERATED)/rust_c_shims.c
RUST_C_SHIMS_DIR := $(RUST_GENERATED)/rust_c_shims

RUSTUP_HOME ?= $(HOME)/.rustup
CARGO_HOME ?= $(HOME)/.cargo
CARGO_TARGET_DIR ?= $(RUST_BUILD_DIR)/target

RUST_BUILD_DIR := $(RUST_GENERATED)/build
RUST_SYMBOLS := $(RUST_BUILD_DIR)/exported.list
RUST_SYMBOLS_CLI := $(RUST_BUILD_DIR)/exported.cli
RUST_CBINDGEN_BIN := $(CARGO_HOME)/bin/cbindgen

# Export c_flags so that lisa-eval-c.sh can compile a C source with the
# appropriate flags. Trying to pass down $(c_flags) via parameters or
# manually-exported env var triggers a nightmare of shell quoting as some of
# the compiler flags have quotes that are significant (e.g. ARM64_ASM_ARCH that
# needs to have double quotes in the value to be a well-formed C string
# literal).
#
# We need to expand the values once to mimic what seems to be happening in
# Kbuild.
export LISA_EVAL_C_CFLAGS=$(shell echo $(c_flags))

# Enable RUSTC_BOOTSTRAP=1 so we can use nightly features on any toolchain,
# including stable ones. This allows using the more tested stable binaries.
rust_cmd = chmod +x '$(LISA_EVAL_C)' && export RUSTC_BOOTSTRAP=1 PATH="$(CARGO_HOME)/bin:$$PATH" 'RUSTUP_HOME=$(RUSTUP_HOME)' 'CARGO_HOME=$(CARGO_HOME)' 'CARGO_TARGET_DIR=$(CARGO_TARGET_DIR)' 'LISA_EVAL_C=$(LISA_EVAL_C)' && $(1)
cargo_cmd = $(call rust_cmd,cargo +$(RUST_VERSION) $(1))

$(RUST_GENERATED): $(GENERATED)
	mkdir -p "$@"

$(RUST_BUILD_DIR): $(RUST_GENERATED)
	mkdir -p "$@"

$(CARGO_TARGET_DIR):
	mkdir -p "$@"

# Build the rust code into a static archive, then prelink it into an object
# file and filter the symbols to only keep as GLOBAL the exported pub
# #[no_mangle] ones. This avoids having tons of GLOBAL symbols from core lib.
$(RUST_OBJECT) $(RUST_C_SHIMS_H) $(RUST_C_SHIMS_C): $(RUST_BUILD_DIR) $(CARGO_TARGET_DIR)
	# Build the crate into a static library
	cd $(RUST_SRC) && export RUSTFLAGS="$(RUSTFLAGS)" && $(call cargo_cmd,build --release --target=$(RUST_SRC)/rustc_targets/$(ARCH)/target.json -Zbuild-std=core$(comma)alloc)

	# Prelink the archive into a single object file.
	$(LD) $(KBUILD_LDFLAGS) -nostdlib -r -o $(RUST_OBJECT) --whole-archive $(CARGO_TARGET_DIR)/target/release/liblisakmod.a

	# Get the list of exported symbols
	python3 "$(MODULE_SRC)/rust_exported_symbols.py" --rust-object $(RUST_OBJECT) --out-symbols-plain $(RUST_SYMBOLS) --out-symbols-cli $(RUST_SYMBOLS_CLI)

	# Extract the C shims created by our inlinec Rust module from the
	# binary, so they can be compiled and linked later.
	mkdir -p "$(RUST_C_SHIMS_DIR)"
	touch $(RUST_C_SHIMS_DIR)/shim_.binstore.c.header. $(RUST_C_SHIMS_DIR)/shim_.binstore.c.code.
	# Both llvm-objdump and llvm-objcopy can have a bug where the input
	# file is overwritten by some garbage. This therefore destroys rust.o,
	# which we unfortunately need intact for linking, so we make a copy for
	# each of those tool to play with.
	cp $(RUST_OBJECT) $(RUST_OBJECT)_objdump.o
	$(OBJDUMP) -h $(RUST_OBJECT)_objdump.o | awk '{ print $$2 }' | grep '^.binstore.c.' | xargs -n1 sh -c '$(OBJCOPY) --dump-section $$0=$(RUST_C_SHIMS_DIR)/shim_$$0 $(RUST_OBJECT) /dev/null'
	LC_ALL=C cat $(RUST_C_SHIMS_DIR)/shim_.binstore.c.header.* > $(RUST_C_SHIMS_H)
	LC_ALL=C cat $(RUST_C_SHIMS_DIR)/shim_.binstore.c.code.* > $(RUST_C_SHIMS_C)
	cat $(RUST_C_SHIMS_H) $(RUST_C_SHIMS_C) | if which clang-format 2>&1 1>/dev/null; then clang-format; else cat; fi

	# Garbage collect unused sections in the object file, after we have
	# extracted the binstore sections (otherwise they get discarded).
	#
	# Use --undefined CLI parameters in $(RUST_SYMBOLS_CLI) instead of
	# EXTERN() command in a linker script since older GNU ld version seem
	# to ignore the EXTERN() command with --gc-sections.
	$(LD) --version
	# Some versions of ld.lld are broken and this will not work. In that
	# case, we just let it be and the binary will simply be bigger
	($(LD) $(KBUILD_LDFLAGS) $$(cat $(RUST_SYMBOLS_CLI)) --gc-sections -nostdlib -r -o $(RUST_OBJECT)_gced.o $(RUST_OBJECT) && mv $(RUST_OBJECT)_gced.o $(RUST_OBJECT)) || true

	# Only keep as GLOBAL symbols the ones that are to be exported (and the
	# undefined ones to be provided by C code)
	$(OBJCOPY) --keep-global-symbols $(RUST_SYMBOLS) $(RUST_OBJECT)

	# Provide the pre-built .o to Kbuild
	cp $(RUST_OBJECT) $(RUST_OBJECT)_shipped
	touch $(RUST_OBJECT_CMD)

# This rule is necessary as Kbuild cannot cope with absolute paths on the
# object file list to link in the module. Unfortunately, make is not smart
# enough to realize that the relative path being asked for is the same location
# as the absolute path we use in $(RUST_OBJECT), so we need to make that
# explicit here.
generated/rust/rust.o: $(RUST_OBJECT)

genbin: $(RUST_OBJECT)

$(RUSTUP_HOME):
	$(call rust_cmd,curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none)

$(CARGO_HOME): $(RUSTUP_HOME)
	$(call rust_cmd,rustup install nightly --profile=minimal)

$(RUST_CBINDGEN_BIN): $(CARGO_HOME)
	$(call cargo_cmd,install --locked cbindgen)

# This is for the benefit of the user and CI to generate the file to be
# committed if it does not exist yet.
refresh-rust-bindings: $(RUST_CBINDGEN_BIN)
	cd $(RUST_SRC) && $(call rust_cmd,cbindgen --config $(RUST_SRC)/cbindgen.toml --lang c --crate lisakmod --output $(RUST_CBINDGEN_H))

$(RUST_SYMBOLS): $(RUST_OBJECT)

# kbuild part of makefile. Only Kbuild-related targets should be used here to
# avoid any sort of clash.
ifneq ($(KERNELRELEASE),)

LISA_KMOD_NAME ?= lisa
obj-m := $(LISA_KMOD_NAME).o
$(LISA_KMOD_NAME)-y := main.o tp.o wq.o features.o pixel6.o introspection_data.o rust/lisakmod/runtime.o rust/lisakmod/tests.o generated/rust/rust.o

# -fno-stack-protector is needed to avoid possibly undefined __stack_chk_guard symbol
ccflags-y := "-I$(MODULE_SRC)" "-I$(MODULE_OBJ)" -std=gnu11 -fno-stack-protector -Wno-declaration-after-statement -Wno-error

# Ensure we get appropriate debug info so that pahole can work on the object
# file regardless of the kernel config.
CFLAGS_introspection_data.o := -g3 -gno-split-dwarf -gdwarf -fno-eliminate-unused-debug-types

FEATURES_LDS := $(MODULE_SRC)/features.lds
SYMBOLS_LDS := $(GENERATED)/symbols.lds

SYMBOL_NAMESPACES_H := $(GENERATED)/symbol_namespaces.h
MODULE_VERSION_H := $(GENERATED)/module_version.h
KALLSYMS := $(GENERATED)/kallsyms

ldflags-y += -T $(FEATURES_LDS) -T $(RUST_LDS) -T $(SYMBOLS_LDS)

INTROSPECTION_DATA_H := $(GENERATED)/introspection_data.h

# Can be either a kernel image built with DWARF debug info, or the BTF blob
# found at /sys/kernel/btf/vmlinux
_BTF_VMLINUX := $(MODULE_SRC)/vmlinux
_REAL_VMLINUX := $(KERNEL_SRC)/vmlinux
_SYNTHETIC_VMLINUX := $(MODULE_OBJ)/introspection_data.o

BTF_BLOB := $(GENERATED)/btf.blob

ifeq ("$(wildcard $(_BTF_VMLINUX))","")
    VMLINUX := $(_REAL_VMLINUX)
else
    VMLINUX := $(_BTF_VMLINUX)
endif

# Ensure we have a BTF blob regardless of vmlinux format (ELF with .BTF
# section, ELF with DWARF debug info and no BTF, BTF blob).
$(BTF_BLOB): $(GENERATED) $(VMLINUX)
	# If file is ELF, objcopy will succeed (and possibly create an empty
	# file if the section does not exist).
	# Otherwise, we assume it is a BTF blob and copy it over to the output.
	# Extracting this way avoids the costly pahole --btf_encode step that
	# would convert DWARF to BTF even if .BTF section is already present.
	# It also avoids calling pahole in the happy path, since
	# --btf_encode_detached option might not be available on older
	# versions.
	$(OBJCOPY) -O binary --only-section=.BTF "$(VMLINUX)" "$@" || cp "$(VMLINUX)" "$@"
	# If the file was empty, the ELF file was missing BTF info so we generate it.
	[ -s "$@" ] || pahole --btf_gen_all --btf_encode_detached "$@" "$(VMLINUX)"

$(INTROSPECTION_DATA_H): $(GENERATED) $(KALLSYMS) $(BTF_BLOB)
	printf '#pragma once\n' > "$@"
	# Create type introspection macros
	python3 "$(MODULE_SRC)/introspect_header.py" --introspect --btf "$(BTF_BLOB)" --internal-type-prefix KERNEL_PRIVATE_ --kallsyms "$(KALLSYMS)" --conf "$(MODULE_SRC)/introspection.json" >> "$@"

$(SYMBOLS_LDS): $(GENERATED) $(KALLSYMS)
	python3 $(MODULE_SRC)/introspect_header.py --kallsyms $(KALLSYMS) --symbols-lds >> "$@"

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
	printf '#pragma once\n#include "linux/module.h"\n#include "utils.h"\n' > $@
	find "$(KERNEL_SRC)" '(' -name '*.c' -o -name '*.h' ')' -print0 | xargs -0 sed -n 's/MODULE_IMPORT_NS([^;]*;/\0/p' | sort -u >> $@
	echo 'LISA_MODULE_IMPORT_NS(VFS_internal_I_am_really_a_filesystem_and_am_NOT_a_driver);' >> $@
	cat $@

$(MODULE_VERSION_H): $(GENERATED)
	printf '#pragma once\n' > $@
	printf "#define LISA_MODULE_VERSION \"" >> $@
	# Show the content of the folder to debug mismatching sha1sum issues
	ls -laR $(MODULE_SRC)
	export LC_ALL=C && (cd $(MODULE_SRC) && find -type f '(' -name '*.c' -or -name '*.h' -or -name '*.rs' -or -name '*.toml' -or -name '*.lock' -or -name '*.txt' -or -name '*.json' -or -name '*.md' -or -name '*.py' -or -name '*.sh' -or -name '*.lds' -or -name 'Makefile' -or -name 'vmlinux' -or -name 'kallsyms' ')' -and -not -path './generated/*' -and -not -path './__pycache__/*' | xargs sha1sum) | sort | sha1sum | cut -d' ' -f1 | tr -d '\n' >> $@
	printf "\"\n" >> $@


$(KALLSYMS): $(GENERATED)
	cat $(MODULE_SRC)/kallsyms > $@ || $(NM) $(_REAL_VMLINUX) > $@ || touch $@

# Make all object files depend on the generated sources. We use both the
# relative and absolute paths as kernels <= 6.12 require an absolute path
# whereas >= 6.13 requires a relative path.
$($(LISA_KMOD_NAME)-y) $(addprefix $(MODULE_OBJ)/,$($(LISA_KMOD_NAME)-y)): $(INTROSPECTION_DATA_H) $(SYMBOL_NAMESPACES_H) $(MODULE_VERSION_H) $(SYMBOLS_LDS) $(RUST_CBINDGEN_H) $(RUST_C_SHIMS_H) $(RUST_C_SHIMS_C)

# Non-Kbuild part
else

.PHONY: all build install clean refresh-rust-bindings

all: install

build:
	"$(MAKE)" -C "$(KERNEL_SRC)" "M=$(MODULE_SRC)" modules

install: build
	"$(MAKE)" -C "$(KERNEL_SRC)" "M=$(MODULE_SRC)" modules_install

clean:
	"$(MAKE)" -C "$(KERNEL_SRC)" "M=$(MODULE_SRC)" clean

endif
