#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate a snippet of C code as an unsigned integer constant and return the
# value in hexadecimal format.

c_headers=$1
c_expr=$2

build=$(mktemp -d)
object=$build/object.o
src=$build/src.c

cleanup() {
    rm -r "$build"
}

trap cleanup EXIT

# Evaluate a C constant expression as the size of an array, and then later
# extract the symbol size from the symbol table after compiling. This is works
# reliably on any compiler and provides the value in hex format.

# We add +1 to the expression, as symbol size for a zero-sized array seems to
# be 1 weirdly enough. That +1 is then substracted in the awk post-processing
cat > $src << EOM
$c_headers
static char __attribute__((used)) LISA_C_CONST_VALUE[($c_expr) + 1];
EOM

CC=${CC:=cc}
NM=${NM:=nm}

if ("$CC" --version | grep -i clang) &>/dev/null; then
    clang_args=(
        # Some CLI args are not used in our simple invocation (e.g. some
        # linker-related things), so we don't want that to become a hard error.
        -Wno-error=unused-command-line-argument

        # If LTO is enabled, clang will emit LLVM bitcode file instead of object files.
        # It turns out llvm-nm is broken and reports 0 size for the symbols
        # when passed an LLVM bitcode file:
        # https://github.com/llvm/llvm-project/issues/33743
        #
        # Consequently, we need to either disable LTO or convince clang to
        # still emit an ELF object file with LLVM bitcode in a section instead.
        # This can be done with -ffat-lto-objects, but that option is only
        # available starting from clang 18. For compat with older versions, we
        # disable LTO, and we also disable CFI sanitizer since it requires LTO.
        # Note that CFI is similar but not the same as the kCFI sanitizer that
        # kernels >= 6.1 exploit. kCFI can be enabled independently of LTO.
        -fno-lto
        -fno-sanitize=cfi
    )
fi

# All -I are relative to the kernel tree root, so we need to run from there.
cd "$KERNEL_SRC" &&
$CC $LISA_EVAL_C_CFLAGS "${clang_args[@]}" -c "$src" -o "$object" && $NM -S "$object" | awk '{if ($4 == "LISA_C_CONST_VALUE")  {print strtonum("0x" $2) - 1}}'

