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

cat > $src << EOM
$c_headers
static char __attribute__((used)) LISA_C_CONST_VALUE[$c_expr];
EOM

CC=${CC:=cc}
NM=${NM:=nm}

if ("$CC" --version | grep -i clang) &>/dev/null; then
    clang_args=(-Wno-error=unused-command-line-argument)
fi

# All -I are relative to the kernel tree root, so we need to run from there.
cd "$KERNEL_SRC" &&
$CC $LISA_EVAL_C_CFLAGS "${clang_args[@]}" -c "$src" -o "$object" && $NM -S "$object" | awk '{if ($4 == "LISA_C_CONST_VALUE") {print "0x" $2}}'

