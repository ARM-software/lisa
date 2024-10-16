/* SPDX-License-Identifier: GPL-2.0 */
#include "main.h"
#include "rust/lisakmod/tests.h"

u64 myc_callback(u64 x) {
	return x + 1;
}

// Use of a function pointer with a global variable forces the compiler to
// issue an indirect call, which exhibits issues with kernels compiled with
// CONFIG_CFI_CLANG
typedef uint64_t fnptr(void);
fnptr *volatile myptr = &do_rust_tests;

int __attribute__((no_sanitize("kcfi"))) rust_tests(void) {
	return (int)myptr();
}
