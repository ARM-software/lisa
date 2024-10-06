/* SPDX-License-Identifier: GPL-2.0 */
#include "main.h"
#include "rust/validate.h"

u64 myc_callback(u64 x) {
	return x + 1;
}

// Use of a function pointer with a global variable forces the compiler to
// issue an indirect call, which exhibits issues with kernels compiled with
// CONFIG_CFI_CLANG
typedef u64 fnptr(u64, u64);
fnptr *myptr = test_1;

int __attribute__((no_sanitize("kcfi"))) rust_validate(void) {
	int ret = 0;

	if (myptr(1, 2) != 3) {
		pr_err("Rust test_1 failed");
		ret |= 1;
	}

	if (test_2(1, 2) != 4) {
		pr_err("Rust test_2 failed");
		ret |= 1;
	}

	if (test_3(1, 3) != 6) {
		pr_err("Rust test_3 failed");
		ret |= 1;
	}

	pr_info("Rust: tests finished");
	return ret;
}
