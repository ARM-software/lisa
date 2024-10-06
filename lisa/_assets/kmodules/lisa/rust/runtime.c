/* SPDX-License-Identifier: GPL-2.0 */
#include "main.h"
#include "rust/cbindgen.h"

/* Basic glue between Rust runtime and kernel functions */


u8 *__lisa_rust_alloc(size_t size) {
	return kmalloc(size, GFP_KERNEL);
}

void __lisa_rust_dealloc(u8 *ptr) {
	kfree(ptr);
}

u8 *__lisa_rust_alloc_zeroed(size_t size) {
	return kzalloc(size, GFP_KERNEL);
}

u8 *__lisa_rust_realloc(u8 *ptr, size_t size) {
	if (!size) {
		// Do not feed a size=0 to krealloc() as it will free it,
		// leading to a double-free.
		size = 1;
	}
	return krealloc(ptr, size, GFP_KERNEL);
}

void __lisa_rust_panic(const u8 *msg, size_t len) {
	if (msg && len) {
		panic("Rust panic: %.*s", (int)len, msg);
	} else {
		panic("Rust panic with no message");
	}
}

void __lisa_rust_pr_info(const u8 *msg, size_t len) {
	if (msg) {
		if (len) {
			pr_info("%.*s", (int)len, msg);
		} else {
			pr_info("");
		}
	} else {
		pr_info("(null)");
	}
}
