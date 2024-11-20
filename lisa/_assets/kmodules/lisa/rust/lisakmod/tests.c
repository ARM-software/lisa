/* SPDX-License-Identifier: GPL-2.0 */
#include "main.h"
#include "rust/lisakmod/bindings.h"

u64 myc_callback(u64 x) {
	return x + 1;
}
