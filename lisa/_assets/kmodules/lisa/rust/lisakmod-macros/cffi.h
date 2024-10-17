/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/types.h>

/// Types defined in this header have ABI compatibility with some types
/// manipulated in lisakmod_macros::inlinec module.
///
/// They must therefore be kept in sync to avoid any undefined behavior.

#ifndef _CFFI_H
#define _CFFI_H

struct slice_u8 {
	uint8_t *data;
	size_t len;
};

struct slice_const_u8 {
	const uint8_t *data;
	const size_t len;
};

#endif /* _CFFI_H */
