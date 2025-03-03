/* SPDX-License-Identifier: GPL-2.0 */

/// Types defined in this header have ABI compatibility with some types
/// manipulated in lisakmod_macros::inlinec module.
///
/// They must therefore be kept in sync to avoid any undefined behavior.

#ifndef _CFFI_H
#define _CFFI_H

#include <linux/types.h>

/* On recent kernels, kCFI is used instead of CFI and __cficanonical is therefore not
 * defined anymore
 */
#ifndef __cficanonical
#    define __cficanonical
#endif

struct slice_u8 {
	uint8_t *data;
	size_t len;
} __no_randomize_layout;

struct slice_const_u8 {
	const uint8_t *data;
	const size_t len;
} __no_randomize_layout;

#endif /* _CFFI_H */
