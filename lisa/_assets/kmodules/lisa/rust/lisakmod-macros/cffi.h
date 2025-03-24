/* SPDX-License-Identifier: GPL-2.0 */

/// Types defined in this header have ABI compatibility with some types
/// manipulated in lisakmod_macros::inlinec module.
///
/// They must therefore be kept in sync to avoid any undefined behavior.

#ifndef _CFFI_H
#define _CFFI_H

#include <linux/types.h>

// FIMXE: Compiler complains about multichar char constant for '\0xf'

// In C, "char", "signed char" and "unsigned char" are 3 separate types, and
// pointers to each of them is incompatible with pointers to others.
// Unfortunately, C code makes heavy use of "char *", for which there is no
// Rust equivalent (as Rust only has an equivalent to int8_t and uint8_t).  As
// a result, Rust uses the core::ffi::c_char type defined as being either u8 or
// i8 based on the architecture, which translates to "unsigned char" or "signed
// char", so we must be able to cast C's "char *" to one of those.
#if '\xff' > 0
    typedef unsigned char c_char;
#else
    typedef signed char c_char;
#endif

#define CHAR_PTR_TO_KNOWN_SIGNEDNESS(x) _Generic((x), \
    const char *: (const c_char *)(x), \
    char *: (c_char *)(x), \
    char **: (c_char **)(x), \
    char * const *: (c_char * const *)(x), \
    const char **: (const c_char **)(x), \
    const char * const *: (const c_char * const *)(x), \
    default: (x) \
)

/* On recent kernels, kCFI is used instead of CFI and __cficanonical is therefore not
 * defined anymore
 */
#ifndef __cficanonical
#    define __cficanonical
#endif

#define __make_slice_ty(name, ty) \
	struct slice_##name { \
		__typeof__(ty) *data; \
		size_t len; \
	} __no_randomize_layout; \
	struct slice_const_##name { \
		const __typeof__(ty) *data; \
		const size_t len; \
	} __no_randomize_layout;

__make_slice_ty(u8, uint8_t)
__make_slice_ty(rust_str, struct slice_u8)

#endif /* _CFFI_H */
