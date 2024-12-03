/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _RUST_BINDINGS_H
#define _RUST_BINDINGS_H

// cbindgen relies on ISO C types such as uint32_t
#include <linux/types.h>
#include "rust/lisakmod/cbindgen.h"

// rust_c_shims.h relies on types defined in cffi.h
#include "rust/lisakmod-macros/cffi.h"
#include "generated/rust/rust_c_shims.h"

#endif /* _RUST_BINDINGS_H */
