/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _UTILS_H
#define _UTILS_H

// Following this recent patch, some kernel.h macros have been split in a new
// header so make sure to include it as well:
// https://lore.kernel.org/lkml/20230714142237.21836-1-andriy.shevchenko@linux.intel.com/t/
#if __has_include ("linux/args.h")
#    include "linux/args.h"
#endif

#include "linux/kernel.h"

#define IGNORE_WARNING(warning, expr) ({ \
	__diag_push(); \
	__diag(ignored warning); \
	typeof(expr) ___expression_value = expr; \
	__diag_pop(); \
	___expression_value; \
})

#ifdef __clang__
#    define PER_COMPILER(clang, gcc) clang
#else
#    define PER_COMPILER(clang, gcc) gcc
#endif

/// CONST_CAST - Same as C++ const_cast<>
#define CONST_CAST(type, expr) IGNORE_WARNING( \
	PER_COMPILER( \
		"-Wincompatible-pointer-types-discards-qualifiers", \
		"-Wignored-qualifiers" \
	), \
	(type)(expr) \
)

#endif /* _UTILS_H */
