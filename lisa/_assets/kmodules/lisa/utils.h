/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _UTILS_H
#define _UTILS_H

// Following this recent patch, some kernel.h macros have been split in a new
// header so make sure to include it as well:
// https://lore.kernel.org/lkml/20230714142237.21836-1-andriy.shevchenko@linux.intel.com/t/
#if __has_include ("linux/args.h")
#    include "linux/args.h"
#endif

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/stringify.h>
#include <linux/version.h>
#include <linux/tracepoint.h>


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


// Kernel commit cdd30ebb1b9f36159d66f088b61aee264e649d7a changed
// MODULE_IMPORT_NS() from taking an identifier to taking a string literal.
#if LINUX_VERSION_CODE < KERNEL_VERSION(6,13,0)
#    define LISA_MODULE_IMPORT_NS(ns) MODULE_IMPORT_NS(ns)
#else
#    define LISA_MODULE_IMPORT_NS(ns) MODULE_IMPORT_NS(__stringify(ns))
#endif

// __assign_str() takes only one parameter since commit 2c92ca849fcc as
// __string() contains the source already. This allows ftrace to reserve the
// appropriate buffer size before-hand.
#if LINUX_VERSION_CODE < KERNEL_VERSION(6,10,0)
#    define __lisa_assign_str(field, src) __assign_str(field, src)
#else
#    define __lisa_assign_str(field, src) __assign_str(field)
#endif

#endif /* _UTILS_H */
