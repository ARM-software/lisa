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

#define __STRINGIFY_12(X, ...)	__stringify(X), __STRINGIFY_11(__VA_ARGS__)
#define __STRINGIFY_11(X, ...)	__stringify(X), __STRINGIFY_10(__VA_ARGS__)
#define __STRINGIFY_10(X, ...)	__stringify(X), __STRINGIFY_9(__VA_ARGS__)
#define __STRINGIFY_9(X, ...)	__stringify(X), __STRINGIFY_8(__VA_ARGS__)
#define __STRINGIFY_8(X, ...)	__stringify(X), __STRINGIFY_7(__VA_ARGS__)
#define __STRINGIFY_7(X, ...)	__stringify(X), __STRINGIFY_6(__VA_ARGS__)
#define __STRINGIFY_6(X, ...)	__stringify(X), __STRINGIFY_5(__VA_ARGS__)
#define __STRINGIFY_5(X, ...)	__stringify(X), __STRINGIFY_4(__VA_ARGS__)
#define __STRINGIFY_4(X, ...)	__stringify(X), __STRINGIFY_3(__VA_ARGS__)
#define __STRINGIFY_3(X, ...)	__stringify(X), __STRINGIFY_2(__VA_ARGS__)
#define __STRINGIFY_2(X, ...)	__stringify(X), __STRINGIFY_1(__VA_ARGS__)
#define __STRINGIFY_1(X)	__stringify(X)
#define __STRINGIFY_0()

/** Stringify up to 12 args. */
#define STRINGIFY_ALL(...)	CONCATENATE(__STRINGIFY_, COUNT_ARGS(__VA_ARGS__))(__VA_ARGS__)

/** Clamp X to a value in [A,B]. */
#define CLAMP(X, A, B)		((X) < (A) ? (A) : ((X) > (B) ? (B) : (X)))

#endif /* _UTILS_H */
