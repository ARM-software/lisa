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

#endif /* _UTILS_H */
