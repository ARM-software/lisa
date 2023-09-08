/* SPDX-License-Identifier: GPL-2.0 */
#ifndef INTROSPECTION_H
#define INTROSPECTION_H

#include <generated/introspection_data.h>


// Inspired by the kernel IS_ENABLED() implementation. This works around the
// fact that the defined() operator only works in #if and #elif, but putting a
// layer of macro breaks it.
#define __ARG_PLACEHOLDER_1 0,
#define __take_second_arg(__ignored, val, ...) val
#define __is_defined(x)			___is_defined(x)
#define ___is_defined(val)		____is_defined(__ARG_PLACEHOLDER_##val)
#define ____is_defined(arg1_or_junk)	__take_second_arg(arg1_or_junk 1, 0)

#define IS_DEFINED(option) (__is_defined(option))



#ifdef _TYPE_INTROSPECTION_INFO_AVAILABLE
#    define HAS_TYPE(kind, typ) IS_DEFINED(_TYPE_EXISTS_##kind##_##typ)
#    define HAS_MEMBER(kind, typ, member) (HAS_TYPE(kind, typ) && IS_DEFINED(_TYPE_HAS_MEMBER_##kind##_##typ##_LISA_SEPARATOR_##member))
#else
#    warning "Type introspection information not available, HAS_TYPE() and HAS_MEMBER() will assume types and members exist"
#    define HAS_TYPE(kind, typ) (1)
#    define HAS_MEMBER(kind, typ, member) (1)
#endif

#ifdef _SYMBOL_INTROSPECTION_INFO_AVAILABLE
#    define HAS_SYMBOL(name) IS_DEFINED(_SYMBOL_EXISTS_##name)
#else
#    warning "Symbol introspection information not available, HAS_SYMBOL() will assume all symbols exist"
#    define HAS_SYMBOL(name) (1)
#endif

#define HAS_KERNEL_FEATURE(name) (_KERNEL_HAS_FEATURE_##name)

#endif /* INTROSPECTION_H */
