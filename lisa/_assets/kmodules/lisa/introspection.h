/* SPDX-License-Identifier: GPL-2.0 */
#ifndef INTROSPECTION_H
#define INTROSPECTION_H

#include <generated/private_types.h>

#ifdef _TYPE_INTROSPECTION_INFO_AVAILABLE
#    define HAS_TYPE(typ) defined(_TYPE_EXISTS_##typ)
#    define HAS_MEMBER(typ, member) (HAS_TYPE(typ) && defined(_TYPE_HAS_MEMBER_##typ##___##member))
#else
#    warning "Type introspection information not available, HAS_TYPE() and HAS_MEMBER() will assume types and members exist"
#    define HAS_TYPE(typ) (1)
#    define HAS_MEMBER(typ, member) (1)
#endif


#ifdef _SYMBOL_INTROSPECTION_INFO_AVAILABLE
#    define HAS_SYMBOL(name) defined(_SYMBOL_EXISTS_##name)
#else
#    warning "Symbol introspection information not available, HAS_SYMBOL() will assume all symbols exist"
#    define HAS_SYMBOL(name) (1)
#endif

#endif /* INTROSPECTION_H */
