/* SPDX-License-Identifier: GPL-2.0 */
#ifndef INTROSPECTION_H
#define INTROSPECTION_H

#include <generated/private_types.h>

#define HAS_TYPE(typ) ((defined(_TYPE_INTROSPECTION_INFO_AVAILABLE) && defined(_TYPE_EXISTS_##typ)) || (!defined(_TYPE_INTROSPECTION_INFO_AVAILABLE)))
#define HAS_MEMBER(typ, member) ((defined(_TYPE_INTROSPECTION_INFO_AVAILABLE) && defined(_TYPE_HAS_MEMBER_##typ##___##member)) || (!defined(_TYPE_INTROSPECTION_INFO_AVAILABLE)))
#define HAS_SYMBOL(name) ((defined(_SYMBOL_INTROSPECTION_INFO_AVAILABLE) && defined(_SYMBOL_EXISTS_##name)) || (!defined(_SYMBOL_INTROSPECTION_INFO_AVAILABLE)))

#endif /* INTROSPECTION_H */
