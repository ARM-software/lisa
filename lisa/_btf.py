# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2023, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This module implements a BTF debug info parser as described by:
https://www.kernel.org/doc/html/latest/bpf/btf.html
"""


import itertools
import struct
import enum
import functools
import io
import copy
import re
import contextlib
from operator import attrgetter
import inspect
import json


def _next_multiple(x, n):
    if n:
        remainder = x % n
        if remainder:
            return x + n - remainder
        else:
            return x
    else:
        return x


def _scan(buf, f):
    xs = []
    while buf:
        buf, x = f(buf)
        xs.append(x)
    return xs


class BTFSignedness(enum.Enum):
    SIGNED = 1
    UNSIGNED = 0

    @classmethod
    def from_int(cls, v):
        if v:
            return cls.SIGNED
        else:
            return cls.UNSIGNED


class BTFIntEncoding(enum.Enum):
    NONE = 0
    SIGNED = 1
    CHAR = 2
    BOOL = 3

    @classmethod
    def from_int(cls, v):
        #define BTF_INT_SIGNED  (1 << 0)
        #define BTF_INT_CHAR    (1 << 1)
        #define BTF_INT_BOOL    (1 << 2)

        if v == 0:
            return cls.NONE
        elif v == 1 << 0:
            return cls.SIGNED
        elif v == 1 << 1:
            return cls.CHAR
        elif v == 1 << 2:
            return cls.BOOL
        else:
            raise ValueError(f'Unknown int encoding: {v}')


class _TypeRef:
    __slots__ = ('index',)
    def __init__(self, index):
        self.index = index

    @classmethod
    def fixup(cls, v, typs):
        if isinstance(v, cls):
            v = typs[v.index]
        return v


class _BTFTypeMeta(type):
    def __new__(metacls, name, bases, dct, **kwargs):
        new = super().__new__(metacls, name, bases, dct, **kwargs)
        mro = inspect.getmro(new)
        new._TYP_ATTRS = tuple(sorted(set(
            attr
            for cls in mro
            for attr in cls.__dict__.get('_TYP_ATTRS', ())
        )))
        new._ITERABLE_TYP_ATTRS = tuple(sorted(set(
            attr
            for cls in mro
            for attr in cls.__dict__.get('_ITERABLE_TYP_ATTRS', ())
        )))

        return new


class BTFType(metaclass=_BTFTypeMeta):
    __slots__ = ('id',)
    _TYP_ATTRS = ()
    _ITERABLE_TYP_ATTRS = ()

    def __init__(self):
        self.id = None

    @classmethod
    def reachable_from(cls, typs):
        reachable_typs = set()
        for typ in typs:
            typ._map_typs(lambda x: x, visited=reachable_typs)
        return reachable_typs

    @classmethod
    def map_typs(cls, f, typs):
        visited = set()
        for typ in typs:
            typ._map_typs(f, visited=visited)

    def _map_typs(self, f, visited):
        if self in visited:
            return
        else:
            visited.add(self)
            for attr in self._TYP_ATTRS:
                x = getattr(self, attr)
                x = f(x)
                setattr(self, attr, x)
                if isinstance(x, BTFType):
                    x._map_typs(f, visited)

            for attr in self._ITERABLE_TYP_ATTRS:
                for x in getattr(self, attr):
                    f(x)
                    x._map_typs(f, visited)

    def _unwrap(self, classinfo):
        typ = self
        while isinstance(typ, classinfo):
            typ = typ.typ
        return typ

    @property
    def unspecified(self):
        return self._unwrap(_CDeclSpecifier)

    @property
    def underlying(self):
        return self._unwrap((BTFTypedef, _CDeclSpecifier))

    @property
    def is_incomplete(self):
        return False

    def _dump_c_introspection(self, ctx):
        return


class _CDecl:
    def _do_dump_c_decls(self, ctx):
        pass

    def _dump_c_decls(self, ctx, **kwargs):
        if kwargs.get('memoize', True):
            try:
                x = ctx._memo[self]
            except KeyError:
                x = self._do_dump_c_decls(ctx, **kwargs)
                ctx._memo[self] = x
        else:
            x = self._do_dump_c_decls(ctx, **kwargs)

        return x


class _CBuiltin(_CDecl):
    def _do_dump_c_decls(self, ctx):
        return self.name


class _CSizedBuiltin(_CBuiltin):
    def _do_dump_c_decls(self, ctx):
        name = self.name
        size = self.size
        ctx.write(
            f'_Static_assert(sizeof({name}) == {size}ull, "{name} does not have expected size: {size}");\n'
        )
        return name


class _FixupTyp:
    _TYP_ATTRS = ('typ',)


class _TransparentType(_FixupTyp):
    @property
    def bits(self):
        return self.typ.bits

    @property
    def size(self):
        return self.typ.size

    @property
    def alignment(self):
        return self.typ.alignment

    @property
    def is_incomplete(self):
        return self.typ.is_incomplete

    def _dump_c_introspection(self, ctx):
        return self.typ._dump_c_introspection(ctx)


class _CQualifier:
    pass


class _CDeclSpecifier(_TransparentType, _CDecl, BTFType):
    __slots__ = ('typ',)

    def __init__(self, typ):
        super().__init__()
        self.typ = typ

    def __str__(self):
        return self._c_specifier

    def _do_dump_c_decls(self, ctx, anonymous=False, **kwargs):
        typ = self.typ

        # It is important to dump anonymous struct/union as an inline
        # definition rather than adding a typedef, since inline anonymous
        # struct/unions are expanded in their parent struct union (which would
        # not happen if we added a typedef layer).
        if anonymous:
            typename = typ._dump_c_decls(
                ctx,
                anonymous=anonymous,
                **kwargs,
            )
            return f'{self._c_specifier} {typename}'
        # Qualifying a function type is forbidden by C spec. For some reason,
        # some kernel seem to include that in BTF, but maybe it's a bug in
        # pahole.
        elif isinstance(typ, BTFFuncProto) and isinstance(self, _CQualifier):
            return typ._dump_c_decls(ctx)
        else:
            name = ctx.make_name()
            typename = typ._dump_c_decls(ctx)
            ctx.write(
                f'typedef {self._c_specifier} {typename} {name};\n',
            )
            return name


class BTFVoid(_CBuiltin, BTFType):
    name = 'void'


class BTFInt(_CSizedBuiltin, BTFType):
    __slots__ = ('name', 'size', 'encoding', 'bit_offset', 'bits')

    def __init__(self, name, size, encoding, bit_offset, bits):
        super().__init__()
        self.name = name
        self.encoding = encoding
        self.bit_offset = bit_offset
        self.size = size
        self.bits = bits
        assert bits <= (size * 8)

    @property
    def is_bitfield(self):
        return self.bits != (self.size * 8)

    @property
    def alignment(self):
        return self.size


class BTFFloat(_CSizedBuiltin, BTFType):
    __slots__ = ('name', 'size')

    def __init__(self, name, size):
        super().__init__()
        self.name = name
        self.size = size

    @property
    def bits(self):
        return self.size * 8

    @property
    def alignment(self):
        return self.size


class BTFPtr(_FixupTyp, _CDecl, BTFType):
    __slots__ = ('typ', 'size')

    def __init__(self, typ, size):
        super().__init__()
        self.typ = typ
        self.size = size

    def _do_dump_c_decls(self, ctx):
        size = self.size
        name = ctx.make_name()

        # Turn a struct/union pointee into a forward reference, so that we
        # break any circular references.
        typ = self._break_loop()
        typename = typ._dump_c_decls(ctx)
        ctx.write(
            f'typedef {typename} *{name};\n'
        )

        if size is not None:
            ctx.write(
                f'_Static_assert(sizeof({name}) == {size}ull, "Pointer to {typename} does not have expected size: {size}");\n'
            )
        return name

    def _break_loop(self):
        underlying = self.typ.underlying

        # If we are pointing at a struct/union, make a new pointee chain that:
        #    * ends up on a forward reference instead of the struct/union
        #    * does not contain any typedef
        #
        # This breaks any loop in self-referential structs, and removing the
        # typedefs avoid creating a typedef to an incomplete type (the forward
        # declaration), which would be unusable for some non-pointer consumers.
        if isinstance(underlying, _BTFStructUnion) and underlying.fwd_decl is not None:
            root = copy.copy(self)
            parent = root
            while True:
                child = parent.typ
                # Get rid of typedefs, since we don't want to dump types that
                # will be possibly dumped later by something else depending on
                # it, as we are creating copies of the beginning of the pointee
                # chain.
                while isinstance(child, BTFTypedef):
                    child = child.typ

                if isinstance(child, _BTFStructUnion):
                    parent.typ = child.fwd_decl
                    break
                else:
                    child = copy.copy(child)
                    parent.typ = child
                    parent = child
            return root.typ
        else:
            return self.typ

    @property
    def bits(self):
        return self.size * 8

    @property
    def alignment(self):
        return self.size


class BTFEnum(_CDecl, BTFType):
    __slots__ = ('name', 'signed', 'size', 'enumerators', 'int_typ')
    _KIND = 'enum'
    _TYP_ATTRS = ('int_typ',)

    def __init__(self, name, signed, size, enumerators, int_typ):
        super().__init__()
        self.name = name
        self.signed = signed
        self.size = size
        self.enumerators = enumerators
        self.int_typ = int_typ

    def __str__(self):
        name = self.name or '<anonymous>'
        return f'enum {name}'

    @property
    def bits(self):
        return self.size * 8

    @property
    def alignment(self):
        return self.size

    def _dump_c_introspection(self, ctx):
        ctx.typ_exists(self)
        with ctx.with_parent(self) as ctx:
            for enumerator in self.enumerators.keys():
                ctx.typ_member(enumerator)

    def _do_dump_c_decls(self, ctx):
        enumerators = self.enumerators
        size = self.size
        int_typ = self.int_typ

        def format_enumerator(name, value):
            value = '' if value is None else f'={value}{"ull" if value >= 0 else "ll"}'
            return f'{name}{value}'

        enumerators_str = ', '.join(
            format_enumerator(name, value)
            for name, value in enumerators.items()
        )

        mode = 'packed' if size < 4 else None
        attrs = f'__attribute__(({mode}))' if mode else ''

        name = self.name or ctx.make_name()
        ctx.write(
            f'enum {name} {{ {enumerators_str} }} {attrs};\n',
        )

        # Old pahole version emit a BTF_KIND_ENUM instead of
        # BTF_KIND_ENUM64, leading to "0" values in enumerators with high
        # values. This prevents creating an enum of the right size, so in order
        # to avoid broken structs layout we just pretend they are an integer of
        # the correct size.
        if size < 8 or not int_typ:
            ctx.write(
                f'_Static_assert(sizeof(enum {name}) == {size}ull, "enum {name} does not have expected size: {size}");\n'
            )
            return f'enum {name}'
        elif int_typ:
            return int_typ._dump_c_decls(ctx)
        else:
            raise ValueError(f'Could not dump enum {name} as its size is {size} and no integer type of the right size has been detected')


class _BTFStructUnion(_CDecl, BTFType):
    __slots__ = ('name', 'size', 'members', 'fwd_decl')
    _ITERABLE_TYP_ATTRS = ('members',)

    def __init__(self, name, size, members):
        super().__init__()
        self.name = name
        self.size = size
        self.members = []
        self.fwd_decl = _LinkedForwardDecl(self) if name else None

        for member in members:
            # Avoid modifying the existing member in case it is shared with
            # another struct.
            member = copy.copy(member)
            member.parent = self
            self.members.append(member)

    def __str__(self):
        name = self.name or '<anonymous>'
        return f'{self._KIND} {name}'

    def _dump_c_introspection(self, ctx):
        ctx.typ_exists(self)
        with ctx.with_parent(self) as ctx:
            for member in self.members:
                member._dump_c_introspection(ctx)

    def _do_dump_c_decls(self, ctx, anonymous=False, memoize=True, parents=None):
        members = self._all_members
        size = self.size
        kind = self._KIND
        name = self.name or ctx.make_name()
        fwd_decl = f'{kind} {name}'
        _parents = parents or []
        children_parents = (*_parents, self)

        def format_member(member):
            name = member.name
            typ = member.typ
            if name:
                typename = typ._dump_c_decls(ctx)
                is_bitfield = member.is_bitfield
                assert member.bits or not is_bitfield
                bitfield = f': {member.bits}' if is_bitfield else ''
                aligned = is_bitfield or not (member.bit_offset % (member.alignment * 8))
                packed = '' if aligned else ' __attribute__((packed))'
                return f'{typename} {name}{bitfield}{packed}'
            # ISO C allows anonymous structs/union inside a struct/union.
            elif isinstance(typ.unspecified, _BTFStructUnion):
                # Bypass the caching since we absolutely do not want to get
                # back an internal typedef, which would break the layout of the
                # parent struct. Anonymous union/struct are only useful if
                # their declaration are expanded in the parent.
                return typ._dump_c_decls(ctx, anonymous=True, memoize=False, parents=children_parents)
            else:
                raise ValueError(f'Unsupported anonymous member in {self}')

        def format_assert(fwd_decl, member):
            name = member.name
            assert name
            bit_offset = member.bit_offset
            byte_offset = bit_offset // 8
            assert byte_offset >= 0
            return f'_Static_assert(__builtin_offsetof({fwd_decl}, {name}) == {byte_offset}ull, "{fwd_decl}.{name} does not have the expected offset: {byte_offset}")'

        attrs, last_padding = self._align_attribute(parents=_parents)
        if last_padding:
            members = (*members, last_padding)
        attrs = attrs or ''
        attrs = f'__no_randomize_layout {attrs}'

        members_str = '; '.join(map(format_member, members))
        members_str = f'{members_str};' if members_str else ''

        if anonymous:
            return f'{kind} {{ {members_str} }} {attrs}'
        else:
            ctx.write(
                f'{fwd_decl} {{ {members_str} }} {attrs};\n',
            )
            sep = ';\n'
            asserts = sep.join(
                format_assert(fwd_decl, member)
                for member in members
                if member.name and not member.is_bitfield
            )
            if asserts:
                ctx.write(f'{asserts}{sep}')

            ctx.write(
                f'_Static_assert(sizeof({fwd_decl}) == {size}ull, "{fwd_decl} does not have expected size: {size}");\n'
            )
            return fwd_decl

    @property
    def bits(self):
        return self.size * 8

    @property
    def _min_alignment(self):
        members = self.members
        if members:
            return max(member.alignment for member in members)
        else:
            return 1

    @property
    def is_incomplete(self):
        return any(member.typ.is_incomplete for member in self.members)

    @property
    def _min_size(self):
        members = self.members
        size = self.size
        min_alignment = self._min_alignment

        if members:
            last = members[-1]
            return _next_multiple(last.bit_offset + last.bits, 8) // 8
        else:
            return 0


class BTFMember(_FixupTyp, BTFType):
    __slots__ = ('name', 'typ', 'bit_offset', '_bits', 'parent')
    _TYP_ATTRS = ('typ',)

    def __init__(self, name, typ, bits, bit_offset):
        super().__init__()
        self.name = name
        self.typ = typ
        self.bit_offset = bit_offset
        # bits could be 0 un a variety of circumsances, but it always means the
        # value is to be taken from somewhere else since a bitfield of size 0
        # is not allowed in C.
        self._bits = bits or None
        self.parent = None

    @property
    def _prev_member(self):
        i = self._index
        if i:
            return self.parent.members[i - 1]
        else:
            return None

    @property
    def _index(self):
        members = self.parent.members
        return members.index(self)

    @property
    def alignment(self):
        return self.typ.alignment

    @property
    def bits(self):
        typ = self.typ
        if isinstance(typ.underlying, (BTFInt, BTFEnum)):
            # If the member had a bitsize defined directly, we use that,
            # otherwise the bitsize is "inherited" from the type.
            return self._bits or typ.bits
        else:
            return typ.bits

    @property
    def is_bitfield(self):
        typ = self.typ
        if self.bits != typ.bits:
            return True
        else:
            underlying = typ.underlying
            return isinstance(underlying, BTFInt) and underlying.is_bitfield

    @property
    def pre_padding(self):
        bit_offset = self.bit_offset
        # If the bit_offset is not a multiple of 8, it means we are in the
        # middle of a bitfield block and there is no padding to be added.
        #
        # Note that anonymous bitfield of size 0 can be used to force
        # allocation of the next bitfield in another block, so if that's the
        # case we very much want to insert the appropriate padding.
        if (bit_offset % 8) or isinstance(self.parent, BTFUnion):
            return ()
        else:
            bits = self.bits
            # Could be a bitfield, in which case we round up to the next byte
            size = _next_multiple(bits, 8) // 8

            prev_member = self._prev_member
            if prev_member:
                # The previous member might be a bitfield. In that case, we
                # round up the end of the bitfield to the next byte, like clang
                # and GCC do.
                first_bit_offset = _next_multiple(
                    prev_member.bit_offset + prev_member.bits,
                    8,
                )
                assert first_bit_offset <= bit_offset
                padding = abs(bit_offset - first_bit_offset)
                assert not (padding % 8)
                padding = padding // 8

                if padding:
                    padding = BTFPaddingMember(
                        # The name must be unique inside the struct
                        # (self._index) but also unique among sibling structs,
                        # as ISO C allows anonymous structs inside a top-level
                        # struct, behaving as if their content was inlined in
                        # the parent struct.
                        name=f'____PADDING_{self.parent.id}_{self._index}',
                        size=padding,
                        bit_offset=first_bit_offset,
                    )
                    return (padding,)
                else:
                    return ()
            else:
                # We are the first member, nothing to align
                return ()

    def _dump_c_introspection(self, ctx):
        typ = self.typ

        ctx.typ_member(self.name)

        # If the member type is a named struct/union, it is to be treated just
        # like any other top-level type, so we need reset the parent chain.
        if isinstance(typ, _BTFStructUnion) and typ.name:
            ctx_mgr = ctx.with_parent(None)
        else:
            ctx_mgr = contextlib.nullcontext(ctx)

        with ctx_mgr as ctx:
            self.typ._dump_c_introspection(ctx)


class BTFStruct(_BTFStructUnion):
    __slots__ = ('_alignment',)
    _KIND = 'struct'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alignment = None

    @property
    def _all_members(self):
        members = self.members
        if members:
            return [
                _member
                for member in self.members
                for _member in (*member.pre_padding, member)
            ]
        # Some structs only contain anonymous bitfields, so they have a
        # non-zero size but no member recorded in BTF
        elif size := self.size:
            return [
                BTFPaddingMember(
                    name=f'____PADDING_{self.id}_0',
                    size=size,
                    bit_offset=0,
                )
            ]
        else:
            return []

    @property
    def _max_alignment(self):
        members = self.members
        if members:
            min_alignment = self._min_alignment
            size = self.size
            min_size = self._min_size

            assert size >= min_size
            last_padding = abs(size - min_size)
            max_alignment = 1 << last_padding.bit_length()
            return max(max_alignment, min_alignment)
        else:
            return 1

    @property
    def alignment(self):
        size = self.size
        max_alignment = self._max_alignment

        # We have something weird going on, like a struct manually padded using
        # anonymous bitfields at the end.
        if size % max_alignment:
            return 1
        else:
            alignment = self._alignment
            if alignment is None:
                # We have something weird going on, like a struct manually padded using
                # anonymous bitfields at the end.
                self._alignment = self._min_alignment if self.size % max_alignment else max_alignment
                return self._alignment
            else:
                return alignment

    def _align_attribute(self, parents):
        members = self.members
        size = self.size
        min_alignment = self._min_alignment
        max_alignment = self._max_alignment

        # We have something weird going on, like a struct manually padded using
        # anonymous bitfields at the end.
        if size % max_alignment:
            packed = 'packed'
            last = members[-1] if members else None
            min_size = self._min_size
            # We cannot add padding if the last member is a flexible array (or
            # any struct/typedef containing one).
            if last and last.is_incomplete:
                raise ValueError(f'Cannot add necessary padding to {self.fwd_decl}')
            else:
                assert size >= min_size
                last_padding = abs(size - min_size)

                id_ = '_'.join(str(parent.id) for parent in parents)
                padding = BTFPaddingMember(
                    # The name must be unique inside the struct
                    # (self._index) but also unique among sibling structs,
                    # as ISO C allows anonymous structs inside a top-level
                    # struct, behaving as if their content was inlined in
                    # the parent struct.
                    name=f'____PADDING_{id_}_{self.id}_{len(members) + 1}',
                    size=last_padding,
                    bit_offset=min_size * 8,
                )
                align = None
        else:
            alignment = self.alignment
            padding = None
            align = f'aligned({alignment})' if alignment > min_alignment else ''
            packed = None

        attrs = ','.join(attr for attr in (packed, align) if attr)
        return (
            f'__attribute__(({attrs}))' if attrs else None,
            padding
        )


class BTFUnion(_BTFStructUnion):
    _KIND = 'union'
    pre_padding = ()

    @property
    def _all_members(self):
        return self.members

    @property
    def alignment(self):
        return self._min_alignment

    def _align_attribute(self, parents):
        size = self.size
        id_ = '_'.join(str(parent.id) for parent in parents)
        return (
            '__attribute__((packed))' if size % self._min_alignment else None,
            # Always add a padding member in unions since some unions are
            # sometimes wrongly reported as having 0 members (vlen=0), probably
            # due to a pahole bug. When that is the case, the padding member
            # ensures the size of the type will be correct.
            BTFPaddingMember(
                # The name must be unique inside the struct
                # (self._index) but also unique among sibling structs,
                # as ISO C allows anonymous structs inside a top-level
                # struct, behaving as if their content was inlined in
                # the parent struct.
                name=f'____ENSURE_UNION_SIZE_{id_}_{self.id}_{len(self.members) + 1}',
                size=size,
                bit_offset=0,
            )
        )


class BTFTypedef(_TransparentType, _CDecl, BTFType):
    __slots__ = ('name', 'typ')
    _KIND = 'typedef'
    _TYP_ATTRS = ('typ',)

    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ

    def _dump_c_introspection(self, ctx):
        ctx.typ_exists(self)
        with ctx.with_parent(self) as ctx:
            self.typ._dump_c_introspection(ctx)

    def _do_dump_c_decls(self, ctx):
        name = self.name
        typ = self.typ
        typename = typ._dump_c_decls(ctx)

        # Some types are special for the compiler and it hates it being
        # re-typedefed, such as __builtin_va_list
        if not name.startswith('__builtin_'):
            ctx.write(
                f'typedef {typename} {name};\n'
            )
        return name


class BTFConst(_CDeclSpecifier, _CQualifier):
    _c_specifier = 'const'


class BTFVolatile(_CDeclSpecifier, _CQualifier):
    _c_specifier = 'volatile'


class BTFRestrict(_CDeclSpecifier, _CQualifier):
    _c_specifier = 'restrict'


class BTFArray(_FixupTyp, _CDecl, BTFType):
    __slots__ = ('typ', 'index_typ', 'nelems')
    _TYP_ATTRS = ('typ', 'index_typ')

    def __init__(self, typ, index_typ, nelems):
        super().__init__()
        self.typ = typ
        self.index_typ = index_typ
        self.nelems = nelems

    def _do_dump_c_decls(self, ctx):
        size = self.size
        name = ctx.make_name()
        typename = self.typ._dump_c_decls(ctx)
        # We can never emit flexible array member (e.g. int arr[]) but instead
        # we always emit zero-length-array (e.g. int arr[0]). Flexible array
        # members are part of the standard while zero-length-arrays are not,
        # but unfortunately they are encoded the same way, and it is forbidden
        # by the standard to have a flexible array member in a struct anywhere
        # else than the last position. Annoyingly, some structs in the kernel
        # use ZLA, which do not have this restriction (in GCC at least).
        array_size = self.nelems or 0
        ctx.write(
            f'typedef {typename} {name}[{array_size}];\n'
        )

        ctx.write(
            f'_Static_assert(sizeof({name}) == {size}ull, "{typename}[{array_size}] does not have expected size: {size}");\n'
        )
        return name

    def _dump_c_introspection(self, ctx):
        self.typ._dump_c_introspection(ctx)
        self.index_typ._dump_c_introspection(ctx)

    @property
    def size(self):
        return self.typ.size * self.nelems

    @property
    def bits(self):
        return self.size * 8

    @property
    def alignment(self):
        return self.typ.alignment

    @property
    def is_incomplete(self):
        # Technically this is not incomplete in the context of a BTFFuncProto
        # parameter, but we just made BTFFuncProto.is_complete always return
        # False
        return not self.nelems


class BTFPaddingMember(BTFMember):
    _ITEM_TYP = BTFInt(
        name='unsigned char',
        size=1,
        encoding=BTFIntEncoding.NONE,
        bit_offset=0,
        bits=8,
    )

    _INDEX_TYP = BTFInt(
        name='unsigned long long',
        size=8,
        encoding=BTFIntEncoding.NONE,
        bit_offset=0,
        bits=64,
    )

    @property
    def pre_padding(self):
        return ()

    def __init__(self, name, bit_offset, size):
        super().__init__(
            name=name,
            typ=self._array_typ(size),
            bits=size * 8,
            bit_offset=bit_offset,
        )

    @classmethod
    # Cache so that we don't re-emit thousands of times the exact same typedef,
    # since padding types would otherwise represent ~20% of the other types
    @functools.lru_cache(maxsize=1024)
    def _array_typ(cls, size):
        return BTFArray(
            typ=cls._ITEM_TYP,
            index_typ=cls._INDEX_TYP,
            nelems=size,
        )


class BTFFuncProto(_FixupTyp, _CDecl, BTFType):
    __slots__ = ('typ', '_params')
    _ITERABLE_TYP_ATTRS = ('_params',)

    def __init__(self, typ, params):
        super().__init__()
        self.typ = typ
        self._params = params

    def _do_dump_c_decls(self, ctx):
        ret_typ = self.typ
        params = self.params
        name = ctx.make_name()

        ret_typename = ret_typ._dump_c_decls(ctx)
        params = ', '.join(
            str(param.typ._dump_c_decls(ctx))
            for param in params
        )

        ctx.write(
            f'typedef {ret_typename} ({name})({params});\n'
        )

        return name

    def _dump_c_introspection(self, ctx):
        self.typ._dump_c_introspection(ctx)
        for param in self._params:
            param.typ._dump_c_introspection(ctx)

    @property
    def params(self):
        return self._params or [
            BTFParam(name=None, typ=BTFVoid())
        ]


class BTFParam(_TransparentType, BTFType):
    __slots__ = ('name', 'typ')

    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ


class _BTFEllipsis(_CBuiltin, BTFType):
    name = '...'


class BTFVarArgParam(BTFParam):
    def __init__(self):
        super().__init__(
            name='...',
            typ=_BTFEllipsis()
        )


class BTFForwardDecl(_CDecl, BTFType):
    __slots__ = ('name', 'typ_cls')

    def __init__(self, name, typ_cls):
        super().__init__()
        self.name = name
        self.typ_cls = typ_cls

    def _do_dump_c_decls(self, ctx):
        name = self.name
        kind = self.typ_cls._KIND
        fwd_decl = f'{kind} {name}'
        ctx.write(f'{fwd_decl};\n')
        return fwd_decl


class _LinkedForwardDecl(BTFForwardDecl):
    __slots__ = ('parent',)
    def __init__(self, parent):
        self.parent = parent

    @property
    def name(self):
        return self.parent.name

    @property
    def typ_cls(self):
        return self.parent.__class__


class BTFFunc(_FixupTyp,  BTFType):
    __slots__ = ('name', 'typ', 'linkage')

    def __init__(self, name, typ, linkage):
        super().__init__()
        self.name = name
        self.typ = typ
        self.linkage = linkage

    def _dump_c_introspection(self, ctx):
        self.typ._dump_c_introspection(ctx)

    def _do_dump_c_decls(self, ctx):
        pass


class BTFFuncLinkage(enum.Enum):
    STATIC = 0
    GLOBAL = 1
    EXTERN = 2

    @classmethod
    def from_int(cls, v):
        if v == 0:
            return cls.STATIC
        elif v == 1:
            return cls.GLOBAL
        elif v == 2:
            return cls.EXTERN
        else:
            raise ValueError(f'Unknown BTF function linkage: {v}')


class BTFVar(_TransparentType, BTFType):
    __slots__ = ('name', 'typ', 'linkage')

    def __init__(self, name, typ, linkage):
        super().__init__()
        self.name = name
        self.typ = typ
        self.linkage = linkage


class BTFVarLinkage(enum.Enum):
    STATIC = 0
    GLOBAL = 1

    @classmethod
    def from_int(cls, v):
        if v == 0:
            return cls.STATIC
        elif v == 1:
            return cls.GLOBAL
        else:
            raise ValueError(f'Unknown BTF variable linkage: {v}')


class BTFDataSec(BTFType):
    __slots__ = ('name', 'size', 'variables')
    _ITERABLE_TYP_ATTRS = ('variables',)

    def __init__(self, name, size, variables):
        super().__init__()
        self.name = name
        self.size = size
        self.variables = variables

    @property
    def bits(self):
        return self.size * 8


class BTFVarSecInfo(_FixupTyp, BTFType):
    __slots__ = ('typ', 'offset', 'size')

    def __init__(self, typ, offset, size):
        super().__init__()
        self.typ = typ
        self.size = size
        self.offset = offset

    @property
    def bits(self):
        return self.size * 8


class BTFAttribute:
    __slots__ = ('name', 'value')

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        value = self.value

        if value is None:
            value = ''
        return f'{self.name}({value})'

    @classmethod
    def from_tag(cls, tag, is_normal_attribute, special_name):
        if is_normal_attribute:
            if (m := re.match(r'(?P<name>.*?)\((?P<value>.*)\)', tag)):
                return cls(
                    name=m.group('name'),
                    value=m.group('value')
                )
            else:
                return cls(
                    name=tag,
                    value=None,
                )
        else:
            return cls(
                name=special_name,
                value=json.dumps(str(tag)),
            )


class BTFDeclTag(_TransparentType, _CDecl, BTFType):
    __slots__ = ('attribute', 'typ', 'component_idx')

    def __init__(self, typ, tag, component_idx, is_normal_attribute):
        super().__init__()
        self.attribute = BTFAttribute.from_tag(
            tag=tag,
            is_normal_attribute=is_normal_attribute,
            special_name='btf_decl_tag',
        )
        self.typ = typ
        self.component_idx = component_idx

    def _do_dump_c_decls(self, ctx, **kwargs):
        # TODO: actually add __attribute__((the btf_decl_tag()))
        return self.typ._do_dump_c_decls(ctx, **kwargs)


class BTFTypeTag(_CDeclSpecifier):
    __slots__ = ('attribute', 'typ', 'is_normal_attribute')

    def __init__(self, typ, tag, is_normal_attribute):
        super().__init__(typ=typ)
        self.attribute = BTFAttribute.from_tag(
            tag=tag,
            is_normal_attribute=is_normal_attribute,
            special_name='btf_type_tag',
        )

    @property
    def _c_specifier(self):
        return f'__attribute__(({self.attribute}))'


def _parse_btf(buf):
    # Creating a memoryview() allows a copy-less slicing operation, so we can
    # manipulate slices just as buffers without having to pay the astronomical
    # cost of copying data every time (which would otherwise make this parser
    # O(N^2) with the input size).
    buf = memoryview(buf).toreadonly()

    magic, = struct.unpack_from('<H', buf)
    if magic == 0xeb9f:
        little_endian = True
    elif magic == 0x9feb:
        little_endian = False
    else:
        raise ValueError(f'Not a BTF binary blob, invalid magic: {magic:02x}')

    buf = buf[2:]

    def make_decode(fmt, array):
        fmt = f'<{fmt}' if little_endian else f'>{fmt}'

        decoder = struct.Struct(fmt)
        size = decoder.size

        if array:
            decoder = decoder.iter_unpack
            def decode(buf, vlen):
                total_size = size * vlen
                _buf = buf[:total_size]
                return (
                    buf[total_size:],
                    decoder(_buf)
                )
        else:
            decoder = decoder.unpack_from
            def decode(buf):
                return (
                    buf[size:],
                    decoder(buf)
                )
        return decode

    decode_B = make_decode('B', array=False)
    decode_BI = make_decode('BI', array=False)
    decode_I = make_decode('I', array=False)
    decode_III = make_decode('III', array=False)
    decode_IIII = make_decode('IIII', array=False)

    decode_array_II = make_decode('II', array=True)
    decode_array_III = make_decode('III', array=True)
    decode_array_Ii = make_decode('II', array=True)


    buf, (version,) = decode_B(buf)
    if version != 1:
        raise ValueError(f'BTF version {version} not supported')


    buf, (flags, hdr_len) = decode_BI(buf)

    _, (type_off, type_len, str_off, str_len) = decode_IIII(buf)

    # We already parsed 8 bytes of the header
    assert hdr_len >= 8
    data = buf[hdr_len - 8:]

    string_section = data[str_off:str_off + str_len]
    _strings = string_section.tobytes().split(b'\x00')


    i = 0
    strings = {}
    for s in _strings:
        strings[i] = s.decode('utf-8')
        i += len(s) + 1

    assert strings[0] == ''

    def resolve_name(i):
        return strings[i] or None

    def parse_type(buf):
        buf, (name_off, info, size_or_type) = decode_III(buf)

        name = resolve_name(name_off)

        vlen = info & 0xffff
        # The doc is wrong: it states that bits 24-28 encode the BTF_KIND_*,
        # but the values go beyond 16 these days so they effectively used the 2
        # subsequent bits.
        kind = (info & (0b111111 << 24)) >> 24
        kind_flag = (info & (0b1 << 31)) >> 31

        if kind == 0:
            typ = BTFVoid()

        #define BTF_KIND_INT            1       /* Integer      */
        elif kind == 1:
            assert kind_flag == 0
            assert vlen == 0

            buf, (meta,) = decode_I(buf)

            #define BTF_INT_ENCODING(VAL)   (((VAL) & 0x0f000000) >> 24)
            encoding = (meta & 0x0f000000) >> 24
            #define BTF_INT_OFFSET(VAL)     (((VAL) & 0x00ff0000) >> 16)
            offset = (meta & 0x00ff0000) >> 16
            #define BTF_INT_BITS(VAL)       ((VAL)  & 0x000000ff)
            bits = meta & 0x000000ff

            encoding = BTFIntEncoding.from_int(encoding)

            typ = BTFInt(
                name=name,
                size=size_or_type,
                encoding=encoding,
                bit_offset=offset,
                bits=bits
            )

        #define BTF_KIND_PTR            2       /* Pointer      */
        elif kind == 2:
            assert kind_flag == 0
            assert vlen == 0
            assert name is None

            typ = BTFPtr(
                typ=_TypeRef(size_or_type),
                size=None,
            )

        #define BTF_KIND_ARRAY          3       /* Array        */
        elif kind == 3:
            assert name is None
            assert kind_flag == 0
            assert vlen == 0
            assert size_or_type == 0

            buf, (typ, index_typ, nelems) = decode_III(buf)

            typ = BTFArray(
                typ=_TypeRef(typ),
                index_typ=_TypeRef(index_typ),
                nelems=nelems,
            )

        #define BTF_KIND_STRUCT         4       /* Struct       */
        #define BTF_KIND_UNION          5       /* Union        */
        elif kind == 4 or kind == 5:

            if kind_flag == 0:
                decode_offset = lambda offset: offset
                decode_bits = lambda offset: 0
            elif kind_flag == 1:
                #define BTF_MEMBER_BIT_OFFSET(val)      ((val) & 0xffffff)
                decode_offset = lambda offset: offset & 0xffffff
                #define BTF_MEMBER_BITFIELD_SIZE(val)   ((val) >> 24)
                decode_bits = lambda offset: offset >> 24
            else:
                raise ValueError(f'Unknown info.kind_flag == {kind_flag} for BTF_KIND_STRUCT/BTF_KIND_UNION')

            buf, members = decode_array_III(buf, vlen)
            members = [
                BTFMember(
                    name=resolve_name(name_off),
                    typ=_TypeRef(typ),
                    bit_offset=decode_offset(offset),
                    bits=decode_bits(offset),
                )
                for name_off, typ, offset in members
            ]

            typ = (BTFStruct if kind == 4 else BTFUnion)(
                name=name,
                size=size_or_type,
                members=members,
            )

        #define BTF_KIND_ENUM64         19      /* Enumeration up to 64-bit values */
        elif kind == 19:
            size = size_or_type
            signed = BTFSignedness.from_int(kind_flag)
            nr_enumerators = vlen
            assert size in (1, 2, 4, 8)

            def cast_value(v_low, v_high):
                v = (v_high << 32) | v_low

                if signed == BTFSignedness.SIGNED and (v & (1 << (64 - 1))):
                    v = v - (1 << 64)

                return v

            buf, enumerators = decode_array_III(buf, vlen)
            enumerators = {
                resolve_name(name_off): cast_value(v_low, v_high)
                for name_off, v_low, v_high in enumerators
            }

            if enumerators:
                typ = BTFEnum(
                    name=name,
                    signed=signed,
                    size=size,
                    enumerators=enumerators,
                    int_typ=None,
                )
            # Forward decl of enum is a GNU extension, and seems to be encoded
            # with an enum with vlen == 0
            else:
                typ = BTFForwardDecl(
                    name=name,
                    typ_cls=BTFEnum,
                )

        #define BTF_KIND_ENUM           6       /* Enumeration up to 32-bit values */
        elif kind == 6:
            size = size_or_type
            signed = BTFSignedness.from_int(kind_flag)
            nr_enumerators = vlen
            assert size in (1, 2, 4, 8)

            buf, enumerators = decode_array_Ii(buf, vlen)
            enumerators = {
                resolve_name(name_off): v
                for name_off, v in enumerators
            }

            if enumerators:
                typ = BTFEnum(
                    name=name,
                    signed=signed,
                    size=size,
                    enumerators=enumerators,
                    int_typ=None,
                )
            # Forward decl of enum is a GNU extension, and seems to be encoded
            # with an enum with vlen == 0
            else:
                typ = BTFForwardDecl(
                    name=name,
                    typ_cls=BTFEnum,
                )

        #define BTF_KIND_FWD            7       /* Forward      */
        elif kind == 7:
            assert vlen == 0
            assert size_or_type == 0

            if kind_flag == 0:
                typ_cls = BTFStruct
            elif kind_flag == 1:
                typ_cls = BTFUnion
            else:
                raise ValueError(f'Unknown info.kind_flag value for BTF_KIND_FWD: {kind_flag}')

            typ = BTFForwardDecl(name=name, typ_cls=typ_cls)

        #define BTF_KIND_TYPEDEF        8       /* Typedef      */
        elif kind == 8:
            assert kind_flag == 0
            assert vlen == 0
            typ = BTFTypedef(
                name=name,
                typ=_TypeRef(size_or_type),
            )

        #define BTF_KIND_VOLATILE       9       /* Volatile     */
        elif kind == 9:
            assert name is None
            assert kind_flag == 0
            assert vlen == 0
            typ = BTFVolatile(typ=_TypeRef(size_or_type))

        #define BTF_KIND_CONST          10      /* Const        */
        elif kind == 10:
            assert name is None
            assert kind_flag == 0
            assert vlen == 0
            typ = BTFConst(typ=_TypeRef(size_or_type))

        #define BTF_KIND_RESTRICT       11      /* Restrict     */
        elif kind == 11:
            assert name is None
            assert kind_flag == 0
            assert vlen == 0
            typ = BTFRestrict(typ=_TypeRef(size_or_type))

        #define BTF_KIND_FUNC           12      /* Function     */
        elif kind == 12:
            assert kind_flag == 0

            typ = BTFFunc(
                name=name,
                linkage=BTFFuncLinkage.from_int(vlen),
                typ=_TypeRef(size_or_type),
            )

        #define BTF_KIND_FUNC_PROTO     13      /* Function Proto       */
        elif kind == 13:
            assert name is None
            assert kind_flag == 0

            buf, params = decode_array_II(buf, vlen)
            params = [
                BTFParam(
                    name=resolve_name(name_off),
                    typ=_TypeRef(typ),
                )
                for name_off, typ in params
            ]

            if params and params[-1].name is None and params[-1].typ.index == 0:
                params[-1] = BTFVarArgParam()

            typ = BTFFuncProto(
                typ=_TypeRef(size_or_type),
                params=params,
            )

        #define BTF_KIND_VAR            14      /* Variable     */
        elif kind == 14:
            assert kind_flag == 0
            assert vlen == 0

            buf, (linkage,) = decode_I(buf)

            typ = BTFVar(
                name=name,
                typ=_TypeRef(size_or_type),
                linkage=BTFVarLinkage.from_int(linkage),
            )

        #define BTF_KIND_DATASEC        15      /* Section      */
        elif kind == 15:
            assert kind_flag == 0

            buf, variables = decode_array_III(buf, vlen)
            variables = [
                BTFVarSecInfo(
                    typ=_TypeRef(typ),
                    offset=offset,
                    size=size,
                )
                for typ, offset, size in variables
            ]

            typ = BTFDataSec(
                name=name,
                size=size_or_type,
                variables=variables,
            )

        #define BTF_KIND_FLOAT          16      /* Floating point       */
        elif kind == 16:
            size = size_or_type
            assert kind_flag == 0
            assert vlen == 0
            assert size in (2, 4, 8, 12, 16)

            typ = BTFFloat(
                name=name,
                size=size,
            )

        #define BTF_KIND_DECL_TAG       17      /* Decl Tag     */
        elif kind == 17:
            assert kind_flag == 0
            assert vlen == 0

            buf, (component_idx,) = decode_I(buf)
            component_idx = None if component_idx == -1 else component_idx

            typ = BTFDeclTag(
                tag=name,
                typ=_TypeRef(size_or_type),
                component_idx=component_idx,
                is_normal_attribute=kind_flag == 1,
            )

        #define BTF_KIND_TYPE_TAG       18      /* Type Tag     */
        elif kind == 18:
            assert name
            assert vlen == 0

            typ = BTFTypeTag(
                tag=name,
                typ=_TypeRef(size_or_type),
                is_normal_attribute=kind_flag == 1,
            )

        else:
            raise ValueError(f'Unknown BTF kind: {kind}')

        return (buf, typ)


    type_section = data[type_off:type_off + type_len]
    typs = _scan(type_section, parse_type)

    # The type at index 0 is by definition void, and all indices are shifted by
    # this.
    typs.insert(0, BTFVoid())

    fixup_refs = lambda typ: _TypeRef.fixup(typ, typs)
    fixup_visited = set()
    for i, typ in enumerate(typs):
        typ.id = i
        typ._map_typs(fixup_refs, visited=fixup_visited)

    def is_ptr_sized(typ):
        if isinstance(typ, BTFTypedef):
            return typ.name in ('intptr_t', 'uintptr_t', 'ptrdiff_t')
        elif isinstance(typ, BTFInt):
            return typ.name in ('long int', 'long unsigned int', 'long', 'unsigned long', 'long unsigned')
        else:
            return False

    uintptr_t = None
    int_sizes = {}
    for typ in typs:
        # Look for typedefs of int types. Something like "uint16_t" should show up in that list
        if isinstance(typ, BTFTypedef) and isinstance(underlying := typ.underlying, BTFInt) and not underlying.is_bitfield:
            int_sizes.setdefault(typ.size, []).append(typ)

        if is_ptr_sized(typ):
            uintptr_t = typ

    # Give priority to standard types
    fixed_size_re = re.compile(r'u?int[0-9]+_t$')
    def select_int(typs):
        for typ in typs:
            if fixed_size_re.match(typ.name):
                return typ

        return typs[0]

    int_sizes = {
        size: select_int(typs)
        for size, typs in int_sizes.items()
    }

    if uintptr_t:
        ptr_size = uintptr_t.size
    else:
        raise ValueError(f'Could not find pointer-sized type in BTF types')

    for typ in typs:
        if isinstance(typ, BTFPtr):
            typ.size = ptr_size
        elif isinstance(typ, BTFEnum):
            typ.int_typ = int_sizes.get(typ.size, None)

    return typs


def parse_btf(buf, select_typ=None, rename_typ=None):
    typs = _parse_btf(buf)

    if select_typ:
        typs = list(filter(select_typ, typs))

        reachable_typs = BTFType.reachable_from(typs)
    else:
        reachable_typs = set(typs)

    if rename_typ:
        for typ in reachable_typs:
            if isinstance(typ, (BTFStruct, BTFUnion, BTFEnum, BTFTypedef, BTFFunc, BTFForwardDecl)) and typ.name:
                typ.name = rename_typ(typ)

    _dedup_names(reachable_typs)

    return typs


class _DeclCtx:
    def __init__(self, fileobj):
        self._i = 0
        self._memo = {}
        self._fileobj = fileobj

    def write(self, x):
        self._fileobj.write(x)

    def make_name(self):
        self._i += 1
        return f'___BTF_HEADER_INTERNAL_TYPE_{self._i}'


class _IntrospectionCtx:
    def __init__(self, fileobj):
        self._exists_memo = set()
        self._members_memo = set()
        self._fileobj = fileobj
        self._parent_typs = tuple()

    @contextlib.contextmanager
    def with_parent(self, typ):
        old = self._parent_typs
        try:
            if typ is None:
                self._parent_typs = ()
            else:
                self._parent_typs = (*self._parent_typs, typ)

            yield self
        finally:
            self._parent_typs = old

    def typ_exists(self, typ):
        name = typ.name
        key = typ
        memo = self._exists_memo

        if name and key not in memo:
            memo.add(key)
            self._write(f'#define _TYPE_EXISTS_{typ._KIND}_{typ.name} 1\n')

    def typ_member(self, member_name):
        if member_name:
            memo = self._members_memo

            for typ in self._parent_typs:
                key = (typ, member_name)
                if key not in memo:
                    memo.add(key)
                    if (typ_name := typ.name):
                        self._write(f'#define _TYPE_HAS_MEMBER_{typ._KIND}_{typ_name}_LISA_SEPARATOR_{member_name} 1\n')

    def _write(self, x):
        self._fileobj.write(x)


def _dedup_names(typs):
    def dedup_typ_names(name, typs):
        for typ in sorted(typs, key=attrgetter('id')):
            typ.name = f'___DEDUP_{typ.id}_{name}'

    def dedup_enumerator_names(name, typs):
        for typ in sorted(typs, key=attrgetter('id')):
            # Preserving the enumerator orders is not necessary since we have
            # explicit value for all of them in BTF
            value = typ.enumerators.pop(name)
            new_name = f'___DEDUP_{typ.id}_{name}'
            typ.enumerators[new_name] = value

    typedef_names = {}
    tagged_names = {}
    fwd_decl_names = {}
    enumerators = {}
    for typ in typs:
        if isinstance(typ, BTFTypedef):
            cat = typedef_names
        # BTFForwardDecl are renamed independently from the type they declare
        # since we don't know what they logically point to. It could be any of
        # the types that share that name, or yet another unknown. Fortunately,
        # they are kind of useless since we will create any actually needed
        # forward decl when dumping C code.
        elif isinstance(typ, (BTFStruct, BTFUnion, BTFEnum)):
            cat = tagged_names
        # We still dedup names there, in case they end up being printed and
        # there is e.g. "union foo;" and "struct foo;"
        elif isinstance(typ, BTFForwardDecl):
            cat = fwd_decl_names
        else:
            continue

        name = typ.name

        if name:
            try:
                _typs = cat[name]
            except KeyError:
                _typs = []
                cat[name] = _typs

            _typs.append(typ)

        if isinstance(typ, BTFEnum):
            for enumerator in typ.enumerators:
                try:
                    _typs = enumerators[enumerator]
                except KeyError:
                    _typs = []
                    enumerators[enumerator] = _typs

                _typs.append(typ)

    for dup_names in (typedef_names, tagged_names, fwd_decl_names):
        for name, _typs in dup_names.items():
            if len(_typs) > 1:
                dedup_typ_names(name, _typs)

    for enumerator, _typs in enumerators.items():
        if len(_typs) > 1:
            dedup_enumerator_names(enumerator, _typs)


def dump_c(typs, fileobj=None, decls=True, introspection=True):
    fileobj = fileobj or io.StringIO()

    introspection_ctx = _IntrospectionCtx(fileobj)
    decl_ctx = _DeclCtx(fileobj)

    for typ in typs:
        # Dump type declarations
        if decls and isinstance(typ, _CDecl) and not isinstance(typ, _CBuiltin):
            typ._dump_c_decls(decl_ctx)

        # Dump introspection macros
        if introspection:
            typ._dump_c_introspection(introspection_ctx)


    if introspection:
        fileobj.write('#define _TYPE_INTROSPECTION_INFO_AVAILABLE 1\n')

    fileobj.write('\n')

    return fileobj

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
