#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2023, Arm Limited and contributors.
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


import abc
import sys
import itertools
import argparse
from collections import deque, defaultdict
import functools
import json
import re

import lisa._btf as btf


def process_btf(out, path, introspect, internal_type_prefix, define_typ_names):
    internal_type_prefix = internal_type_prefix or ''
    with open(path, 'rb') as f:
        data = f.read()

    # TODO: this will dedup enumerators without trying to "elect" one to
    # preserve its original name. We need to make it so that types reachable
    # from define_typs don't have renamed enumerators if possible.

    # This will rename all types that have the same name. If it turns out that
    # one of those type is a type we asked for in define_typs, there is not
    # much we can do since we would not be able to know which one is to elect
    # as the one we want.
    typs = btf.parse_btf(data)

    if define_typ_names:
        define_typ_names = set(define_typ_names)
        def select(typ):
            if isinstance(typ, (btf.BTFStruct, btf.BTFUnion, btf.BTFEnum, btf.BTFTypedef)):
                return typ.name in define_typ_names
            else:
                return False

        define_typs = list(filter(select, typs))
    else:
        define_typs = set()

    # Dump introspection information for all types available in BTF
    btf.dump_c(typs, fileobj=out, introspection=introspect, decls=False)

    # Rename internal typs so we don't clash with included kernel headers
    reachable_typs = btf.BTFType.reachable_from(define_typs)

    named_classes = (
        btf.BTFStruct,
        btf.BTFUnion,
        btf.BTFEnum,
        btf.BTFTypedef,
        btf.BTFFunc,
        btf.BTFForwardDecl
    )
    def rename(name):
        return f'{internal_type_prefix}{name}'

    def rename_internal_typ(typ):
        name = typ.name
        if name in define_typ_names:
            return name
        else:
            return rename(name)

    tagged = {
        cls: defaultdict(set)
        for cls in (btf.BTFStruct, btf.BTFUnion, btf.BTFEnum)
    }
    fwd_decls = set()

    for typ in reachable_typs:
        for _cls, _typs in tagged.items():
            if isinstance(typ, _cls):
                _typs[typ.name].add(typ)

        if isinstance(typ, btf.BTFForwardDecl):
            fwd_decls.add(typ)
        elif isinstance(typ, btf.BTFEnum):
            typ.enumerators = {
                (
                    name
                    if typ.name in define_typ_names else
                    rename(name)
                ): value
                for name, value in typ.enumerators.items()
            }

    resolved = {}
    for typ in fwd_decls:
        for _cls, _typs in tagged.items():
            if issubclass(typ.typ_cls, _cls):
                try:
                    (target,) = _typs[typ.name]
                except (KeyError, ValueError):
                    pass
                else:
                    resolved[typ] = target
                break

    # Resolve all the forward declarations we can to the type they point to.
    # This prevents arbitrary breakage of code in pointer chains spanning
    # multiple structs that were not explicitly asked by the user.
    #
    # The main risk is e.g. "struct foo;" forward decl being matched with a
    # "struct foo {...};" that was done in another file and has nothing to do
    # with it. But as long as we only print a subset of kernel types, we are
    # unlikely to run into that, or at least less likely to run into that than
    # running into broken pointer chains because of forward decls.
    btf.BTFType.map_typs(lambda typ: resolved.get(typ, typ), reachable_typs)

    for typ in reachable_typs:
        if isinstance(typ, named_classes) and typ.name:
            typ.name = rename_internal_typ(typ)

    # Dump declaration for all the types we are interested in
    btf.dump_c(reachable_typs, fileobj=out, introspection=False, decls=True)


def is_exported_symbol(code):
    # Unfortunately, a symbol being STB_GLOBAL does not mean it is exported, so
    # we just scrap that address of all symbols and make a linker script for
    # the module.
    if code in ('U', ):
        return False
    elif code in ('u', 'v', 'w'):
        return True
    else:
        # Since we parse kallsyms, we have access to symbols even if they were
        # not global in the first place, so if we see it we can use it.
        return True or code.isupper()


def open_kallsyms(path):
    with open(path, 'r') as f:
        yield from (
            line.split(maxsplit=3)
            for line in map(str.strip, f)
            if line
        )


def process_kallsyms_introspection(path):
    has_records = False
    def make_record(addr, code, name, module=None):
        nonlocal has_records

        # Uppercase codes are for STB_GLOBAL symbols, i.e. exported symbols.
        if name.isidentifier() and is_exported_symbol(code):
            has_records = True
            return f'#define _SYMBOL_EXISTS_{name} 1\n'
        else:
            return None

    records = itertools.starmap(
        make_record,
        open_kallsyms(path),
    )
    return itertools.chain(
        filter(bool, records),
        # Make a lazy generator so that "has_records" is evaluated after the
        # records have been dumped.
        (
            rec
            for rec in ('#define _SYMBOL_INTROSPECTION_INFO_AVAILABLE 1\n',)
            if has_records
        ),
    )


def process_kallsyms_lds(path):
    def make_record(addr, code, name, module=None):
        if name.isidentifier():
            return f'PROVIDE({name} = 0x{addr});\n'
        else:
            return None

    return filter(
        bool,
        itertools.starmap(
            make_record,
            open_kallsyms(path),
        )
    )


def process_kernel_features(features):
    # Macros cannot be recursive, so we expand them manually.
    def expand(value):
        def replace(m):
            name = m.group(1)
            name = name.strip()
            value = features[name]
            expanded = expand(value)
            return f'({expanded})'

        return re.sub(r'HAS_KERNEL_FEATURE\(([a-zA-Z0-9_]+)\)', replace, value)

    features = {
        name: expand(value)
        for name, value in features.items()
    }

    features = list(features.items())
    names, values = zip(*features)

    names = ', '.join(
        f'"{name}"'
        for name in names
    )
    names = f'#define __KERNEL_FEATURE_NAMES {names}\n'

    values = ', '.join(
        f'({value})'
        for value in values
    )
    values = f'#define __KERNEL_FEATURE_VALUES {values}\n'

    return itertools.chain(
        (
            names,
            values
        ),
        (
            f'#define _KERNEL_HAS_FEATURE_{name} ({value})\n'
            for name, value in features
        )
    )


def main():
    parser = argparse.ArgumentParser("""
    Parse a BTF blob and generate macros to allow compile-time introspection
    of types.
    """)

    parser.add_argument('--btf', help='BTF blob to parse')

    parser.add_argument('--introspect', action='store_true', help='Create introspection macros for the given --btf or --kallsyms')
    parser.add_argument('--internal-type-prefix', help='Add the given prefix to the types found in --btf and dump the resulting renamed C header')

    parser.add_argument('--kallsyms', help='kallsyms content to parse')
    parser.add_argument('--conf', help='JSON configuration')
    parser.add_argument('--symbols-lds', action='store_true', help='Create a linker script with the content of --kallsyms')

    args = parser.parse_args()

    if args.internal_type_prefix and not args.btf:
        parser.error('--btf is required if --type-prefix is used')


    if args.conf:
        with open(args.conf, 'r') as f:
            conf = json.load(f)
    else:
        conf = dict()

    out = sys.stdout
    def dump_records(records):
        # Fastest way to consume an iterator
        deque(map(out.write, records), maxlen=0)

    try:
        if args.btf:
            process_btf(out, args.btf, args.introspect, args.internal_type_prefix, conf.get('btf-types', []))

        if args.kallsyms and args.introspect:
            dump_records(process_kallsyms_introspection(args.kallsyms))

        if kernel_features := conf.get('kernel-features'):
            dump_records(process_kernel_features(kernel_features))

        if args.kallsyms and args.symbols_lds:
            dump_records(process_kallsyms_lds(args.kallsyms))
    finally:
        out.flush()

if __name__ == '__main__':
    main()
