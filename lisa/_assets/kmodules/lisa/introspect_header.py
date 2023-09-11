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
import copy
import itertools
import argparse
import subprocess
from collections import namedtuple
import functools
import json
import re

from pycparser import c_ast
from pycparserext.ext_c_parser import GnuCParser
from pycparserext.ext_c_generator import GnuCGenerator


class Record(abc.ABC):
    @abc.abstractmethod
    def make_define(self):
        pass


class TypeMemberRecord(namedtuple('TypeMemberRecord', ['type_kind', 'type_name', 'member_name']), Record):
    def make_define(self):
        return f'#define _TYPE_HAS_MEMBER_{self.type_kind}_{self.type_name}_LISA_SEPARATOR_{self.member_name} 1'


class TypeExistsRecord(namedtuple('TypeExistsRecord', ['type_kind', 'type_name']), Record):
    def make_define(self):
        return f'#define _TYPE_EXISTS_{self.type_kind}_{self.type_name} 1'


class TypedefMemo:
    def __init__(self, types):
        self.complete_memo = {}
        self.resolved = {}

        # Seed the memo so that we know complete type defintions when typedef
        # is created based on a forward definition.
        def walk(typ):
            self.register_complete(typ)
            self.resolve_typ(typ)

        for typ in types:
            walk(typ)

    @staticmethod
    def _get_key(typ):
        if isinstance(typ, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
            return (typ.__class__, typ.name)
        else:
            raise ValueError('unhandled type')

    @staticmethod
    def _is_complete(typ):
        if isinstance(typ, (c_ast.Struct, c_ast.Union)):
            return typ.decls is not None
        elif isinstance(typ, c_ast.Enum):
            return typ.values is not None
        else:
            return True

    def register_complete(self, typ):
        try:
            key = self._get_key(typ)
        except ValueError:
            return
        else:
            # Only memoize complete definitions, i.e. not forward definitions
            if self._is_complete(typ):
                self.complete_memo[key] = typ

    def get_complete(self, typ):
        try:
            key = self._get_key(typ)
        except ValueError:
            return typ
        else:
            return self.complete_memo.get(key, typ)


    def resolve_typ(self, typ):
        try:
            return self.resolved[typ]
        except KeyError:
            pass

        recurse = self.resolve_typ

        if isinstance(typ, (c_ast.Struct, c_ast.Union)):
            def expand_child(decl):
                typ = decl.type
                # An struct/union anonymous member inside a struct/union is
                # expanded in its parent.
                if decl.name is None:
                    _, _, children, children_typs = recurse(typ)
                    return (children, list(children_typs) + [typ])
                else:
                    return ([decl], [typ.type])

            name = typ.name
            children = typ.decls

            if children:
                children, children_typs = map(itertools.chain.from_iterable, zip(*map(expand_child, children)))
            else:
                children = []
                children_typs = []

            if isinstance(typ, c_ast.Struct):
                kind = 'struct'
            elif isinstance(typ, c_ast.Union):
                kind = 'union'
            else:
                raise ValueError(f'Unhandled type: {typ}')

        elif isinstance(typ, c_ast.Enum):
            children = typ.values or []
            children_typs = []
            name = typ.name
            kind = 'enum'
        elif isinstance(typ, c_ast.Typedef):
            name = typ.name
            # Typedef is refering to a type by name, so look the full definition of
            # that type
            child_typ = self.get_complete(typ.type.type)
            _, _, children, children_typs = recurse(child_typ)
            kind = 'typedef'
        else:
            return (None, None, [], [])

        res = (kind, name, tuple(children), tuple(children_typs))
        self.resolved[typ] = res
        return res


def make_records(memo, types):

    def recurse_multi(types):
        return itertools.chain.from_iterable(
            map(recurse, types)
        )

    visited = set()
    def recurse(typ):
        if typ in visited:
            return tuple()
        else:
            visited.add(typ)

            kind, name, children, children_typs = memo.resolve_typ(typ)
            return itertools.chain(
                (
                    TypeExistsRecord(type_kind=kind, type_name=name),
                ) if name else tuple(),
                (
                    TypeMemberRecord(type_kind=kind, type_name=name, member_name=child.name)
                    for child in children
                    if name and child.name
                ),
                recurse_multi(children_typs)
            )

    return recurse_multi(types)


def introspect_header(ast):
    assert isinstance(ast, c_ast.FileAST)

    def expand_decl(node):
        if isinstance(node, c_ast.Decl):
            return node.type
        else:
            return node

    types = [
        expand_decl(node)
        for node in ast
    ]

    memo = TypedefMemo(types)
    records = make_records(memo, types)
    return itertools.chain(
        (
            '#define _TYPE_INTROSPECTION_INFO_AVAILABLE 1',
        ),
        (
            record.make_define()
            for record in records
        ),
    )


class TypeRenameVisitor(c_ast.NodeVisitor):
    def __init__(self, type_prefix, non_renamed):
        self.type_prefix = type_prefix
        self.names = {
            name: name
            for name in (non_renamed or [])
        }

    def _rename(self, name):
        if name:
            try:
                return self.names[name]
            except KeyError:
                new = f'{self.type_prefix}{name}'
                self.names[name] = new
                return new
        else:
            return name

    def visit_IdentifierType(self, node):
        node.names = [
            self.names.get(name, name)
            for name in node.names
        ]

    def visit_Typedef(self, node):
        def rename_decl(node, name):
            if isinstance(node, c_ast.TypeDecl):
                node.declname = name
            else:
                # Go through layers of PtrDecl, ArrayDecl etc
                return rename_decl(node.type, name)

        new = self._rename(node.name)
        node.name = new
        rename_decl(node.type, new)
        self.visit(node.type)

    def visit_Enum(self, node):
        node.name = self._rename(node.name)
        if node.values is not None:
            self.visit(node.values)

    def visit_Enumerator(self, node):
        node.name = self._rename(node.name)

    def _visit_StructUnion(self, node):
        node.name = self._rename(node.name)
        # Not a forward decl
        if node.decls is not None:
            self.visit(node.decls)

    def visit_Struct(self, node):
        self._visit_StructUnion(node)

    def visit_Union(self, node):
        self._visit_StructUnion(node)

    # pycparserext types added by:
    # https://github.com/inducer/pycparserext/pull/76
    visit_EnumExt = visit_Enum
    visit_StructExt = visit_Struct
    visit_UnionExt = visit_Union
    visit_EnumeratorExt = visit_Enumerator

def rename_types(ast, type_prefix, non_renamed):
    ast = copy.deepcopy(ast)
    TypeRenameVisitor(type_prefix, non_renamed).visit(ast)
    code = GnuCGenerator().visit(ast)
    return code


def process_header(path, introspect, type_prefix, non_renamed_types):
    with open(path, 'r') as f:
        header = f.read()

    if non_renamed_types:
        with open(non_renamed_types, 'r') as f:
            non_renamed_types = [name.strip() for name in f.read().splitlines()]
    else:
        non_renamed_types = []

    # Remove comments and the non-standard GNU C extensions that pycparser cannot
    # process
    cmd = ['cpp', '-P', '-']
    res = subprocess.run(cmd, input=header, capture_output=True, text=True, check=True)
    header = res.stdout

    parser = GnuCParser()
    ast = parser.parse(header, filename=path)

    return itertools.chain(
        introspect_header(ast) if introspect else [],
        [rename_types(ast, type_prefix, non_renamed_types)] if type_prefix else [],
    )


class SymbolRecord(namedtuple('SymbolRecord', ['name']), Record):
    def make_define(self):
        return f'#define _SYMBOL_EXISTS_{self.name} 1'


def is_global_symbol(code):
    if code in ('u', 'v', 'w'):
        return True
    elif code in ('U', ):
        return False
    else:
        return code.isupper()


def process_kallsyms(path):
    with open(path, 'r') as f:
        kallsyms = f.read()

    def make_record(addr, code, name):
        # Uppercase codes are for STB_GLOBAL symbols, i.e. exported symbols.
        if name.isidentifier() and is_global_symbol(code):
            return SymbolRecord(name=name).make_define()
        else:
            return None

    records = itertools.starmap(
        make_record,
        (
            line.split(maxsplit=2)
            for line in kallsyms.splitlines()
            if line
        )
    )
    records = set(record for record in records if record)

    if records:
        return itertools.chain(
            records,
            (
                '#define _SYMBOL_INTROSPECTION_INFO_AVAILABLE 1',
            ),
        )
    # If the file is empty, we assume there is no kallsyms available
    else:
        return []


def process_kernel_features(path):
    with open(path, 'r') as f:
        features = json.load(f)

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
    names = f'#define __KERNEL_FEATURE_NAMES {names}'

    values = ', '.join(
        f'({value})'
        for value in values
    )
    values = f'#define __KERNEL_FEATURE_VALUES {values}'

    return itertools.chain(
        (names, values),
        (
            f'#define _KERNEL_HAS_FEATURE_{name} ({value})'
            for name, value in features
        )
    )


def main():
    parser = argparse.ArgumentParser("""
    Parse a header file and generate macros to allow compile-time introspection
    of types.
    """)

    parser.add_argument('--header', help='C header file to parse')
    parser.add_argument('--introspect', action='store_true', help='Create introspection macros for the given --header or --kallsyms')

    parser.add_argument('--type-prefix', help='Add the given prefix to the types found in --header and dump the resulting renamed header')
    parser.add_argument('--non-renamed-types', help='File containing list of type names that will not be renamed by --type-prefix')
    parser.add_argument('--kallsyms', help='kallsyms content to parse')
    parser.add_argument('--kernel-features', help='JSON list of kernel features')

    args = parser.parse_args()

    if args.type_prefix and not args.header:
        parser.error('--header is required if --type-prefix is used')

    out = []
    if args.header:
        out.append(process_header(args.header, args.introspect, args.type_prefix, args.non_renamed_types))

    if args.kallsyms and args.introspect:
        out.append(process_kallsyms(args.kallsyms))

    if args.kernel_features:
        out.append(process_kernel_features(args.kernel_features))

    for rec in sorted(set(itertools.chain.from_iterable(out))):
        print(rec)

if __name__ == '__main__':
    main()
