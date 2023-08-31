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
import itertools
import argparse
import subprocess
from collections import namedtuple
import functools

from pycparser import c_parser, c_ast


class Record(abc.ABC):
    @abc.abstractmethod
    def make_define(self):
        pass


class TypeMemberRecord(namedtuple('TypeMemberRecord', ['type_kind', 'type_name', 'member_name']), Record):
    def make_define(self):
        return f'#define _TYPE_HAS_MEMBER_{self.type_kind}_{self.type_name}_LISA_SEPARATOR_{self.member_name}'


class TypeExistsRecord(namedtuple('TypeExistsRecord', ['type_kind', 'type_name']), Record):
    def make_define(self):
        return f'#define _TYPE_EXISTS_{self.type_kind}_{self.type_name}'


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

def process_header(path):
    with open(path, 'r') as f:
        header = f.read()

    # Remove comments and the non-standard GNU C extensions that pycparser cannot
    # process
    cmd = ['cpp', '-P', '-D__attribute__(x)=', '-']
    res = subprocess.run(cmd, input=header, capture_output=True, text=True, check=True)
    header = res.stdout

    parser = c_parser.CParser()
    node = parser.parse(header, filename=path)

    assert isinstance(node, c_ast.FileAST)

    def expand_decl(node):
        if isinstance(node, c_ast.Decl):
            return node.type
        else:
            return node

    types = [
        expand_decl(_node)
        for _node in node
    ]

    memo = TypedefMemo(types)
    records = make_records(memo, types)
    return itertools.chain(
        (
            '#define _TYPE_INTROSPECTION_INFO_AVAILABLE',
        ),
        (
            record.make_define()
            for record in records
        ),
    )


class SymbolRecord(namedtuple('SymbolRecord', ['name']), Record):
    def make_define(self):
        return f'#define _SYMBOL_EXISTS_{self.name}'


def process_kallsyms(path):
    with open(path, 'r') as f:
        kallsyms = f.read()

    def make_record(addr, code, name):
        # Uppercase codes are for STB_GLOBAL symbols, i.e. exported symbols.
        if code.isupper() and name.isidentifier():
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
                '#define _SYMBOL_INTROSPECTION_INFO_AVAILABLE',
            ),
        )
    # If the file is empty, we assume there is no kallsyms available
    else:
        return []


def main():
    parser = argparse.ArgumentParser("""
    Parse a header file and generate macros to allow compile-time introspection
    of types.
    """)

    parser.add_argument('--header', help='C header file to parse')
    parser.add_argument('--kallsyms', help='kallsyms content to parse')

    args = parser.parse_args()

    out = []
    if args.header:
        out.append(process_header(args.header))

    if args.kallsyms:
        out.append(process_kallsyms(args.kallsyms))

    for rec in sorted(set(itertools.chain.from_iterable(out))):
        print(rec)

if __name__ == '__main__':
    main()
