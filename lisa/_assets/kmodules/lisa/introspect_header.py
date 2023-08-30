#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

from pycparser import c_parser, c_ast


class Record(abc.ABC):
    @abc.abstractmethod
    def make_define(self):
        pass


class TypeMemberRecord(namedtuple('TypeMemberRecord', ['type_name', 'member_name']), Record):
    def make_define(self):
        return f'#define _TYPE_HAS_MEMBER_{self.type_name}___{self.member_name}'

    def is_printable(self):
        return self.type_name and self.member_name


class TypeExistsRecord(namedtuple('TypeExistsRecord', ['type_name']), Record):
    def make_define(self):
        return f'#define _TYPE_EXISTS_{self.type_name}'

    def is_printable(self):
        return bool(self.type_name)


def expand_typ(typ):
    if isinstance(typ, c_ast.TypeDecl):
        return expand_typ(typ.type)
    else:
        return [typ]


def resolve_typ(typ):
    if isinstance(typ, (c_ast.Struct, c_ast.Union)):
        children = typ.decls or []
        children_typs = [
            child.type
            for child in children
        ]
        name = typ.name
    elif isinstance(typ, c_ast.Enum):
        children = typ.values or []
        children_typs = []
        name = typ.name
    elif isinstance(typ, c_ast.TypeDecl):
        name = typ.declname
        _, children, children_typs = resolve_typ(typ.type)
    else:
        raise ValueError('Unhandled type')

    return (name, children, children_typs)


def walk_type(typ):
    try:
        name, children, children_typs = resolve_typ(typ)
    except ValueError:
        return []
    else:
        children_typs = itertools.chain.from_iterable(map(expand_typ, children_typs))

        return itertools.chain(
            [
                TypeExistsRecord(type_name=name)
            ],
            (
                TypeMemberRecord(type_name=name, member_name=child.name)
                for child in children
            ),
            itertools.chain.from_iterable(map(walk_type, children_typs))
        )


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

    types = [
        _node.type
        for _node in node
        if isinstance(_node, (c_ast.Decl, c_ast.Typedef, c_ast.TypeDecl))
    ]

    records = set(itertools.chain.from_iterable(map(walk_type, types)))
    return itertools.chain(
        (
            '#define _TYPE_INTROSPECTION_INFO_AVAILABLE',
        ),
        (
            record.make_define()
            for record in records
            if record.is_printable()
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
        out.extend(process_header(args.header))

    if args.kallsyms:
        out.extend(process_kallsyms(args.kallsyms))

    print('\n'.join(sorted(
        s
        for s in out
        if s
    )))

if __name__ == '__main__':
    main()
