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


import itertools
import argparse
import subprocess
from collections import namedtuple

from pycparser import c_parser, c_ast


class TypeMember(namedtuple('TypeMember', ['type_name', 'member_name'])):
    def make_define(self):
        return f'#define _TYPE_HAS_MEMBER_{self.type_name}___{self.member_name}'

    @property
    def printable(self):
        return self.type_name and self.member_name


class TypeExists(namedtuple('TypeExists', ['type_name'])):
    def make_define(self):
        return f'#define _TYPE_EXISTS_{self.type_name}'

    @property
    def printable(self):
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
                TypeExists(type_name=name)
            ],
            (
                TypeMember(type_name=name, member_name=child.name)
                for child in children
            ),
            itertools.chain.from_iterable(map(walk_type, children_typs))
        )


def main():
    parser = argparse.ArgumentParser("""
    Parse a header file and generate macros to allow compile-time introspection
    of types.
    """)

    parser.add_argument('header', help='C header file to parse')

    args = parser.parse_args()
    with open(args.header, 'r') as f:
        header = f.read()

    # Remove comments and the non-standard GNU C extensions that pycparser cannot
    # process
    cmd = ['cpp', '-P', '-D__attribute__(x)=', '-']
    res = subprocess.run(cmd, input=header, capture_output=True, text=True, check=True)
    header = res.stdout

    parser = c_parser.CParser()
    node = parser.parse(header, filename=args.header)

    assert isinstance(node, c_ast.FileAST)

    types = [
        _node.type
        for _node in node
        if isinstance(_node, (c_ast.Decl, c_ast.Typedef, c_ast.TypeDecl))
    ]

    records = set(itertools.chain.from_iterable(map(walk_type, types)))
    records = sorted(
        record
        for record in records
        if record.printable
    )
    macros = '\n'.join(
        record.make_define()
        for record in records
    )
    print('#define _TYPE_INTROSPECTION_INFO_AVAILABLE')
    print(macros)



main()
