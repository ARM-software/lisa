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

import argparse
from pathlib import Path
from shlex import quote
import subprocess
import os
import textwrap
import json
from functools import cache
from operator import itemgetter
import textwrap
import itertools
import json
import re


SEP = '\n  '


def parse_json(path):
    content = Path(path).read_text()
    print(f'JSON data:\n{content}')
    return [
        json.loads(line)
        for line in content.splitlines()
    ]



def main():
    parser = argparse.ArgumentParser("""
    Process the Rust object file and associated JSON data.
    """)

    parser.add_argument('--rust-object', help='Built Rust object file')
    parser.add_argument('--json', help='JSON data extracted from the Rust object file')
    parser.add_argument('--out-symbols-plain', help='File to write the exported symbol list, one per line')
    parser.add_argument('--out-symbols-cli', help='File to write the exported symbol list as ld CLI --undefined options')
    parser.add_argument('--out-start-stop-lds', help='File to write the linker script to provide __start_SECNAME and __stop_SECNAME symbols, which are not provided by Kbuild since the module is never linked into an executable or DSO')
    parser.add_argument('--out-trace-events-header', help='File to write the ftrace event definitions')

    args = parser.parse_args()

    @cache
    def get_symbols():
        nm = os.environ.get('NM', 'nm')
        path = args.rust_object
        if path is None:
            parser.error(f'--rust-object needs to be specified to get symbols list')
        else:
            out = subprocess.check_output(
                [nm, '-gj', str(path)],
            )

            all_symbols = sorted(set(out.decode().split()))
            return all_symbols

    @cache
    def get_data():
        path = args.json
        if path is None:
            parser.error(f'--json needs to be specified for JSON-related features')
        else:
            return parse_json(path)

    @cache
    def get_exported_symbols():
        # We create a JSON entry for each symbol we want to export in Rust. There
        # unfortunately seems to be no cleaner way to convey the list of exported
        # symbols from Rust code as of 2024.
        user_symbols = [
            entry["symbol"]
            for entry in get_data()
            if entry["type"] == "export-symbol"
        ]
        symbols = sorted(user_symbols)
        print(f'Found exported symbols:{SEP}{SEP.join(symbols)}')
        return symbols

    if (path := args.out_symbols_plain):
        content = '\n'.join(get_exported_symbols()) + '\n'
        Path(path).write_text(content)

    if (path := args.out_symbols_cli):
        content = ' '.join((
            f'--undefined {quote(sym)}'
            for sym in get_exported_symbols()
        )) + '\n'
        Path(path).write_text(content)

    if (path := args.out_start_stop_lds):
        def parse(sym, prefix):
            if sym.startswith(prefix):
                sym = sym[len(prefix):]
                return sym
            else:
                return None

        sections = sorted({
            section
            for sym in get_symbols()
            if (
                (section := parse(sym, '__start_')) or
                (section := parse(sym, '__stop_'))
            )
        })
        print(f'Found sections to generate __start_SECNAME and __stop_SECNAME symbols for:{SEP}{SEP.join(sections)}')

        def make_lds(section):
            return textwrap.dedent(f'''
                SECTIONS {{
                    {section} : {{
                        PROVIDE(__start_{section} = .);
                        KEEP(*({section}));
                        PROVIDE(__stop_{section} = .);
                    }}
                }}
            ''')

        lds = '\n'.join(map(make_lds, sections)) + '\n'
        Path(path).write_text(lds)

    if (path := args.out_trace_events_header):

        class Field:
            def __init__(self, name, logical_type, c_field_type, c_arg_type, c_arg_header):
                self.name = name
                self.logical_type = logical_type
                self.c_arg_type = c_arg_type
                self.c_arg_header = c_arg_header
                self._c_field_type = c_field_type

            @property
            def c_field_type(self):
                # Avoid unnecessary wrapping with __typeof__() when it is only
                # wrapping an identifier to increase compatibiity with parsers
                # that do not support __typeof__(<type>) syntax
                return re.sub(
                    r'__typeof__\((([_0-9A-ZA-z]+))\)',
                    r'\1',
                    self._c_field_type
                )

            @property
            def tp_struct_entry(self):
                typ = self.logical_type
                # TODO: support arrays
                if typ == 'c-string':
                    return f'__string({self.name}, *({self.name}))'
                elif typ == 'rust-string':
                    # Add +1 for the null-terminator
                    return f'__dynamic_array(char, {self.name}, {self.name}->len + 1)'
                elif typ in ('u8', 's8', 'u16', 's16', 'u32', 's32', 'u64', 's64', 'c-static-string'):
                    return f'__field({self.c_field_type}, {self.name})'
                else:
                    raise ValueError(f'Unsupported logical type: {typ}')

            @property
            def entry(self):
                return f'__entry->{self.name}'

            @property
            def tp_fast_assign(self):
                typ = self.logical_type
                # TODO: support arrays
                if typ == 'c-string':
                    return f'__lisa_assign_str({self.name}, *({self.name}));'
                elif typ == 'rust-string':
                    return textwrap.dedent(f'''
                    memcpy(__get_dynamic_array({self.name}), {self.name}->data, {self.name}->len * sizeof(char));
                    ((char *)__get_dynamic_array({self.name}))[{self.name}->len] = 0;
                    ''')
                elif typ in ('u8', 's8', 'u16', 's16', 'u32', 's32', 'u64', 's64', 'c-static-string'):
                    return f'{self.entry} = *({self.name});'
                else:
                    raise ValueError(f'Unsupported logical type: {typ}')

            @property
            def tp_printk(self):
                typ = self.logical_type
                # TODO: support arrays
                if typ in ('s8', 's16', 's32'):
                    return (f'{self.name}=%d', [self.entry])
                elif typ in ('u8', 'u16', 'u32'):
                    return (f'{self.name}=%u', [self.entry])
                elif typ == 's64':
                    return (f'{self.name}=%lld', [self.entry])
                elif typ == 'u64':
                    return (f'{self.name}=%llu', [self.entry])
                elif typ == 'c-static-string':
                    return (f'{self.name}=%s', [self.entry])
                elif typ in ('rust-string', 'c-string'):
                    return (f'{self.name}=%s', [f'__get_str({self.name})'])
                else:
                    raise ValueError(f'Unsupported logical type: {typ}')

        def make_event(entry):
            name = entry['name']
            fields = [
                Field(
                    name=field['name'],
                    logical_type=field['logical-type'],
                    c_arg_type=field['c-arg-type'],
                    c_arg_header=field['c-arg-header'],
                    c_field_type=field['c-field-type'],
                )
                for field in entry['fields']
            ]

            def wrap_c_type(c_typ):
                # Avoid unnecessary wrapping with __typeof__() to increase
                # compatibiity with parsers that do not support
                # __typeof__(<type>) syntax
                if c_typ.isidentifier():
                    return c_typ
                else:
                    return f'__typeof__({c_typ})'

            nl = '\n                    '
            proto = ', '.join(
                f'{wrap_c_type(field.c_arg_type)} {field.name}'
                for field in fields
            )
            args = ', '.join(
                field.name
                for field in fields
            )
            struct_entry = nl.join(
                field.tp_struct_entry
                for field in fields
            )

            assign = nl.join(
                field.tp_fast_assign
                for field in fields
            )

            printk_fmts, printk_args = zip(*(
                field.tp_printk
                for field in fields
            ))
            printk_fmt = ' '.join(printk_fmts)
            # Use json escaping as an easy way to produce a C string literal.
            printk_fmt = json.dumps(printk_fmt)
            printk_args = ', '.join(itertools.chain.from_iterable(printk_args))

            headers = '\n'.join(
                f'#include "{header}"'
                for field in fields
                if (header := field.c_arg_header)
            )

            return textwrap.dedent(f'''
            {headers}

            TRACE_EVENT({name},
                TP_PROTO({proto}),
                TP_ARGS({args}),
                TP_STRUCT__entry(
                    {struct_entry}
                ),
                TP_fast_assign(
                    {assign}
                ),
                TP_printk(
                    {printk_fmt}, {printk_args}
                )
            )
            ''')

        events = sorted(
            (
                entry
                for entry in get_data()
                if entry["type"] == "define-ftrace-event"
            ),
            key=itemgetter('name'),
        )
        out = '\n\n'.join(map(make_event, events))
        Path(path).write_text(out)



if __name__ == '__main__':
    main()
