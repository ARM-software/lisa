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


if __name__ == '__main__':
    main()
