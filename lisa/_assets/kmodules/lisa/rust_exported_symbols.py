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


def main():
    parser = argparse.ArgumentParser("""
    Get the list of exported Rust functions to make it available to C code (and
    not garbage collect these entry points it when linking).
    """)

    parser.add_argument('--rust-object', help='Built Rust object file', required=True)
    parser.add_argument('--out-symbols-plain', help='File to write the symbol list, one per line')
    parser.add_argument('--out-symbols-cli', help='File to write the symbol list as ld CLI --undefined options')
    parser.add_argument('--out-start-stop-lds', help='File to write the linker script to provide __start_SECNAME and __stop_SECNAME symbols, which are not provided by Kbuild since the module is never linked into an executable or DSO')

    args = parser.parse_args()

    nm = os.environ.get('NM', 'nm')
    out = subprocess.check_output(
        [nm, '-gj', args.rust_object],
    )
    all_symbols = sorted(set(out.decode().split()))

    def parse(sym, prefix):
        if sym.startswith(prefix):
            sym = sym[len(prefix):]
            return sym
        else:
            return None

    # For each symbol we want to export in Rust, we create a companion symbol
    # with a prefix that we pick up here. There unfortunately seems to be no
    # cleaner way to convey the list of exported symbols from Rust code as of
    # 2024.
    user_symbols = [
        sym
        for _sym in all_symbols
        if (sym := parse(_sym, '__export_rust_symbol_'))
    ]

    symbols = sorted(user_symbols)

    sep = '\n  '
    print(f'Found exported symbols:{sep}{sep.join(symbols)}')

    if (path := args.out_symbols_plain):
        content = '\n'.join(symbols) + '\n'
        Path(path).write_text(content)

    if (path := args.out_symbols_cli):
        content = ' '.join((
            f'--undefined {quote(sym)}'
            for sym in symbols
        )) + '\n'
        Path(path).write_text(content)

    if (path := args.out_start_stop_lds):
        sections = sorted({
            section
            for sym in all_symbols
            if (
                (section := parse(sym, '__start_')) or
                (section := parse(sym, '__stop_'))
            )
        })
        print(f'Found sections to generate __start_SECNAME and __stop_SECNAME symbols for:{sep}{sep.join(sections)}')

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
