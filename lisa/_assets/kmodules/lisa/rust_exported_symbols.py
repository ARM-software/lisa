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


def main():
    parser = argparse.ArgumentParser("""
    Get the list of exported Rust functions to make it available to C code (and
    not garbage collect these entry points it when linking).
    """)

    parser.add_argument('--rust-object', help='Built Rust object file', required=True)
    parser.add_argument('--out-symbols-plain', help='File to write the symbol list, one per line')
    parser.add_argument('--out-symbols-cli', help='File to write the symbol list as ld CLI --undefined options')

    args = parser.parse_args()

    nm = os.environ.get('NM', 'nm')
    out = subprocess.check_output(
        [nm, '-gj', args.rust_object],
    )

    # For each symbol we want to export in Rust, we create a companion symbol
    # with a prefix that we pick up here. There unfortunately seems to be no
    # cleaner way to convey the list of exported symbols from Rust code as of
    # 2024.
    prefix = b'__export_rust_symbol_'
    def parse(sym):
        if sym.startswith(prefix):
            sym = sym[len(prefix):]
            return sym.decode()
        else:
            return None

    symbols = sorted(
        sym
        for _sym in out.split()
        if (sym := parse(_sym))
    )

    sep = '\n  '
    pretty_symbols = sep.join(symbols)
    print(f'Found exported symbols:{sep}{pretty_symbols}')

    if (path := args.out_symbols_plain):
        content = '\n'.join(symbols) + '\n'
        Path(path).write_text(content)

    if (path := args.out_symbols_cli):
        content = ' '.join((
            f'--undefined {quote(sym)}'
            for sym in symbols
        )) + '\n'
        Path(path).write_text(content)

if __name__ == '__main__':
    main()
