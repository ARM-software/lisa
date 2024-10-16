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
import json
from pathlib import Path
import sys
from shlex import quote


def main():
    parser = argparse.ArgumentParser("""
    Parse the JSON output of rustdoc --output-format=json and extract the
    exported C symbols.
    """)

    parser.add_argument('--rustdoc-json', help='JSON file to parse', required=True)
    parser.add_argument('--out-symbols-plain', help='File to write the symbol list, one per line')
    parser.add_argument('--out-symbols-cli', help='File to write the symbol list as ld CLI --undefined options')

    args = parser.parse_args()
    path = Path(args.rustdoc_json)

    with open(path, 'r') as f:
        data = json.load(f)

    items = [
        item
        for item in data['index'].values()
        if '#[no_mangle]' in item['attrs']
    ]
    symbols = sorted(
        item['name']
        for item in items
    )

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
