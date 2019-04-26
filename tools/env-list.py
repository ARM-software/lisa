#! /usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
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

import os
import textwrap
import argparse

VAR_PREFIX = ['LISA', 'BISECTOR', 'EXEKALL']
TABLE_HEADER = """
.. list-table::
  :widths: auto
  :align: left
"""

def format_doc(doc, style=None):
    if not doc:
        return ''

    doc = doc.strip()
    doc = textwrap.fill(doc)

    if style == 'rst':
        return doc
    else:
        return "# {}\n".format(doc.replace("\n", "\n# "))

def main():
    parser = argparse.ArgumentParser(
        description="List LISA environment variable, their value and doc"
    )
    parser.add_argument('--rst',
        action='store_true',
        help='format as reStructuredText'
    )
    parser.add_argument('--filter-home',
        action='store_true',
        help='filter out the value of LISA_HOME from the env var val'
    )
    args = parser.parse_args()

    style = 'rst' if args.rst else None
    filter_lisa_home = args.filter_home

    env_list = [
        var for var in os.environ.keys()
        if any(var.startswith(prefix) for prefix in VAR_PREFIX)
    ]

    doc_map = {
        var: os.getenv("_DOC_{}".format(var))
        for var in env_list
    }

    if style == 'rst':
        print(TABLE_HEADER)

    LISA_HOME = os.environ['LISA_HOME']
    for var, doc in sorted(doc_map.items(), key=lambda k_v: k_v[0]):
        val = os.getenv(var)
        if filter_lisa_home and os.path.isabs(val):
            val = os.path.relpath(val, LISA_HOME)

        doc = format_doc(doc, style=style)

        if style == 'rst':
            entry = "* - {var}\n  - {doc}\n  - {val}".format(
                var=var,
                doc=doc.replace('\n', '\n    '),
                val=val
            )
            entry = textwrap.indent(entry, ' ' * 2)
        else:
            entry = "{doc}{var}={val}\n".format(
                var=var,
                doc=doc,
                val=val,
            )

        print(entry)

if __name__ == '__main__':
    main()

