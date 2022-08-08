#!/usr/bin/env python
#    Copyright 2015-2019 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
import string
from copy import copy

from wa.framework.instrument import SIGNAL_MAP
from wa.framework.signal import CallbackPriority
from wa.utils.doc import format_simple_table

OUTPUT_TEMPLATE_FILE =  os.path.join(os.path.dirname(__file__), 'source', 'instrument_method_map.template')


def generate_instrument_method_map(outfile):
    signal_table = format_simple_table([(k, v) for k, v in SIGNAL_MAP.items()],
                                       headers=['method name', 'signal'], align='<<')
    decorator_names = map(lambda x: x.replace('high', 'fast').replace('low', 'slow'), CallbackPriority.names)
    priority_table = format_simple_table(zip(decorator_names, CallbackPriority.names, CallbackPriority.values),
            headers=['decorator', 'CallbackPriority name', 'CallbackPriority value'],  align='<>')
    with open(OUTPUT_TEMPLATE_FILE) as fh:
        template = string.Template(fh.read())
    with open(outfile, 'w') as wfh:
        wfh.write(template.substitute(signal_names=signal_table, priority_prefixes=priority_table))


if __name__ == '__main__':
    generate_instrument_method_map(sys.argv[1])
