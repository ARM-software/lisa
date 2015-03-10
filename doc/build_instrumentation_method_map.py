#!/usr/bin/env python
#    Copyright 2015-2015 ARM Limited
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

from wlauto.core.instrumentation import SIGNAL_MAP, PRIORITY_MAP
from wlauto.utils.doc import format_simple_table


CONVINIENCE_ALIASES = ['initialize', 'setup', 'start', 'stop', 'process_workload_result',
                       'update_result', 'teardown', 'finalize']

OUTPUT_TEMPLATE_FILE =  os.path.join(os.path.dirname(__file__), 'source', 'instrumentation_method_map.template')


def escape_trailing_underscore(value):
    if value.endswith('_'):
        return value[:-1] + '\_'


def generate_instrumentation_method_map(outfile):
    signal_table = format_simple_table([(k, v) for k, v in SIGNAL_MAP.iteritems()],
                                       headers=['method name', 'signal'], align='<<')
    priority_table = format_simple_table([(escape_trailing_underscore(k), v)  for k, v in PRIORITY_MAP.iteritems()],
                                         headers=['prefix', 'priority'],  align='<>')
    with open(OUTPUT_TEMPLATE_FILE) as fh:
        template = string.Template(fh.read())
    with open(outfile, 'w') as wfh:
        wfh.write(template.substitute(signal_names=signal_table, priority_prefixes=priority_table))


if __name__ == '__main__':
    generate_instrumentation_method_map(sys.argv[1])
