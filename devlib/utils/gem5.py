#    Copyright 2017-2018 ARM Limited
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

import re
import logging

from devlib.utils.types import numeric


GEM5STATS_FIELD_REGEX = re.compile(r"^(?P<key>[^- ]\S*) +(?P<value>[^#]+).+$")
GEM5STATS_DUMP_HEAD = '---------- Begin Simulation Statistics ----------'
GEM5STATS_DUMP_TAIL = '---------- End Simulation Statistics   ----------'
GEM5STATS_ROI_NUMBER = 8

logger = logging.getLogger('gem5')


def iter_statistics_dump(stats_file):
    '''
    Yields statistics dumps as dicts. The parameter is assumed to be a stream
    reading from the statistics log file.
    '''
    cur_dump = {}
    while True:
        line = stats_file.readline()
        if not line:
            break
        if GEM5STATS_DUMP_TAIL in line:
            yield cur_dump
            cur_dump = {}
        else:
            res = GEM5STATS_FIELD_REGEX.match(line)
            if res:
                k = res.group("key")
                vtext = res.group("value")
                try:
                    v = list(map(numeric, vtext.split()))
                    cur_dump[k] = v[0] if len(v) == 1 else set(v)
                except ValueError:
                    msg = 'Found non-numeric entry in gem5 stats ({}: {})'
                    logger.warning(msg.format(k, vtext))
