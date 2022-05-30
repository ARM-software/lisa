# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022, ARM Limited and contributors.
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

"""
Rust Analysis companion module.


This modules is designed to be fast to import so that the
:class:`multiprocessing.Pool` can spawn new interpreters cheaply.
"""


import ujson
import regex as regex_re


def _convert_value(x):
    try:
        return int(x)
    except Exception:
        pass

    if x.startswith(b'0x'):
        try:
            return int(x, 16)
        except Exception:
            pass

    try:
        return float(x)
    except Exception:
        pass

    return x


_TRACE_CMD_REGEX_COMMON = (
    rb'\s*(?P<__comm>.*?)\s*\[(?P<__cpu>[0-9]+)\]\s*(?P<__ts>[0-9.]+):'
)

_TRACE_CMD_REGEX = regex_re.compile(
    _TRACE_CMD_REGEX_COMMON + rb'\s*(?P<__type>\S+):(?:\s*(?P<field>\S+)=(?P<value>(.*?(?= \S+=|$))))*'
)

_TRACE_CMD_REGEX_COMMON = regex_re.compile(_TRACE_CMD_REGEX_COMMON)


def _json_record(line):
    match = _TRACE_CMD_REGEX.match(line)
    if match is None:
        raise ValueError(f'Could not parse: {line}')
    else:
        record = match.groupdict()
        del record['field']
        del record['value']
        record['__ts'] = int(float(record['__ts']) * 1e9)
        record['__cpu'] = int(record['__cpu'])
        comm, pid = record['__comm'].rsplit(b'-', 1)
        record['__comm'] = comm
        record['__pid'] = int(pid)

        caps = match.captures
        record.update(
            zip(
                caps('field'),
                map(_convert_value, caps('value'))
            )
        )
        return record


def _json_line(line):
    try:
        record = _json_record(line)
    except ValueError:
        match = _TRACE_CMD_REGEX_COMMON.match(line)
        if match is None:
            # We need a value to avoid parsing failure in the Rust code.
            ts = 0
        else:
            ts = int(float(match.captures('__ts') * 1e9))

        record = {
            '__type': b'__lisa_unknown_event',
            '__ts': ts,
            'data': line,
        }

    txt = ujson.dumps(record, reject_bytes=False, ensure_ascii=False, indent=0)
    return (record, txt)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
