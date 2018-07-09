#    Copyright 2018 ARM Limited
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

import logging
from datetime import datetime

from wa.utils.types import enum


LogcatLogLevel = enum(['verbose', 'debug', 'info', 'warn', 'error', 'assert'], start=2)

log_level_map = ''.join(n[0].upper() for n in LogcatLogLevel.names)

logger = logging.getLogger('logcat')


class LogcatEvent(object):

    __slots__ = ['timestamp', 'pid', 'tid', 'level', 'tag', 'message']

    def __init__(self, timestamp, pid, tid, level, tag, message):
        self.timestamp = timestamp
        self.pid = pid
        self.tid = tid
        self.level = level
        self.tag = tag
        self.message = message

    def __repr__(self):
        return '{} {} {} {} {}: {}'.format(
            self.timestamp, self.pid, self.tid,
            self.level.name.upper(), self.tag,
            self.message,
        )

    __str__ = __repr__


class LogcatParser(object):

    def parse(self, filepath):
        with open(filepath) as fh:
            for line in fh:
                event = self.parse_line(line)
                if event:
                    yield event

    def parse_line(self, line):  # pylint: disable=no-self-use
        line = line.strip()
        if not line or line.startswith('-') or ': ' not in line:
            return None

        metadata, message = line.split(': ', 1)

        parts = metadata.split(None, 5)
        try:
            ts = ' '.join([parts.pop(0), parts.pop(0)])
            timestamp = datetime.strptime(ts, '%m-%d %H:%M:%S.%f').replace(year=datetime.now().year)
            pid = int(parts.pop(0))
            tid = int(parts.pop(0))
            level = LogcatLogLevel.levels[log_level_map.index(parts.pop(0))]
            tag = (parts.pop(0) if parts else '').strip()
        except Exception as e:  # pylint: disable=broad-except
            message = 'Invalid metadata for line:\n\t{}\n\tgot: "{}"'
            logger.warning(message.format(line, e))
            return None

        return LogcatEvent(timestamp, pid, tid, level, tag, message)
