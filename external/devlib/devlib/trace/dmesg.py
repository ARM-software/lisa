#    Copyright 2019 ARM Limited
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

from __future__ import division
import re
from itertools import takewhile
from datetime import timedelta

from devlib.trace import TraceCollector


class KernelLogEntry(object):
    """
    Entry of the kernel ring buffer.

    :param facility: facility the entry comes from
    :type facility: str

    :param level: log level
    :type level: str

    :param timestamp: Timestamp of the entry
    :type timestamp: datetime.timedelta

    :param msg: Content of the entry
    :type msg: str
    """

    _TIMESTAMP_MSG_REGEX = re.compile(r'\[(.*?)\] (.*)')
    _RAW_LEVEL_REGEX = re.compile(r'<([0-9]+)>(.*)')
    _PRETTY_LEVEL_REGEX = re.compile(r'\s*([a-z]+)\s*:([a-z]+)\s*:\s*(.*)')

    def __init__(self, facility, level, timestamp, msg):
        self.facility = facility
        self.level = level
        self.timestamp = timestamp
        self.msg = msg

    @classmethod
    def from_str(cls, line):
        """
        Parses a "dmesg --decode" output line, formatted as following:
        kern  :err   : [3618282.310743] nouveau 0000:01:00.0: systemd-logind[988]: nv50cal_space: -16

        Or the more basic output given by "dmesg -r":
        <3>[3618282.310743] nouveau 0000:01:00.0: systemd-logind[988]: nv50cal_space: -16

        """

        def parse_raw_level(line):
            match = cls._RAW_LEVEL_REGEX.match(line)
            if not match:
                raise ValueError('dmesg entry format not recognized: {}'.format(line))
            level, remainder = match.groups()
            levels = DmesgCollector.LOG_LEVELS
            # BusyBox dmesg can output numbers that need to wrap around
            level = levels[int(level) % len(levels)]
            return level, remainder

        def parse_pretty_level(line):
            match = cls._PRETTY_LEVEL_REGEX.match(line)
            facility, level, remainder = match.groups()
            return facility, level, remainder

        def parse_timestamp_msg(line):
            match = cls._TIMESTAMP_MSG_REGEX.match(line)
            timestamp, msg = match.groups()
            timestamp = timedelta(seconds=float(timestamp.strip()))
            return timestamp, msg

        line = line.strip()

        # If we can parse the raw prio directly, that is a basic line
        try:
            level, remainder = parse_raw_level(line)
            facility = None
        except ValueError:
            facility, level, remainder = parse_pretty_level(line)

        timestamp, msg = parse_timestamp_msg(remainder)

        return cls(
            facility=facility,
            level=level,
            timestamp=timestamp,
            msg=msg.strip(),
        )

    @classmethod
    def from_dmesg_output(cls, dmesg_out):
        """
        Return a generator of :class:`KernelLogEntry` for each line of the
        output of dmesg command.

        .. note:: The same restrictions on the dmesg output format as for
            :meth:`from_str` apply.
        """
        for line in dmesg_out.splitlines():
            if line.strip():
                yield cls.from_str(line)

    def __str__(self):
        facility = self.facility + ': ' if self.facility else ''
        return '{facility}{level}: [{timestamp}] {msg}'.format(
            facility=facility,
            level=self.level,
            timestamp=self.timestamp.total_seconds(),
            msg=self.msg,
        )


class DmesgCollector(TraceCollector):
    """
    Dmesg output collector.

    :param level: Minimum log level to enable. All levels that are more
        critical will be collected as well.
    :type level: str

    :param facility: Facility to record, see dmesg --help for the list.
    :type level: str

    .. warning:: If BusyBox dmesg is used, facility and level will be ignored,
        and the parsed entries will also lack that information.
    """

    # taken from "dmesg --help"
    # This list needs to be ordered by priority
    LOG_LEVELS = [
        "emerg",        # system is unusable
        "alert",        # action must be taken immediately
        "crit",         # critical conditions
        "err",          # error conditions
        "warn",         # warning conditions
        "notice",       # normal but significant condition
        "info",         # informational
        "debug",        # debug-level messages
    ]

    def __init__(self, target, level=LOG_LEVELS[-1], facility='kern'):
        super(DmesgCollector, self).__init__(target)

        if level not in self.LOG_LEVELS:
            raise ValueError('level needs to be one of: {}'.format(
                ', '.join(self.LOG_LEVELS)
            ))
        self.level = level

        # Check if dmesg is the BusyBox one, or the one from util-linux in a
        # recent version.
        # Note: BusyBox dmesg does not support -h, but will still print the
        # help with an exit code of 1
        self.basic_dmesg = '--force-prefix' not in \
                self.target.execute('dmesg -h', check_exit_code=False)
        self.facility = facility
        self.reset()

    @property
    def entries(self):
        return KernelLogEntry.from_dmesg_output(self.dmesg_out)

    def reset(self):
        self.dmesg_out = None

    def start(self):
        self.reset()
        # Empty the dmesg ring buffer
        self.target.execute('dmesg -c', as_root=True)

    def stop(self):
        levels_list = list(takewhile(
            lambda level: level != self.level,
            self.LOG_LEVELS
        ))
        levels_list.append(self.level)
        if self.basic_dmesg:
            cmd = 'dmesg -r'
        else:
            cmd = 'dmesg --facility={facility} --force-prefix --decode --level={levels}'.format(
                levels=','.join(levels_list),
                facility=self.facility,
            )

        self.dmesg_out = self.target.execute(cmd)

    def get_trace(self, outfile):
        with open(outfile, 'wt') as f:
            f.write(self.dmesg_out + '\n')
