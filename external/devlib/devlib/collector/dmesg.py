#    Copyright 2024 ARM Limited
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

import re
from itertools import takewhile
from datetime import timedelta
import logging

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.exception import TargetStableError
from devlib.utils.misc import memoized


_LOGGER = logging.getLogger('dmesg')


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

    :param line_nr: Line number at which this entry appeared in the ``dmesg``
        output. Note that this is not guaranteed to be unique across collectors, as
        the buffer can be cleared. The timestamp is the only reliable index.
    :type line_nr: int
    """

    _TIMESTAMP_MSG_REGEX = re.compile(r'\[(.*?)\] (.*)')
    _RAW_LEVEL_REGEX = re.compile(r'<([0-9]+)>(.*)')
    _PRETTY_LEVEL_REGEX = re.compile(r'\s*([a-z]+)\s*:([a-z]+)\s*:\s*(.*)')

    def __init__(self, facility, level, timestamp, msg, line_nr=0):
        self.facility = facility
        self.level = level
        self.timestamp = timestamp
        self.msg = msg
        self.line_nr = line_nr

    @classmethod
    def from_str(cls, line, line_nr=0):
        """
        Parses a "dmesg --decode" output line, formatted as following:
        kern  :err   : [3618282.310743] nouveau 0000:01:00.0: systemd-logind[988]: nv50cal_space: -16

        Or the more basic output given by "dmesg -r":
        <3>[3618282.310743] nouveau 0000:01:00.0: systemd-logind[988]: nv50cal_space: -16

        """

        def parse_raw_level(line):
            match = cls._RAW_LEVEL_REGEX.match(line)
            if not match:
                raise ValueError(f'dmesg entry format not recognized: {line}')
            level, remainder = match.groups()
            levels = DmesgCollector.LOG_LEVELS
            # BusyBox dmesg can output numbers that need to wrap around
            level = levels[int(level) % len(levels)]
            return level, remainder

        def parse_pretty_level(line):
            match = cls._PRETTY_LEVEL_REGEX.match(line)
            if not match:
                raise ValueError(f'dmesg entry pretty format not recognized: {line}')
            facility, level, remainder = match.groups()
            return facility, level, remainder

        def parse_timestamp_msg(line):
            match = cls._TIMESTAMP_MSG_REGEX.match(line)
            if not match:
                raise ValueError(f'dmesg entry timestamp format not recognized: {line}')
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
            line_nr=line_nr,
        )

    @classmethod
    def from_dmesg_output(cls, dmesg_out, error=None):
        """
        Return a generator of :class:`KernelLogEntry` for each line of the
        output of dmesg command.

        :param error: If ``"raise"`` or ``None``, an exception will be raised
            if a parsing error occurs. If ``"warn"``, it will be logged at
            WARNING level. If ``"ignore"``, it will be ignored. If a callable
            is passed, the exception will be passed to it.
        :type error: str or None or typing.Callable[[BaseException], None]

        .. note:: The same restrictions on the dmesg output format as for
            :meth:`from_str` apply.
        """
        for i, line in enumerate(dmesg_out.splitlines()):
            if line.strip():
                try:
                    yield cls.from_str(line, line_nr=i)
                except Exception as e:
                    if error in (None, 'raise'):
                        raise e
                    elif error == 'warn':
                        _LOGGER.warn(f'error while parsing line "{line!r}": {e}')
                    elif error == 'ignore':
                        pass
                    elif callable(error):
                        error(e)
                    else:
                        raise ValueError(f'Unknown error handling strategy: {error}')

    def __str__(self):
        facility = self.facility + ': ' if self.facility else ''
        return '{facility}{level}: [{timestamp}] {msg}'.format(
            facility=facility,
            level=self.level,
            timestamp=self.timestamp.total_seconds(),
            msg=self.msg,
        )


class DmesgCollector(CollectorBase):
    """
    Dmesg output collector.

    :param level: Minimum log level to enable. All levels that are more
        critical will be collected as well.
    :type level: str

    :param facility: Facility to record, see dmesg --help for the list.
    :type level: str

    :param empty_buffer: If ``True``, the kernel dmesg ring buffer will be
        emptied before starting. Note that this will break nesting of collectors,
        so it's not recommended unless it's really necessary.
    :type empty_buffer: bool

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

    def __init__(self, target, level=LOG_LEVELS[-1], facility='kern', empty_buffer=False, parse_error=None):
        super(DmesgCollector, self).__init__(target)

        if not target.is_rooted:
            raise TargetStableError('Cannot collect dmesg on non-rooted target')

        self.output_path = None

        if level not in self.LOG_LEVELS:
            raise ValueError('level needs to be one of: {}'.format(
                ', '.join(self.LOG_LEVELS)
            ))
        self.level = level

        # Check if we have a dmesg from a recent util-linux build, rather than
        # e.g. busybox's dmesg or the one shipped on some Android versions
        # (toybox).  Note: BusyBox dmesg does not support -h, but will still
        # print the help with an exit code of 1
        help_ = self.target.execute('dmesg -h', check_exit_code=False)
        self.basic_dmesg = not all(
            opt in help_
            for opt in ('--facility', '--force-prefix', '--decode', '--level')
        )

        self.facility = facility
        try:
            needs_root = target.read_sysctl('kernel.dmesg_restrict')
        except ValueError:
            needs_root = True
        else:
            needs_root = bool(int(needs_root))
        self.needs_root = needs_root

        self._begin_timestamp = None
        self.empty_buffer = empty_buffer
        self._dmesg_out = None
        self._parse_error = parse_error

    @property
    def dmesg_out(self):
        out = self._dmesg_out
        if out is None:
            return None
        else:
            try:
                entry = self.entries[0]
            except IndexError:
                return ''
            else:
                i = entry.line_nr
                return '\n'.join(out.splitlines()[i:])

    @property
    def entries(self):
        return self._get_entries(
            self._dmesg_out,
            self._begin_timestamp,
            error=self._parse_error,
        )

    @memoized
    def _get_entries(self, dmesg_out, timestamp, error):
        entries = KernelLogEntry.from_dmesg_output(dmesg_out, error=error)
        entries = list(entries)
        if timestamp is None:
            return entries
        else:
            try:
                first = entries[0]
            except IndexError:
                pass
            else:
                if first.timestamp > timestamp:
                    msg = 'The dmesg ring buffer has ran out of memory or has been cleared and some entries have been lost'
                    raise ValueError(msg)

            return [
                entry
                for entry in entries
                # Only select entries that are more recent than the one at last
                # reset()
                if entry.timestamp > timestamp
            ]

    def _get_output(self):
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

        self._dmesg_out = self.target.execute(cmd, as_root=self.needs_root)

    def reset(self):
        self._dmesg_out = None

    def start(self):
        # If the buffer is emptied on start(), it does not matter as we will
        # not end up with entries dating from before start()
        if self.empty_buffer:
            # Empty the dmesg ring buffer. This requires root in all cases
            self.target.execute('dmesg -c', as_root=True)
        else:
            self._get_output()
            try:
                entry = self.entries[-1]
            except IndexError:
                pass
            else:
                self._begin_timestamp = entry.timestamp

    def stop(self):
        self._get_output()

    def set_output(self, output_path):
        self.output_path = output_path

    def get_data(self):
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")
        with open(self.output_path, 'wt') as f:
            f.write((self.dmesg_out or '') + '\n')
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
