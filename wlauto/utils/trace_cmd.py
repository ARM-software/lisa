#    Copyright 2015 ARM Limited
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
import logging

from wlauto.utils.misc import isiterable
from wlauto.utils.types import numeric


logger = logging.getLogger('trace-cmd')


# These markers can be injected into trace to identify the  "interesting"
# portion.
TRACE_MARKER_START = 'TRACE_MARKER_START'
TRACE_MARKER_STOP = 'TRACE_MARKER_STOP'


class TraceCmdEvent(object):
    """
    A single trace-cmd event. This will appear in the trace cmd report in the format ::

          <idle>-0     [000]  3284.126993: sched_rq_runnable_load: cpu=0 load=54
             |           |         |              |                |___________|
             |           |         |              |                      |
          thread        cpu    timestamp        name                    body

    """

    __slots__ = ['thread', 'reporting_cpu_id', 'timestamp', 'name', 'text', 'fields']

    def __init__(self, thread, cpu_id, ts, name, body, parser=None):
        """
        parameters:

        :thread: thread which generated the event
        :cpu: cpu on which the event has occurred
        :ts: timestamp of the event
        :name: the name of the event
        :bodytext: a string with the rest of the event text
        :parser: optionally, a function that will parse bodytext to populate
                 this event's attributes

        The parser can be any callable that can be invoked with

            parser(event, text)

        Where ``event`` is this TraceCmdEvent instance, and ``text`` is the body text to be
        parsed. The parser should updated the passed event instance and not return anything
        (the return value will be ignored). Any exceptions raised by the parser will be silently
        ignored (note that this means that the event's attributes may be partially initialized).

        """
        self.thread = thread
        self.reporting_cpu_id = int(cpu_id)
        self.timestamp = numeric(ts)
        self.name = name
        self.text = body
        self.fields = {}

        if parser:
            try:
                parser(self, self.text)
            except Exception:  # pylint: disable=broad-except
                # unknown format assume user does not care or know how to
                # parse self.text
                pass

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name)

    def __str__(self):
        return 'TE({} @ {})'.format(self.name, self.timestamp)

    __repr__ = __str__


class DroppedEventsEvent(object):

    __slots__ = ['thread', 'reporting_cpu_id', 'timestamp', 'name', 'text', 'fields']

    def __init__(self, cpu_id):
        self.thread = None
        self.reporting_cpu_id = None
        self.timestamp = None
        self.name = 'DROPPED EVENTS DETECTED'
        self.text = None
        self.fields = {'cpu_id': int(cpu_id)}

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name)

    def __str__(self):
        return 'DROPPED_EVENTS_ON_CPU{}'.format(self.cpu_id)

    __repr__ = __str__


def try_convert_to_numeric(v):
    try:
        if isiterable(v):
            return map(numeric, v)
        else:
            return numeric(v)
    except ValueError:
        return v


def default_body_parser(event, text):
    """
    Default parser to attempt to use to parser body text for the event (i.e. after
    the "header" common to all events has been parsed). This assumes that the body is
    a whitespace-separated list of key=value pairs. The parser will attempt to convert
    the value into a numeric type, and failing that, keep it as string.

    """
    kv_pairs = [e.split('=', 1) for e in text.split()]
    new_values = {k: try_convert_to_numeric(v)
                  for (k, v) in kv_pairs}
    event.fields.update(new_values)


def regex_body_parser(regex, flags=0):
    """
    Creates an event body parser form the specified regular expression (could be an
    ``re.RegexObject``, or a string). The regular expression should contain some named
    groups, as those will be extracted as the event attributes (unnamed groups and the
    reset of the match will be ignored).

    If the specified regex is a string, it will be compiled, in which case ``flags`` may
    be provided for the resulting regex object (see ``re`` standard module documentation).
    If regex is a pre-compiled object, flags will be ignored.

    """
    if isinstance(regex, basestring):
        regex = re.compile(regex, flags)

    def regex_parser_func(event, text):
        match = regex.search(text)
        if match:
            event.fields.update(match.groupdict())

    return regex_parser_func


# Maps event onto the corresponding parser for it's body text. A parser may be
# a callable with signature
#
#   parser(event, bodytext)
#
# a re.RegexObject, or a string (in which case it will be compiled into a
# regex). In case of a string/regex, it's named groups will be used to populate
# the event's attributes.
EVENT_PARSER_MAP = {
}

TRACE_EVENT_REGEX = re.compile(r'^\s+(?P<thread>\S+)\s+\[(?P<cpu_id>\d+)\]\s+(?P<ts>[\d.]+):\s+'
                               r'(?P<name>[^:]+):\s+(?P<body>.*?)\s*$')

DROPPED_EVENTS_REGEX = re.compile(r'CPU:(?P<cpu_id>\d+) \[\d*\s*EVENTS DROPPED\]')


class TraceCmdTrace(object):

    def __init__(self, filter_markers=True):
        self.filter_markers = filter_markers

    def parse(self, filepath, names=None, check_for_markers=True):  # pylint: disable=too-many-branches
        """
        This is a generator for the trace event stream.

        """
        inside_maked_region = False
        filters = [re.compile('^{}$'.format(n)) for n in names or []]
        if check_for_markers:
            with open(filepath) as fh:
                for line in fh:
                    if TRACE_MARKER_START in line:
                        break
                else:
                    # maker not found force filtering by marker to False
                    self.filter_markers = False
        with open(filepath) as fh:
            for line in fh:
                # if processing trace markers, skip marker lines as well as all
                # lines outside marked region
                if self.filter_markers:
                    if not inside_maked_region:
                        if TRACE_MARKER_START in line:
                            inside_maked_region = True
                        continue
                    elif TRACE_MARKER_STOP in line:
                        inside_maked_region = False
                        continue

                match = DROPPED_EVENTS_REGEX.search(line)
                if match:
                    yield DroppedEventsEvent(match.group('cpu_id'))
                    continue

                match = TRACE_EVENT_REGEX.search(line)
                if not match:
                    logger.warning('Invalid trace event: "{}"'.format(line))
                    continue

                event_name = match.group('name')

                if filters:
                    found = False
                    for f in filters:
                        if f.search(event_name):
                            found = True
                            break
                    if not found:
                        continue

                body_parser = EVENT_PARSER_MAP.get(event_name, default_body_parser)
                if isinstance(body_parser, basestring):
                    body_parser = regex_body_parser(body_parser)
                yield TraceCmdEvent(parser=body_parser, **match.groupdict())

