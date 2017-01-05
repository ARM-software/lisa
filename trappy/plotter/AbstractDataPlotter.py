#    Copyright 2015-2017 ARM Limited
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

"""This is the template class that all Plotters inherit"""
from abc import abstractmethod, ABCMeta
from pandas import DataFrame
import re
from trappy.utils import listify
from functools import reduce
# pylint: disable=R0921
# pylint: disable=R0903


class AbstractDataPlotter(object):
    """This is an abstract data plotting Class defining an interface
       for the various Plotting Classes"""

    __metaclass__ = ABCMeta

    def __init__(self, traces=None, attr=None, templates=None):
        self._event_map = {}
        self._attr = attr if attr else {}
        self.traces = traces
        self.templates = templates

    @abstractmethod
    def view(self):
        """View the graph"""
        raise NotImplementedError("Method Not Implemented")

    @abstractmethod
    def savefig(self, path):
        """Save the image as a file

        :param path: Location of the Saved File
        :type path: str
        """
        raise NotImplementedError("Method Not Implemented")

    def _check_data(self):
        """Internal function to check the received data"""

        data = listify(self.traces)

        if len(data):
            mask = map(lambda x: isinstance(x, DataFrame), data)
            data_frame = reduce(lambda x, y: x and y, mask)
            sig_or_template = self.templates or "signals" in self._attr

            if not data_frame and not sig_or_template:
                raise ValueError(
                    "Cannot understand data. Accepted DataFormats are pandas.DataFrame or trappy.FTrace/BareTrace/SysTrace (with templates)")
            elif data_frame and "column" not in self._attr:
                raise ValueError("Column not specified for DataFrame input")
        else:
            raise ValueError("Empty Data received")

    def _parse_value(self, signal_def):
        """Parse a signal definition into a (template, column) tuple

        :param signal_def: A signal definition. E.g. "trace_class:column"
        :type signal_def: str
        """

        match = re.match(r"(?P<event>[^:]+):(?P<column>[^:]+)(?P<color>:.+)?",
                         signal_def)
        event = match.group("event")
        column = match.group("column")
        color_match = match.group("color")
        if color_match:
            color_list = color_match[1:].split(",", 2)
            color = [int(n, 16) if n.startswith("0x") else int(n) for n in color_list]
        else:
            color = None

        try:
            return self._event_map[event], column, color
        except KeyError:
            for trace in listify(self.traces):

                if event in trace.class_definitions:
                    self._event_map[event] = trace.class_definitions[event]
                    return self._event_map[event], column, color

            raise ValueError(
                "Event: " +
                event +
                " not found in Trace Object")

    def _describe_signals(self):
        """Internal Function for populating templates and columns
        from signals
        """

        if "column" in self._attr or self.templates:
            raise ValueError("column/templates specified with values")

        self._attr["column"] = []
        self.templates = []
        colors = []

        for value in listify(self._attr["signals"]):
            template, column, color = self._parse_value(value)
            self.templates.append(template)
            self._attr["column"].append(column)
            colors.append(color)

        if any(colors):
            self._attr["colors"] = colors
