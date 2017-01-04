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

import re

class BareTrace(object):
    """A wrapper class that holds dataframes for all the events in a trace.

    BareTrace doesn't parse any file so it's a class that should
    either be (a) subclassed to parse a particular trace (like FTrace)
    or (b) be instantiated and the events added with add_parsed_event()

    :param name: is a string describing the trace.
    :type name: str

    """

    def __init__(self, name=""):
        self.name = name
        self.normalized_time = False
        self.class_definitions = {}
        self.trace_classes = []
        self.basetime = 0

    def get_duration(self):
        """Returns the largest time value of all classes,
        returns 0 if the data frames of all classes are empty"""
        durations = []

        for trace_class in self.trace_classes:
            try:
                durations.append(trace_class.data_frame.index[-1])
            except IndexError:
                pass

        if len(durations) == 0:
            return 0

        if self.normalized_time:
            return max(durations)
        else:
            return max(durations) - self.basetime

    def get_filters(self, key=""):
        """Returns an array with the available filters.

        :param key: If specified, returns a subset of the available filters
            that contain 'key' in their name (e.g., :code:`key="sched"` returns
            only the :code:`"sched"` related filters)."""
        filters = []

        for cls in self.class_definitions:
            if re.search(key, cls):
                filters.append(cls)

        return filters

    def normalize_time(self, basetime=None):
        """Normalize the time of all the trace classes

        :param basetime: The offset which needs to be subtracted from
            the time index
        :type basetime: float
        """

        if basetime is not None:
            self.basetime = basetime

        for trace_class in self.trace_classes:
            trace_class.normalize_time(self.basetime)

        self.normalized_time = True

    def add_parsed_event(self, name, dfr, pivot=None):
        """Add a dataframe to the events in this trace

        This function lets you add other events that have been parsed
        by other tools to the collection of events in this instance.  For
        example, assuming you have some events in a csv, you could add
        them to a trace instance like this:

        >>> trace = trappy.BareTrace()
        >>> counters_dfr = pd.DataFrame.from_csv("counters.csv")
        >>> trace.add_parsed_event("pmu_counters", counters_dfr)

        Now you can access :code:`trace.pmu_counters` as you would with any
        other trace event and other trappy classes can interact with
        them.

        :param name: The attribute name in this trace instance.  As in the example above, if :code:`name` is "pmu_counters", the parsed event will be accessible using :code:`trace.pmu_counters`.
        :type name: str

        :param dfr: :mod:`pandas.DataFrame` containing the events.  Its index should be time in seconds.  Its columns are the events.
        :type dfr: :mod:`pandas.DataFrame`

        :param pivot: The data column about which the data can be grouped
        :type pivot: str

        """
        from trappy.base import Base
        from trappy.dynamic import DynamicTypeFactory, default_init

        if hasattr(self, name):
            raise ValueError("event {} already present".format(name))

        kwords = {
            "__init__": default_init,
            "unique_word": name + ":",
            "name": name,
        }

        trace_class = DynamicTypeFactory(name, (Base,), kwords)
        self.class_definitions[name] = trace_class

        event = trace_class()
        self.trace_classes.append(event)
        event.data_frame = dfr
        if pivot:
            event.pivot = pivot

        setattr(self, name, event)

    def finalize_objects(self):
        for trace_class in self.trace_classes:
            trace_class.create_dataframe()
            trace_class.finalize_object()
