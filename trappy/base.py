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

"""Base class to parse trace.dat dumps"""

import re
import pandas as pd
import warnings

from resource import getrusage, RUSAGE_SELF

def _get_free_memory_kb():
    try:
        with open("/proc/meminfo") as f:
            memfree_line = [l for l in f.readlines() if "MemFree" in l][0]
            _, num_kb, _ = memfree_line.split()
            return int(num_kb)
    except:
        # Probably either not running on Linux (no /proc/meminfo), or format has
        # changed (we didn't find num_kb).
        return None

def trace_parser_explode_array(string, array_lengths):
    """Explode an array in the trace into individual elements for easy parsing

    Basically, turn :code:`load={1 1 2 2}` into :code:`load0=1 load1=1 load2=2
    load3=2`.

    :param string: Input string from the trace
    :type string: str

    :param array_lengths: A dictionary of array names and their
        expected length.  If we get array that's shorter than the expected
        length, additional keys have to be introduced with value 0 to
        compensate.
    :type array_lengths: dict

    For example:
    ::

        trace_parser_explode_array(string="load={1 2}",
                                   array_lengths={"load": 4})
        "load0=1 load1=2 load2=0 load3=0"
    """

    while True:
        match = re.search(r"[^ ]+={[^}]+}", string)
        if match is None:
            break

        to_explode = match.group()
        col_basename = re.match(r"([^=]+)=", to_explode).groups()[0]
        vals_str = re.search(r"{(.+)}", to_explode).groups()[0]
        vals_array = vals_str.split(' ')

        exploded_str = ""
        for (idx, val) in enumerate(vals_array):
            exploded_str += "{}{}={} ".format(col_basename, idx, val)

        vals_added = len(vals_array)
        if vals_added < array_lengths[col_basename]:
            for idx in range(vals_added, array_lengths[col_basename]):
                exploded_str += "{}{}=0 ".format(col_basename, idx)

        exploded_str = exploded_str[:-1]
        begin_idx = match.start()
        end_idx = match.end()

        string = string[:begin_idx] + exploded_str + string[end_idx:]

    return string

class Base(object):
    """Base class to parse trace.dat dumps.

    Don't use directly, create a subclass that has a unique_word class
    variable.  unique_word is a string that can uniquely identify
    lines in the trace that correspond to this event.  This is usually
    the trace_name (optionally followed by a semicolong,
    e.g. "sched_switch:") but it can be anything else for trace points
    generated using trace_printk().

    :param parse_raw: If :code:`True`, raw trace data (-R option) to
        trace-cmd will be used

    This class acts as a base class for all TRAPpy events

    """
    def __init__(self, parse_raw=False):
        self.data_frame = pd.DataFrame()
        self.data_array = []
        self.time_array = []
        self.comm_array = []
        self.pid_array = []
        self.cpu_array = []
        self.parse_raw = parse_raw

    def finalize_object(self):
        pass

    def __get_trace_array_lengths(self):
        """Calculate the lengths of all arrays in the trace

        Returns a dict with the name of each array found in the trace
        as keys and their corresponding length as value

        """
        from collections import defaultdict

        pat_array = re.compile(r"([A-Za-z0-9_]+)={([^}]+)}")

        ret = defaultdict(int)

        for line in self.data_array:
            while True:
                match = re.search(pat_array, line)
                if not match:
                    break

                (array_name, array_elements) = match.groups()

                array_len = len(array_elements.split(' '))

                if array_len > ret[array_name]:
                    ret[array_name] = array_len

                line = line[match.end():]

            # Stop scanning if the trace doesn't have arrays
            if len(ret) == 0:
                break

        return ret

    def append_data(self, time, comm, pid, cpu, data):
        """Append data parsed from a line to the corresponding arrays

        The :mod:`DataFrame` will be created from this when the whole trace
        has been parsed.

        :param time: The time for the line that was printed in the trace
        :type time: float

        :param comm: The command name or the execname from which the trace
            line originated
        :type comm: str

        :param pid: The PID of the process from which the trace
            line originated
        :type pid: int

        :param data: The data for matching line in the trace
        :type data: str
        """

        self.time_array.append(time)
        self.comm_array.append(comm)
        self.pid_array.append(pid)
        self.cpu_array.append(cpu)
        self.data_array.append(data)

    def generate_parsed_data(self):

        # Get a rough idea of how much memory we have to play with
        kb_free = _get_free_memory_kb()
        starting_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        check_memory_usage = True

        for (comm, pid, cpu, data_str) in zip(self.comm_array, self.pid_array,
                                              self.cpu_array, self.data_array):
            data_dict = {"__comm": comm, "__pid": pid, "__cpu": cpu}
            prev_key = None
            for field in data_str.split():
                if "=" not in field:
                    # Concatenation is supported only for "string" values
                    if type(data_dict[prev_key]) is not str:
                        continue
                    data_dict[prev_key] += ' ' + field
                    continue
                (key, value) = field.split('=')
                try:
                    value = int(value)
                except ValueError:
                    pass
                data_dict[key] = value
                prev_key = key

            # When running out of memory, Pandas has been observed to segfault
            # rather than throwing a proper Python error.
            # Look at how much memory our process is using and warn if we seem
            # to be getting close to the system's limit.
            kb_used = (getrusage(RUSAGE_SELF).ru_maxrss - starting_maxrss)
            if check_memory_usage and kb_free and kb_used > kb_free * 0.9:
                warnings.warn("TRAPpy: Appear to be low on memory. "
                              "If errors arise, try providing more RAM")
                check_memory_usage = False

            yield data_dict

    def create_dataframe(self):
        """Create the final :mod:`pandas.DataFrame`"""
        if not self.time_array:
            return

        trace_arr_lengths = self.__get_trace_array_lengths()

        if trace_arr_lengths.items():
            for (idx, val) in enumerate(self.data_array):
                expl_val = trace_parser_explode_array(val, trace_arr_lengths)
                self.data_array[idx] = expl_val

        time_idx = pd.Index(self.time_array, name="Time")
        self.data_frame = pd.DataFrame(self.generate_parsed_data(), index=time_idx)

        self.time_array = []
        self.comm_array = []
        self.pid_array = []
        self.cpu_array = []
        self.data_array = []

    def write_csv(self, fname):
        """Write the csv info into a CSV file

        :param fname: The name of the CSV file
        :type fname: str
        """
        self.data_frame.to_csv(fname)

    def normalize_time(self, basetime):
        """Substract basetime from the Time of the data frame

        :param basetime: The offset which needs to be subtracted from
            the time index
        :type basetime: float
        """
        if basetime and not self.data_frame.empty:
            self.data_frame.reset_index(inplace=True)
            self.data_frame["Time"] = self.data_frame["Time"] - basetime
            self.data_frame.set_index("Time", inplace=True)
