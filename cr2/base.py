#!/usr/bin/python
"""Base class to parse trace.dat dumps"""

import os
import re
import pandas as pd

def trace_parser_explode_array(string, array_lengths):
    """Explode an array in the trace into individual elements for easy parsing

    Basically, turn "load={1 1 2 2}" into "load0=1 load1=1 load2=2
    load3=2".  array_lengths is a dictionary of array names and their
    expected length.  If we get array that's shorter than the expected
    length, additional keys have to be introduced with value 0 to
    compensate.  For example, "load={1 2}" with array_lengths being
    {"load": 4} returns "load0=1 load1=2 load2=0 load3=0"

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

    Don't use directly, create a subclass that defines the unique_word
    you want to match in the output"""
    def __init__(self, basepath, unique_word):
        if basepath is None:
            basepath = "."

        self.basepath = basepath
        self.data_frame = pd.DataFrame()
        self.unique_word = unique_word

        if not os.path.isfile(os.path.join(basepath, "trace.txt")):
            self.__run_trace_cmd_report()

        self.__parse_into_dataframe()

    def __run_trace_cmd_report(self):
        """Run "trace-cmd report > trace.txt".

        Overwrites the contents of trace.txt if it exists."""
        from subprocess import check_output

        trace_fname = os.path.join(self.basepath, "trace.dat")
        if not os.path.isfile(trace_fname):
            raise IOError("No such file or directory: {}".format(trace_fname))

        with open(os.devnull) as devnull:
            out = check_output(["trace-cmd", "report", trace_fname],
                               stderr=devnull)

        with open(os.path.join(self.basepath, "trace.txt"), "w") as fout:
            fout.write(out)

    def get_trace_array_lengths(self, fname):
        """Calculate the lengths of all arrays in the trace

        Returns a dict with the name of each array found in the trace
        as keys and their corresponding length as value

        """
        from collections import defaultdict

        pat_array = re.compile(r"([A-Za-z0-9_]+)={([^}]+)}")

        ret = defaultdict(int)

        with open(fname) as fin:
            for line in fin:
                if not re.search(self.unique_word, line):
                    continue

                while True:
                    match = re.search(pat_array, line)
                    if not match:
                        break

                    (array_name, array_elements) = match.groups()

                    array_len = len(array_elements.split(' '))

                    if array_len > ret[array_name]:
                        ret[array_name] = array_len

                    line = line[match.end():]

        return ret

    def __parse_into_dataframe(self):
        """parse the trace and create a pandas DataFrame"""

        fin_fname = os.path.join(self.basepath, "trace.txt")

        array_lengths = self.get_trace_array_lengths(fin_fname)

        pat_timestamp = re.compile(r"([0-9]+\.[0-9]+):")
        pat_data_start = re.compile("[A-Za-z0-9_]+=")
        pat_empty_array = re.compile(r"[A-Za-z0-9_]+=\{\} ")

        parsed_data = []
        time_array = []

        with open(fin_fname) as fin:
            for line in fin:
                if not re.search(self.unique_word, line):
                    continue

                line = line[:-1]

                timestamp_match = re.search(pat_timestamp, line)
                timestamp = float(timestamp_match.group(1))
                time_array.append(timestamp)

                data_start_idx = re.search(pat_data_start, line).start()
                data_str = line[data_start_idx:]

                # Remove empty arrays from the trace
                data_str = re.sub(pat_empty_array, r"", data_str)

                data_str = trace_parser_explode_array(data_str, array_lengths)

                line_data = {}
                for field in data_str.split():
                    (key, value) = field.split('=')
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    line_data[key] = value

                parsed_data.append(line_data)

        time_idx = pd.Index(time_array, name="Time")
        self.data_frame = pd.DataFrame(parsed_data, index=time_idx)

    def write_csv(self, fname):
        """Write the csv info in thermal.csv"""
        self.data_frame.to_csv(fname)

    def normalize_time(self, basetime):
        """Substract basetime from the Time of the data frame"""
        if basetime:
            self.data_frame.reset_index(inplace=True)
            self.data_frame["Time"] = self.data_frame["Time"] - basetime
            self.data_frame.set_index("Time", inplace=True)
