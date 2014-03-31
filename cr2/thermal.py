#!/usr/bin/python
"""Process the output of the power allocator trace in the current directory's trace.dat"""

import os
import re
from StringIO import StringIO
import pandas as pd

class Thermal(object):
    def __init__(self):
        if not os.path.isfile("trace.txt"):
            self.__run_trace_cmd_report()
        self.data_csv = ""
        self.data_frame = False


    def __run_trace_cmd_report(self):
        """Run "trace-cmd report > trace.txt".  Overwrites the contents of trace.txt if it exists."""
        from subprocess import check_output

        with open(os.devnull) as devnull:
            out = check_output(["trace-cmd", "report"], stderr=devnull)

        with open("trace.txt", "w") as f:
            f.write(out)

    def __parse_into_csv(self):
        """Create a csv representation of the thermal data and store it in self.data_csv"""
        pat_timestamp = re.compile(r"([0-9]+\.[0-9]+):")
        pat_data = re.compile(r"[A-Za-z0-9_]+=([0-9]+) ")
        header = ""

        with open("trace.txt") as fin:
            for line in fin:
                if not re.search("Ptot_out", line):
                    continue

                line = line[:-1]

                m = re.search(pat_timestamp, line)
                timestamp = m.group(1)

                semi_idx = line.index(" : ")
                data_str = line[semi_idx + 3:]

                if not header:
                    header = re.sub(r"([A-Za-z0-9_]+)=[0-9]+ ", r"\1,", data_str)
                    header = header[:-1]
                    header = "time," + header + "\n"
                    self.data_csv = header

                parsed_data = re.sub(pat_data, r"\1,", data_str)
                # Drop the last comma
                parsed_data = parsed_data[:-1]

                parsed_data = timestamp + "," + parsed_data + "\n"
                self.data_csv += parsed_data

    def write_thermal_csv(self):
        """Write the csv info in thermal.csv"""
        if not self.data_csv:
            self.__parse_into_csv()

        with open("thermal.csv", "w") as fout:
            fout.write(self.data_csv)

    def get_data_frame(self):
        """Return a pandas data frame for the run"""
        if self.data_frame:
            return self.data_frame

        if not self.data_csv:
            self.__parse_into_csv()

        self.data_frame = pd.read_csv(StringIO(self.data_csv)).set_index("time")
        return self.data_frame
