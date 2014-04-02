#!/usr/bin/python
"""Process the output of the power allocator trace in the current directory's trace.dat"""

import os
import re
from StringIO import StringIO
import pandas as pd
from matplotlib import pyplot as plt

GOLDEN_RATIO = 1.618034

class BaseThermal(object):
    def __init__(self, unique_word):
        if not os.path.isfile("trace.txt"):
            self.__run_trace_cmd_report()

        self.unique_word = unique_word
        self.data_csv = ""
        self.data_frame = False

    def __run_trace_cmd_report(self):
        """Run "trace-cmd report > trace.txt".  Overwrites the contents of trace.txt if it exists."""
        from subprocess import check_output

        if not os.path.isfile("trace.dat"):
            raise IOError("No such file or directory: trace.dat")

        with open(os.devnull) as devnull:
            out = check_output(["trace-cmd", "report"], stderr=devnull)

        with open("trace.txt", "w") as f:
            f.write(out)

    def parse_into_csv(self):
        """Create a csv representation of the thermal data and store it in self.data_csv"""
        pat_timestamp = re.compile(r"([0-9]+\.[0-9]+):")
        pat_data = re.compile(r"[A-Za-z0-9_]+=([a-f0-9]+)")
        pat_header = re.compile(r"([A-Za-z0-9_]+)=[a-f0-9]+")
        header = ""

        with open("trace.txt") as fin:
            for line in fin:
                if not re.search(self.unique_word, line):
                    continue

                line = line[:-1]

                m = re.search(pat_timestamp, line)
                timestamp = m.group(1)

                data_start_idx = re.search(r"[A-Za-z0-9_]+=", line).start()
                data_str = line[data_start_idx:]

                if not header:
                    header = re.sub(pat_header, r"\1", data_str)
                    header = re.sub(r" ", r",", header)
                    header = "time," + header + "\n"
                    self.data_csv = header

                parsed_data = re.sub(pat_data, r"\1", data_str)
                parsed_data = re.sub(r" ", r",", parsed_data)

                parsed_data = timestamp + "," + parsed_data + "\n"
                self.data_csv += parsed_data

    def get_data_frame(self):
        """Return a pandas data frame for the run"""
        if self.data_frame:
            return self.data_frame

        if not self.data_csv:
            self.parse_into_csv()

        try:
            self.data_frame = pd.read_csv(StringIO(self.data_csv)).set_index("time")
        except StopIteration:
            if not self.data_frame:
                return pd.DataFrame()
            raise

        return self.data_frame

class Thermal(BaseThermal):
    def __init__(self):
        super(Thermal, self).__init__(
            unique_word="Ptot_out"
        )

    def write_thermal_csv(self):
        """Write the csv info in thermal.csv"""
        if not self.data_csv:
            self.parse_into_csv()

        with open("thermal.csv", "w") as fout:
            fout.write(self.data_csv)

    def set_plot_size(self, width, height):
        """Set the plot size.

        This has to be called before plot()
        """
        if height is None:
            if width is None:
                height = 6
                width = 10
            else:
                height = width / GOLDEN_RATIO
        else:
            if width is None:
                width = height * GOLDEN_RATIO
        
        plt.figure(figsize=(width, height))

    def __default_plot_settings(self, title=""):
        """Set xlabel and title of the plot

        This has to be called after the plot() command
        """

        plt.xlabel("Time")
        if title:
            plt.title(title)

    def plot_temperature(self, width=None, height=None):
        """Plot the temperature"""
        self.set_plot_size(width, height)
        df = self.get_data_frame()
        (df["currT"] / 1000).plot()
        self.__default_plot_settings(title="Temperature")

    def plot_multivalue(self, values, title, width, height):
        """Plot multiple values of the DataFrame

        values is an array with the keys of the DataFrame to plot
        """
        self.set_plot_size(width, height)
        df = self.get_data_frame()
        df[values].plot()
        self.__default_plot_settings(title=title)

    def plot_input_power(self, width=None, height=None):
        """Plot input power"""
        self.plot_multivalue( ["Pa7_in", "Pa15_in", "Pgpu_in"], "Input Power", width, height)
