#!/usr/bin/python
"""Process the output of the power allocator trace in the current
directory's trace.dat"""

import os
import re
from StringIO import StringIO
import pandas as pd
from matplotlib import pyplot as plt

from plot_utils import normalize_title, pre_plot_setup, post_plot_setup

class BaseThermal(object):
    """Base class to parse trace.dat dumps.

    Don't use directly, create a subclass that defines the unique_word
    you want to match in the output"""
    def __init__(self, basepath, unique_word):
        if basepath is None:
            basepath = "."

        self.basepath = basepath
        self.data_csv = ""
        self.data_frame = None
        self.unique_word = unique_word

        if not os.path.isfile(os.path.join(basepath, "trace.txt")):
            self.__run_trace_cmd_report()

    def __run_trace_cmd_report(self):
        """Run "trace-cmd report > trace.txt".

        Overwrites the contents of trace.txt if it exists."""
        from subprocess import check_output

        if not os.path.isfile(os.path.join(self.basepath, "trace.dat")):
            raise IOError("No such file or directory: trace.dat")

        previous_path = os.getcwd()
        os.chdir(self.basepath)

        # This would better be done with a context manager (i.e.
        # http://stackoverflow.com/a/13197763/970766)
        try:
            with open(os.devnull) as devnull:
                out = check_output(["trace-cmd", "report"], stderr=devnull)

        finally:
            os.chdir(previous_path)

        with open(os.path.join(self.basepath, "trace.txt"), "w") as fout:
            fout.write(out)

    def parse_into_csv(self):
        """Create a csv representation of the thermal data and store
        it in self.data_csv"""
        pat_timestamp = re.compile(r"([0-9]+\.[0-9]+):")
        pat_data = re.compile(r"[A-Za-z0-9_]+=([^ ]+)")
        pat_header = re.compile(r"([A-Za-z0-9_]+)=[^ ]+")
        header = ""

        with open(os.path.join(self.basepath, "trace.txt")) as fin:
            for line in fin:
                if not re.search(self.unique_word, line):
                    continue

                line = line[:-1]

                timestamp_match = re.search(pat_timestamp, line)
                timestamp = timestamp_match.group(1)

                data_start_idx = re.search(r"[A-Za-z0-9_]+=", line).start()
                data_str = line[data_start_idx:]

                if not header:
                    header = re.sub(pat_header, r"\1", data_str)
                    header = re.sub(r" ", r",", header)
                    header = "Time," + header + "\n"
                    self.data_csv = header

                parsed_data = re.sub(pat_data, r"\1", data_str)
                parsed_data = re.sub(r" ", r",", parsed_data)

                parsed_data = timestamp + "," + parsed_data + "\n"
                self.data_csv += parsed_data

    def get_data_frame(self):
        """Return a pandas data frame for the run"""
        if self.data_frame is not None:
            return self.data_frame

        if not self.data_csv:
            self.parse_into_csv()

        if self.data_csv is "":
            return pd.DataFrame()

        unordered_df = pd.read_csv(StringIO(self.data_csv))
        self.data_frame = unordered_df.set_index("Time")

        return self.data_frame

    def plot_multivalue(self, values, title, width, height):
        """Plot multiple values of the DataFrame

        values is an array with the keys of the DataFrame to plot
        """

        dfr = self.get_data_frame()

        ax = pre_plot_setup(width, height)
        dfr[values].plot(ax=ax)
        post_plot_setup(ax, title=title)

class Thermal(BaseThermal):
    """Process the thermal framework data in a ftrace dump"""
    def __init__(self, path=None):
        super(Thermal, self).__init__(
            basepath=path,
            unique_word="thermal_zone=",
        )

    def plot_temperature(self, title="", width=None, height=None, ylim="range"):
        """Plot the temperature"""
        dfr = self.get_data_frame()
        title = normalize_title("Temperature", title)

        ax = pre_plot_setup(width, height)
        (dfr["temp"] / 1000).plot(ax=ax)
        post_plot_setup(ax, title=title, ylim=ylim)

        plt.legend()

class ThermalGovernor(BaseThermal):
    """Process the power allocator data in a ftrace dump"""
    def __init__(self, path=None):
        super(ThermalGovernor, self).__init__(
            basepath=path,
            unique_word="Ptot_out",
        )

    def write_thermal_csv(self):
        """Write the csv info in thermal.csv"""
        if not self.data_csv:
            self.parse_into_csv()

        with open("thermal.csv", "w") as fout:
            fout.write(self.data_csv)

    def plot_temperature(self, title="", width=None, height=None, ylim="range"):
        """Plot the temperature"""
        dfr = self.get_data_frame()
        control_temp_series = (dfr["currT"] + dfr["deltaT"]) / 1000
        title = normalize_title("Temperature", title)

        ax = pre_plot_setup(width, height)

        (dfr["currT"] / 1000).plot(ax=ax)
        control_temp_series.plot(ax=ax, color="y", linestyle="--",
                                 label="control temperature")

        post_plot_setup(ax, title=title, ylim=ylim)
        plt.legend()

    def plot_input_power(self, title="", width=None, height=None):
        """Plot input power"""
        dfr = self.get_data_frame()
        in_cols = [s for s in dfr.columns
                   if re.match("P.*_in", s) and s != "Ptot_in"]

        title = normalize_title("Input Power", title)
        self.plot_multivalue(in_cols, title, width, height)

    def plot_output_power(self, title="", width=None, height=None):
        """Plot output power"""
        dfr = self.get_data_frame()
        out_cols = [s for s in dfr.columns
                   if re.match("P.*_out", s) and s != "Ptot_out"]

        title = normalize_title("Output Power", title)
        self.plot_multivalue(out_cols,
                             title, width, height)

    def plot_inout_power(self, title="", width=None, height=None):
        """Make multiple plots showing input and output power for each actor"""
        dfr = self.get_data_frame()

        actors = []
        for col in dfr.columns:
                match = re.match("P(.*)_in", col)
                if match and col != "Ptot_in":
                    actors.append(match.group(1))

        for actor in actors:
            cols = ["P" + actor + "_in", "P" + actor + "_out"]
            this_title = normalize_title(actor, title)
            dfr[cols].plot(title=this_title)
