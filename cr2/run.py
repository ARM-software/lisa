#!/usr/bin/python

import os
import re
import pandas as pd

from thermal import Thermal, ThermalGovernor
from pid_controller import PIDController
from power import InPower, OutPower
from sched import *
import plot_utils

def _plot_freq_hists(power_inst, map_label, what, axis, title):
    """Helper function for plot_freq_hists

    power_obj is either an InPower() or OutPower() instance.  what is
    a string: "in" or "out"

    """
    freqs = power_inst.get_all_freqs(map_label)
    for ax, actor in zip(axis, freqs):
        this_title = "freq {} {}".format(what, actor)
        this_title = plot_utils.normalize_title(this_title, title)
        xlim = (0, freqs[actor].max())

        plot_utils.plot_hist(freqs[actor], ax, this_title, 20,
                             "Frequency (KHz)", xlim, "default")

class Run(object):
    """A wrapper class that initializes all the classes of a given run"""

    thermal_classes = {
                "thermal": "Thermal",
                "thermal_governor": "ThermalGovernor",
                "pid_controller": "PIDController",
                "in_power": "InPower",
                "out_power": "OutPower",
    }

    sched_classes = {
                "sched_load_avg_sched_group": "SchedLoadAvgSchedGroup",
                "sched_load_avg_task": "SchedLoadAvgTask",
                "sched_load_avg_cpu": "SchedLoadAvgCpu",
                "sched_contrib_scale_factor": "SchedContribScaleFactor",
                "sched_cpu_capacity": "SchedCpuCapacity",
                "sched_cpu_frequency": "SchedCpuFrequency",
    }

    classes = {}

    def __init__(self, path=None, name="", normalize_time=True, scope="all"):

        if path is None:
            path = "."
        self.name = name
        self.basepath = path

        if scope == "thermal":
            self.classes = dict(self.thermal_classes.items())
        elif scope == "sched":
            self.classes = dict(self.sched_classes.items())
        else:
            self.classes = dict(self.thermal_classes.items() + self.sched_classes.items())

        for attr, class_name in self.classes.iteritems():
            setattr(self, attr, globals()[class_name](path))

        self.__parse_trace_file()
        self.__finalize_objects()

        if normalize_time:
            basetime = self.get_basetime()
            self.normalize_time(basetime)

    def get_basetime(self):
        """Returns the smallest time value of all classes,
        returns 0 if the data frames of all classes are empty"""
        basetimes = []

        for attr in self.classes.iterkeys():
            try:
                basetimes.append(getattr(self, attr).data_frame.index[0])
            except IndexError:
                pass

        if len(basetimes) == 0:
            return 0

        return min(basetimes)

    def normalize_time(self, basetime):
        """Normalize the time of all the trace classes"""
        for attr in self.classes.iterkeys():
            getattr(self, attr).normalize_time(basetime)

    def __contains_unique_word(self, line):
        for attr in self.classes.iterkeys():
            if re.search(getattr(self, attr).unique_word, line):
                return attr;
        return None;

    def __parse_trace_file(self):
        """parse the trace and create a pandas DataFrame"""

        fin_fname = os.path.join(self.basepath, "trace.txt")

        pat_timestamp = re.compile(r"([0-9]+\.[0-9]+):")
        pat_data_start = re.compile("[A-Za-z0-9_]+=")
        pat_empty_array = re.compile(r"[A-Za-z0-9_]+=\{\} ")

        with open(fin_fname) as fin:
            for line in fin:
                attr = self.__contains_unique_word(line)
                if not attr:
                    continue

                line = line[:-1]

                timestamp_match = re.search(pat_timestamp, line)
                timestamp = float(timestamp_match.group(1))

                data_start_idx = re.search(pat_data_start, line).start()
                data_str = line[data_start_idx:]

                # Remove empty arrays from the trace
                data_str = re.sub(pat_empty_array, r"", data_str)

                getattr(self, attr).append_data(timestamp, data_str)

    def __finalize_objects(self):
        for attr in self.classes.iterkeys():
            getattr(self, attr).create_dataframe()
            getattr(self, attr).finalize_object()

    def get_all_freqs_data(self, map_label):
        """get an array of tuple of names and DataFrames suitable for the
        allfreqs plot"""

        in_freqs = self.in_power.get_all_freqs(map_label)
        out_freqs = self.out_power.get_all_freqs(map_label)

        ret = []
        for label in map_label.values():
            in_label = label + "_freq_in"
            out_label = label + "_freq_out"

            inout_freq_dict = {in_label: in_freqs[label],
                               out_label: out_freqs[label]}
            dfr = pd.DataFrame(inout_freq_dict).fillna(method="pad")
            ret.append((label, dfr))

        return ret

    def plot_freq_hists(self, map_label, ax):
        """Plot histograms for each actor input and output frequency

        ax is an array of axis, one for the input power and one for
        the output power

        """

        num_actors = len(map_label)
        _plot_freq_hists(self.out_power, map_label, "out", ax[0:num_actors], self.name)
        _plot_freq_hists(self.in_power, map_label, "in", ax[num_actors:], self.name)

    def plot_load(self, mapping_label, title="", width=None, height=None, ax=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """

        load_data = self.in_power.get_load_data(mapping_label)
        title = plot_utils.normalize_title("Utilisation", title)

        if not ax:
            ax = plot_utils.pre_plot_setup(width=width, height=height)

        load_data.plot(ax=ax)

        plot_utils.post_plot_setup(ax, title=title)

    def plot_allfreqs(self, map_label, width=None, height=None, ax=None):
        """Do allfreqs plots similar to those of CompareRuns

        if ax is not none, it must be an array of the same size as
        map_label.  Each plot will be done in each of the axis in
        ax

        """
        all_freqs = self.get_all_freqs_data(map_label)

        setup_plot = False
        if ax is None:
            ax = [None] * len(all_freqs)
            setup_plot = True

        for this_ax, (label, dfr) in zip(ax, all_freqs):
            this_title = plot_utils.normalize_title("allfreqs " + label, self.name)

            if setup_plot:
                this_ax = plot_utils.pre_plot_setup(width=width, height=height)

            dfr.plot(ax=this_ax)
            plot_utils.post_plot_setup(this_ax, title=this_title)
