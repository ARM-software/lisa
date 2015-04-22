#!/usr/bin/python
# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        run.py
# ----------------------------------------------------------------
# $
#

import os
import re
import pandas as pd

import plot_utils

def _plot_freq_hists(power_inst, map_label, what, axis, title):
    """Helper function for plot_freq_hists

    power_obj is either an CpuInPower() or CpuOutPower() instance.  what is
    a string: "in" or "out"

    """
    freqs = power_inst.get_all_freqs(map_label)
    for ax, actor in zip(axis, freqs):
        this_title = "freq {} {}".format(what, actor)
        this_title = plot_utils.normalize_title(this_title, title)
        xlim = (0, freqs[actor].max())

        plot_utils.plot_hist(freqs[actor], ax, this_title, "KHz", 20,
                             "Frequency", xlim, "default")

class Run(object):
    """A wrapper class that initializes all the classes of a given run

The run class can receive the following optional parameters.

path contains the path to the trace file.  If no path is given, it
uses the current directory by default.  If path is a file, and ends in
.dat, it's run through "trace-cmd report".  If it doesn't end in
".dat", then it must be the output of a trace-cmd report run.  If path
is a directory that contains a trace.txt, that is assumed to be the
output of "trace-cmd report".  If path is a directory that doesn't
have a trace.txt but has a trace.dat, it runs trace-cmd report on the
trace.dat, saves it in trace.txt and then uses that.

name is a string describing the trace.

normalize_time is used to make all traces start from time 0 (the
default).  If normalize_time is False, the trace times are the same as
in the trace file.

scope can be used to limit the parsing done on the trace.  The default
scope parses all the traces known to cr2.  If scope is thermal, only
the thermal classes are parsed.  If scope is sched, only the sched
classes are parsed.

    """

    thermal_classes = {}

    sched_classes = {}

    dynamic_classes = {}

    def __init__(self, path=".", name="", normalize_time=True, scope="all"):
        self.name = name
        self.trace_path, self.trace_path_raw = self.__process_path(path)
        self.class_definitions = self.dynamic_classes.copy()

        if scope == "thermal":
            self.class_definitions.update(self.thermal_classes.items())
        elif scope == "sched":
            self.class_definitions.update(self.sched_classes.items())
        else:
            self.class_definitions.update(self.thermal_classes.items() +
                                          self.sched_classes.items())

        self.trace_classes = []
        for attr, class_name in self.class_definitions.iteritems():
            trace_class = globals()[class_name]()
            setattr(self, attr, trace_class)
            self.trace_classes.append(trace_class)

        self.__parse_trace_file()
        self.__parse_trace_file(raw=True)
        self.__finalize_objects()

        if normalize_time:
            basetime = self.get_basetime()
            self.normalize_time(basetime)

    def __process_path(self, basepath):
        """Process the path and return the path to the trace text file"""

        if os.path.isfile(basepath):
            trace_txt = os.path.splitext(basepath)[0]
            raw_trace_path = trace_txt + ".raw.txt"
            trace_path = trace_txt + ".txt"

            if basepath.endswith(".dat"):
                self.__run_trace_cmd_report(basepath)

            elif basepath.endswith(".txt"):
                trace_path = basepath
                if not os.path.isfile(raw_trace_path):
                    raw_trace_path = None
        else:
            trace_path = os.path.join(basepath, "trace.txt")
            raw_trace_path = os.path.join(basepath, "trace.raw.txt")
            dat_file = os.path.join(basepath, "trace.dat")

            if not os.path.isfile(trace_path):
                self.__run_trace_cmd_report(dat_file)

            # The condition below handles the the following cases
            # trace.dat and trace.txt are both present
            # We can still generate the trace.raw.txt
            if not os.path.isfile(raw_trace_path):
                if os.path.isfile(dat_file):
                    self.__run_trace_cmd_report(dat_file)
                else:
                    raw_trace_path = None

        return trace_path, raw_trace_path

    def __run_trace_cmd_report(self, fname):
        """Run "trace-cmd report fname > fname.txt"
           and "trace-cmd report -R fname > fname.raw.txt"

        The resulting traces are stored in files with extension ".txt"
        and ".raw.txt" respectively.  If fname is "my_trace.dat", the
        trace is stored in "my_trace.txt" and "my_trace.raw.txt".  The
        contents of the destination files are overwritten if they
        exist.

        """
        from subprocess import check_output

        cmd = ["trace-cmd", "report"]

        if not os.path.isfile(fname):
            raise IOError("No such file or directory: {}".format(fname))

        raw_trace_output = os.path.splitext(fname)[0] + ".raw.txt"
        trace_output = os.path.splitext(fname)[0] + ".txt"
        cmd.append(fname)

        with open(os.devnull) as devnull:
            out = check_output(cmd, stderr=devnull)

            # Add the -R flag to the trace-cmd
            # for raw parsing
            cmd.insert(-1, "-R")
            raw_out = check_output(cmd, stderr=devnull)

        with open(trace_output, "w") as fout:
            fout.write(out)

        with open(raw_trace_output, "w") as fout:
            fout.write(raw_out)

    def get_basetime(self):
        """Returns the smallest time value of all classes,
        returns 0 if the data frames of all classes are empty"""
        basetimes = []

        for trace_class in self.trace_classes:
            try:
                basetimes.append(trace_class.data_frame.index[0])
            except IndexError:
                pass

        if len(basetimes) == 0:
            return 0

        return min(basetimes)

    @classmethod
    def register_class(cls, cobject, scope="all"):
        # Add the class to the classes dictionary
        if scope == "all":
            cls.dynamic_classes[cobject.name] = cobject.__name__
        else:
            getattr(cls, scope + "_classes")[cobject.name] = cobject.__name__
        globals()[cobject.__name__] = cobject

    def get_filters(self, key=""):
        """Returns an array with the available filters.
        If 'key' is specified, returns a subset of the available filters
        that contain 'key' in their name (e.g., key="sched" returns
        only the "sched" related filters)."""
        filters = []

        for c in self.class_definitions:
            if re.search(key, c):
                filters.append(c)

        return filters

    def normalize_time(self, basetime):
        """Normalize the time of all the trace classes"""
        for trace_class in self.trace_classes:
            trace_class.normalize_time(basetime)

    def __contains_unique_word(self, line, unique_words):
        for unique_word, trace_name in unique_words:
            if unique_word in line:
                return trace_name
        return None

    def __parse_trace_file(self, raw=False):
        """parse the trace and create a pandas DataFrame"""

        # Memoize the unique words to speed up parsing the trace file
        unique_words = []
        for trace_name in self.class_definitions.iterkeys():
            parse_raw = getattr(self, trace_name).parse_raw

            if parse_raw != raw:
                continue

            unique_word = getattr(self, trace_name).unique_word
            unique_words.append((unique_word, trace_name))

        if len(unique_words) == 0:
            return

        if raw:
                if self.trace_path_raw != None:
                    trace_file = self.trace_path_raw
                else:
                    return
        else:
            trace_file = self.trace_path

        with open(trace_file) as fin:
            for line in fin:
                attr = self.__contains_unique_word(line, unique_words)
                if not attr:
                    continue

                line = line[:-1]

                special_fields_match = re.search(r"^\s+([^\[]+)-(\d+)\s+\[(\d+)\]\s+([0-9]+\.[0-9]+):",
                                                 line)
                comm = special_fields_match.group(1)
                pid = int(special_fields_match.group(2))
                cpu = int(special_fields_match.group(3))
                timestamp = float(special_fields_match.group(4))

                data_start_idx = re.search(r"[A-Za-z0-9_]+=", line).start()
                data_str = line[data_start_idx:]

                # Remove empty arrays from the trace
                data_str = re.sub(r"[A-Za-z0-9_]+=\{\} ", r"", data_str)

                getattr(self, attr).append_data(timestamp, comm, pid, cpu,
                                                data_str)

    def __finalize_objects(self):
        for trace_class in self.trace_classes:
            trace_class.create_dataframe()
            trace_class.finalize_object()

    def get_all_freqs_data(self, map_label):
        """get an array of tuple of names and DataFrames suitable for the
        allfreqs plot"""

        cpu_in_freqs = self.cpu_in_power.get_all_freqs(map_label)
        cpu_out_freqs = self.cpu_out_power.get_all_freqs(map_label)

        ret = []
        for label in map_label.values():
            in_label = label + "_freq_in"
            out_label = label + "_freq_out"

            cpu_inout_freq_dict = {in_label: cpu_in_freqs[label],
                                   out_label: cpu_out_freqs[label]}
            dfr = pd.DataFrame(cpu_inout_freq_dict).fillna(method="pad")
            ret.append((label, dfr))

        inout_freq_dict = {"gpu_freq_in": self.devfreq_in_power.get_all_freqs(),
                           "gpu_freq_out": self.devfreq_out_power.get_all_freqs()
                       }
        dfr = pd.DataFrame(inout_freq_dict).fillna(method="pad")
        ret.append(("GPU", dfr))

        return ret

    def plot_freq_hists(self, map_label, ax):
        """Plot histograms for each actor input and output frequency

        ax is an array of axis, one for the input power and one for
        the output power

        """

        num_actors = len(map_label)
        _plot_freq_hists(self.cpu_out_power, map_label, "out", ax[0:num_actors], self.name)
        _plot_freq_hists(self.cpu_in_power, map_label, "in", ax[num_actors:], self.name)

    def plot_load(self, mapping_label, title="", width=None, height=None, ax=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """

        load_data = self.cpu_in_power.get_load_data(mapping_label)
        gpu_data = pd.DataFrame({"GPU":
                                 self.devfreq_in_power.data_frame["load"]})
        load_data = pd.concat([load_data, gpu_data], axis=1)
        load_data = load_data.fillna(method="pad")
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
