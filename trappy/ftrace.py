#    Copyright 2015-2016 ARM Limited
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


# pylint can't see any of the dynamically allocated classes of FTrace
# pylint: disable=no-member

import itertools
import os
import re
import pandas as pd

from trappy.bare_trace import BareTrace
from trappy.utils import listify

def _plot_freq_hists(allfreqs, what, axis, title):
    """Helper function for plot_freq_hists

    allfreqs is the output of a Cpu*Power().get_all_freqs() (for
    example, CpuInPower.get_all_freqs()).  what is a string: "in" or
    "out"

    """
    import trappy.plot_utils

    for ax, actor in zip(axis, allfreqs):
        this_title = "freq {} {}".format(what, actor)
        this_title = trappy.plot_utils.normalize_title(this_title, title)
        xlim = (0, allfreqs[actor].max())

        trappy.plot_utils.plot_hist(allfreqs[actor], ax, this_title, "KHz", 20,
                             "Frequency", xlim, "default")

class GenericFTrace(BareTrace):
    """Generic class to parse output of FTrace.  This class is meant to be
subclassed by FTrace (for parsing FTrace coming from trace-cmd) and SysTrace."""

    thermal_classes = {}

    sched_classes = {}

    dynamic_classes = {}

    def __init__(self, name="", normalize_time=True, scope="all",
                 events=[], window=(0, None), abs_window=(0, None)):
        super(GenericFTrace, self).__init__(name)

        if not hasattr(self, "needs_raw_parsing"):
            self.needs_raw_parsing = False

        self.class_definitions.update(self.dynamic_classes.items())
        self.__add_events(listify(events))

        if scope == "thermal":
            self.class_definitions.update(self.thermal_classes.items())
        elif scope == "sched":
            self.class_definitions.update(self.sched_classes.items())
        elif scope != "custom":
            self.class_definitions.update(self.thermal_classes.items() +
                                          self.sched_classes.items())

        for attr, class_def in self.class_definitions.iteritems():
            trace_class = class_def()
            setattr(self, attr, trace_class)
            self.trace_classes.append(trace_class)

        self.__parse_trace_file(self.trace_path, window, abs_window)
        if self.needs_raw_parsing and (self.trace_path_raw is not None):
            self.__parse_trace_file(self.trace_path_raw, window, abs_window,
                                    raw=True)
        self.finalize_objects()

        if normalize_time:
            self.normalize_time()

    @classmethod
    def register_parser(cls, cobject, scope):
        """Register the class as an Event. This function
        can be used to register a class which is associated
        with an FTrace unique word.

        .. seealso::

            :mod:`trappy.dynamic.register_dynamic_ftrace` :mod:`trappy.dynamic.register_ftrace_parser`

        """

        if not hasattr(cobject, "name"):
            cobject.name = cobject.unique_word.split(":")[0]

        # Add the class to the classes dictionary
        if scope == "all":
            cls.dynamic_classes[cobject.name] = cobject
        else:
            getattr(cls, scope + "_classes")[cobject.name] = cobject

    @classmethod
    def unregister_parser(cls, cobject):
        """Unregister a parser

        This is the opposite of FTrace.register_parser(), it removes a class
        from the list of classes that will be parsed on the trace

        """

        # TODO: scopes should not be hardcoded (nor here nor in the FTrace object)
        all_scopes = [cls.thermal_classes, cls.sched_classes,
                      cls.dynamic_classes]
        known_events = ((n, c, sc) for sc in all_scopes for n, c in sc.items())

        for name, obj, scope_classes in known_events:
            if cobject == obj:
                del scope_classes[name]

    def __add_events(self, events):
        """Add events to the class_definitions

        If the events are known to trappy just add that class to the
        class definitions list.  Otherwise, register a class to parse
        that event

        """

        from trappy.dynamic import DynamicTypeFactory, default_init
        from trappy.base import Base

        # TODO: scopes should not be hardcoded (nor here nor in the FTrace object)
        all_scopes = [self.thermal_classes, self.sched_classes,
                      self.dynamic_classes]
        known_events = {k: v for sc in all_scopes for k, v in sc.iteritems()}

        for event_name in events:
            for cls in known_events.itervalues():
                if (event_name == cls.unique_word) or \
                   (event_name + ":" == cls.unique_word):
                    self.class_definitions[event_name] = cls
                    break
            else:
                kwords = {
                    "__init__": default_init,
                    "unique_word": event_name + ":",
                    "name": event_name,
                }
                trace_class = DynamicTypeFactory(event_name, (Base,), kwords)
                self.class_definitions[event_name] = trace_class

    def __populate_data(self, fin, cls_for_unique_word, window, abs_window):
        """Append to trace data from a txt trace"""

        def contains_unique_word(line, unique_words=cls_for_unique_word.keys()):
            for unique_word in unique_words:
                if unique_word in line:
                    return True
            return False

        special_fields_regexp = r"^\s*(?P<comm>.*)-(?P<pid>\d+)(?:\s+\(.*\))"\
                                r"?\s+\[(?P<cpu>\d+)\](?:\s+....)?\s+"\
                                r"(?P<timestamp>[0-9]+\.[0-9]+):"
        special_fields_regexp = re.compile(special_fields_regexp)
        start_match = re.compile(r"[A-Za-z0-9_]+=")

        actual_trace = itertools.dropwhile(self.trace_hasnt_started(), fin)
        actual_trace = itertools.takewhile(self.trace_hasnt_finished(),
                                           actual_trace)

        for line in itertools.ifilter(contains_unique_word, actual_trace):
            for unique_word, cls in cls_for_unique_word.iteritems():
                if unique_word in line:
                    trace_class = cls
                    break
            else:
                raise ValueError("No unique in {}".format(line))

            line = line[:-1]

            special_fields_match = special_fields_regexp.match(line)
            comm = special_fields_match.group('comm')
            pid = int(special_fields_match.group('pid'))
            cpu = int(special_fields_match.group('cpu'))
            timestamp = float(special_fields_match.group('timestamp'))

            if not self.basetime:
                self.basetime = timestamp

            if (timestamp < window[0] + self.basetime) or \
               (timestamp < abs_window[0]):
                continue

            if (window[1] and timestamp > window[1] + self.basetime) or \
               (abs_window[1] and timestamp > abs_window[1]):
                return

            try:
                data_start_idx =  start_match.search(line).start()
            except AttributeError:
                continue

            data_str = line[data_start_idx:]

            # Remove empty arrays from the trace
            data_str = re.sub(r"[A-Za-z0-9_]+=\{\} ", r"", data_str)

            trace_class.append_data(timestamp, comm, pid, cpu, data_str)

    def trace_hasnt_started(self):
        """Return a function that accepts a line and returns true if this line
is not part of the trace.

        Subclasses of GenericFTrace may override this to skip the
        beginning of a file that is not part of the trace.  The first
        time the returned function returns False it will be considered
        the beginning of the trace and this function will never be
        called again (because once it returns False, the trace has
        started).

        """
        return lambda x: False

    def trace_hasnt_finished(self):
        """Return a function that accepts a line and returns true if this line
is part of the trace.

        This function is called with each line of the file *after*
        trace_hasnt_started() returns True so the first line it sees
        is part of the trace.  The returned function should return
        True as long as the line it receives is part of the trace.  As
        soon as this function returns False, the rest of the file will
        be dropped.  Subclasses of GenericFTrace may override this to
        stop processing after the end of the trace is found to skip
        parsing the end of the file if it contains anything other than
        trace.

        """
        return lambda x: True

    def __parse_trace_file(self, trace_file, window, abs_window, raw=False):
        """parse the trace and create a pandas DataFrame"""

        # Memoize the unique words to speed up parsing the trace file
        cls_for_unique_word = {}
        for trace_name in self.class_definitions.iterkeys():
            trace_class = getattr(self, trace_name)

            if self.needs_raw_parsing and (trace_class.parse_raw != raw):
                continue

            unique_word = trace_class.unique_word
            cls_for_unique_word[unique_word] = trace_class

        if len(cls_for_unique_word) == 0:
            return

        with open(trace_file) as fin:
            self.__populate_data(fin, cls_for_unique_word, window, abs_window)

    # TODO: Move thermal specific functionality

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

        try:
            gpu_freq_in_data = self.devfreq_in_power.get_all_freqs()
            gpu_freq_out_data = self.devfreq_out_power.get_all_freqs()
        except KeyError:
            gpu_freq_in_data = gpu_freq_out_data = None

        if gpu_freq_in_data is not None:
            inout_freq_dict = {"gpu_freq_in": gpu_freq_in_data["freq"],
                               "gpu_freq_out": gpu_freq_out_data["freq"]
                           }
            dfr = pd.DataFrame(inout_freq_dict).fillna(method="pad")
            ret.append(("GPU", dfr))

        return ret

    def plot_freq_hists(self, map_label, ax):
        """Plot histograms for each actor input and output frequency

        ax is an array of axis, one for the input power and one for
        the output power

        """

        in_base_idx = len(ax) / 2

        try:
            devfreq_out_all_freqs = self.devfreq_out_power.get_all_freqs()
            devfreq_in_all_freqs = self.devfreq_in_power.get_all_freqs()
        except KeyError:
            devfreq_out_all_freqs = None
            devfreq_in_all_freqs = None

        out_allfreqs = (self.cpu_out_power.get_all_freqs(map_label),
                        devfreq_out_all_freqs, ax[0:in_base_idx])
        in_allfreqs = (self.cpu_in_power.get_all_freqs(map_label),
                       devfreq_in_all_freqs, ax[in_base_idx:])

        for cpu_allfreqs, devfreq_freqs, axis in (out_allfreqs, in_allfreqs):
            if devfreq_freqs is not None:
                devfreq_freqs.name = "GPU"
                allfreqs = pd.concat([cpu_allfreqs, devfreq_freqs], axis=1)
            else:
                allfreqs = cpu_allfreqs

            allfreqs.fillna(method="pad", inplace=True)
            _plot_freq_hists(allfreqs, "out", axis, self.name)

    def plot_load(self, mapping_label, title="", width=None, height=None,
                  ax=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """
        import trappy.plot_utils

        load_data = self.cpu_in_power.get_load_data(mapping_label)
        try:
            gpu_data = pd.DataFrame({"GPU":
                                     self.devfreq_in_power.data_frame["load"]})
            load_data = pd.concat([load_data, gpu_data], axis=1)
        except KeyError:
            pass

        load_data = load_data.fillna(method="pad")
        title = trappy.plot_utils.normalize_title("Utilization", title)

        if not ax:
            ax = trappy.plot_utils.pre_plot_setup(width=width, height=height)

        load_data.plot(ax=ax)

        trappy.plot_utils.post_plot_setup(ax, title=title)

    def plot_normalized_load(self, mapping_label, title="", width=None,
                             height=None, ax=None):
        """plot the normalized load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """
        import trappy.plot_utils

        load_data = self.cpu_in_power.get_normalized_load_data(mapping_label)
        if "load" in self.devfreq_in_power.data_frame:
            gpu_dfr = self.devfreq_in_power.data_frame
            gpu_max_freq = max(gpu_dfr["freq"])
            gpu_load = gpu_dfr["load"] * gpu_dfr["freq"] / gpu_max_freq

            gpu_data = pd.DataFrame({"GPU": gpu_load})
            load_data = pd.concat([load_data, gpu_data], axis=1)

        load_data = load_data.fillna(method="pad")
        title = trappy.plot_utils.normalize_title("Normalized Utilization", title)

        if not ax:
            ax = trappy.plot_utils.pre_plot_setup(width=width, height=height)

        load_data.plot(ax=ax)

        trappy.plot_utils.post_plot_setup(ax, title=title)

    def plot_allfreqs(self, map_label, width=None, height=None, ax=None):
        """Do allfreqs plots similar to those of CompareRuns

        if ax is not none, it must be an array of the same size as
        map_label.  Each plot will be done in each of the axis in
        ax

        """
        import trappy.plot_utils

        all_freqs = self.get_all_freqs_data(map_label)

        setup_plot = False
        if ax is None:
            ax = [None] * len(all_freqs)
            setup_plot = True

        for this_ax, (label, dfr) in zip(ax, all_freqs):
            this_title = trappy.plot_utils.normalize_title("allfreqs " + label,
                                                        self.name)

            if setup_plot:
                this_ax = trappy.plot_utils.pre_plot_setup(width=width,
                                                        height=height)

            dfr.plot(ax=this_ax)
            trappy.plot_utils.post_plot_setup(this_ax, title=this_title)

class FTrace(GenericFTrace):
    """A wrapper class that initializes all the classes of a given run

    - The FTrace class can receive the following optional parameters.

    :param path: Path contains the path to the trace file.  If no path is given, it
        uses the current directory by default.  If path is a file, and ends in
        .dat, it's run through "trace-cmd report".  If it doesn't end in
        ".dat", then it must be the output of a trace-cmd report run.  If path
        is a directory that contains a trace.txt, that is assumed to be the
        output of "trace-cmd report".  If path is a directory that doesn't
        have a trace.txt but has a trace.dat, it runs trace-cmd report on the
        trace.dat, saves it in trace.txt and then uses that.

    :param name: is a string describing the trace.

    :param normalize_time: is used to make all traces start from time 0 (the
        default).  If normalize_time is False, the trace times are the same as
        in the trace file.

    :param scope: can be used to limit the parsing done on the trace.  The default
        scope parses all the traces known to trappy.  If scope is thermal, only
        the thermal classes are parsed.  If scope is sched, only the sched
        classes are parsed.

    :param events: A list of strings containing the name of the trace
        events that you want to include in this FTrace object.  The
        string must correspond to the event name (what you would pass
        to "trace-cmd -e", i.e. 4th field in trace.txt)

    :param window: a tuple indicating a time window.  The first
        element in the tuple is the start timestamp and the second one
        the end timestamp.  Timestamps are relative to the first trace
        event that's parsed.  If you want to trace until the end of
        the trace, set the second element to None.  If you want to use
        timestamps extracted from the trace file use "abs_window". The
        window is inclusive: trace events exactly matching the start
        or end timestamps will be included.

    :param abs_window: a tuple indicating an absolute time window.
        This parameter is similar to the "window" one but its values
        represent timestamps that are not normalized, (i.e. the ones
        you find in the trace file). The window is inclusive.


    :type path: str
    :type name: str
    :type normalize_time: bool
    :type scope: str
    :type events: list
    :type window: tuple
    :type abs_window: tuple

    This is a simple example:
    ::

        import trappy
        trappy.FTrace("trace_dir")

    """

    def __init__(self, path=".", name="", normalize_time=True, scope="all",
                 events=[], window=(0, None), abs_window=(0, None)):
        self.trace_path, self.trace_path_raw = self.__process_path(path)
        self.needs_raw_parsing = True

        self.__populate_metadata()

        super(FTrace, self).__init__(name, normalize_time, scope, events,
                                     window, abs_window)

    def __process_path(self, basepath):
        """Process the path and return the path to the trace text file"""

        if os.path.isfile(basepath):
            trace_name = os.path.splitext(basepath)[0]
        else:
            trace_name = os.path.join(basepath, "trace")

        trace_txt = trace_name + ".txt"
        trace_raw = trace_name + ".raw.txt"
        trace_dat = trace_name + ".dat"

        if os.path.isfile(trace_dat):
            # Both TXT and RAW traces must always be generated
            if not os.path.isfile(trace_txt) or \
               not os.path.isfile(trace_raw):
                self.__run_trace_cmd_report(trace_dat)
            # TXT (and RAW) traces must match the most recent binary trace
            elif os.path.getmtime(trace_txt) < os.path.getmtime(trace_dat):
                self.__run_trace_cmd_report(trace_dat)

        if not os.path.isfile(trace_raw):
            trace_raw = None

        return trace_txt, trace_raw

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
            try:
                out = check_output(cmd, stderr=devnull)
            except OSError as exc:
                if exc.errno == 2 and not exc.filename:
                    raise OSError(2, "trace-cmd not found in PATH, is it installed?")
                else:
                    raise

            # Add the -R flag to the trace-cmd
            # for raw parsing
            cmd.insert(-1, "-R")
            raw_out = check_output(cmd, stderr=devnull)

        with open(trace_output, "w") as fout:
            fout.write(out)

        with open(raw_trace_output, "w") as fout:
            fout.write(raw_out)

    def __populate_metadata(self):
        """Populates trace metadata"""

        # Meta Data as expected to be found in the parsed trace header
        metadata_keys = ["version", "cpus"]

        for key in metadata_keys:
            setattr(self, "_" + key, None)

        with open(self.trace_path) as fin:
            for line in fin:
                if not metadata_keys:
                    return

                metadata_pattern = r"^\b(" + "|".join(metadata_keys) + \
                                   r")\b\s*=\s*([0-9]+)"
                match = re.search(metadata_pattern, line)
                if match:
                    setattr(self, "_" + match.group(1), match.group(2))
                    metadata_keys.remove(match.group(1))

                if re.search(r"^\s+[^\[]+-\d+\s+\[\d+\]\s+\d+\.\d+:", line):
                    # Reached a valid trace line, abort metadata population
                    return
