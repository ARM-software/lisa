#!/usr/bin/python

import pandas as pd

from thermal import Thermal, ThermalGovernor
from pid_controller import PIDController
from power import InPower, OutPower
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

    classes = {"thermal": "Thermal",
               "thermal_governor": "ThermalGovernor",
               "pid_controller": "PIDController",
               "in_power": "InPower",
               "out_power": "OutPower",
    }

    def __init__(self, path=None, name=""):
        self.name = name
        for attr, class_name in self.classes.iteritems():
            setattr(self, attr, globals()[class_name](path))

    def normalize_time(self, basetime):
        """Normalize the time of all the trace classes"""
        for attr in self.classes.iterkeys():
            getattr(self, attr).normalize_time(basetime)

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
