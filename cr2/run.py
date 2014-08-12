#!/usr/bin/python

import pandas as pd

from thermal import Thermal, ThermalGovernor
from pid_controller import PIDController
from power import InPower, OutPower
import plot_utils

def _plot_power_hists(power_inst, map_label, what, title):
    """Helper function for plot_power_hists

    power_obj is either an InPower() or OutPower() instance.  what is
    a string: "in" or "out"

    """
    freqs = power_inst.get_all_freqs(map_label)
    for actor in freqs:
        this_title = "freq {} {}".format(what, actor)
        this_title = plot_utils.normalize_title(this_title, title)
        xlim = (0, freqs[actor].max())

        plot_utils.plot_hist(freqs[actor], this_title, 20, "Frequency (KHz)",
                             xlim, "default")

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
        """get a dict of DataFrames suitable for the allfreqs plot"""

        in_freqs = self.in_power.get_all_freqs(map_label)
        out_freqs = self.out_power.get_all_freqs(map_label)

        ret_dict = {}
        for label in map_label.values():
            in_label = label + "_freq_in"
            out_label = label + "_freq_out"

            inout_freq_dict = {in_label: in_freqs[label], out_label: out_freqs[label]}
            ret_dict[label] = pd.DataFrame(inout_freq_dict).fillna(method="pad")

        return ret_dict

    def plot_power_hists(self, map_label, title=""):
        """Plot histograms for each actor input and output power"""

        _plot_power_hists(self.out_power, map_label, "out", title)
        _plot_power_hists(self.in_power, map_label, "in", title)

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

        for this_ax, label in zip(ax, all_freqs):
            dfr = all_freqs[label]
            this_title = plot_utils.normalize_title("allfreqs " + label, self.name)

            if setup_plot:
                this_ax = plot_utils.pre_plot_setup(width=width, height=height)

            dfr.plot(ax=this_ax)
            plot_utils.post_plot_setup(this_ax, title=this_title)
