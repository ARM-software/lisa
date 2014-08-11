#!/usr/bin/python

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

    def __init__(self, path=None):
        for name, class_name in self.classes.iteritems():
            setattr(self, name, globals()[class_name](path))

    def normalize_time(self, basetime):
        """Normalize the time of all the trace classes"""
        for attr in self.classes.iterkeys():
            getattr(self, attr).normalize_time(basetime)

    def plot_power_hists(self, map_label, title=""):
        """Plot histograms for each actor input and output power"""

        _plot_power_hists(self.out_power, map_label, "out", title)
        _plot_power_hists(self.in_power, map_label, "in", title)
