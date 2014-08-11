#!/usr/bin/python

from thermal import Thermal, ThermalGovernor
from pid_controller import PIDController
from power import InPower, OutPower

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
