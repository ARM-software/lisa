#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current
directory's trace.dat"""

from thermal import BaseThermal

class Power(BaseThermal):
    """Process the cpufreq cooling power actor data in a ftrace dump"""

    def __init__(self, path=None):
        super(Power, self).__init__(
            basepath=path,
            unique_word="thermal_power_limit",
        )
