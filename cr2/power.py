#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current
directory's trace.dat"""

from thermal import BaseThermal

class OutPower(BaseThermal):
    """Process the cpufreq cooling power actor data in a ftrace dump"""

    def __init__(self, path=None):
        super(OutPower, self).__init__(
            basepath=path,
            unique_word="thermal_power_limit",
        )

class InPower(BaseThermal):
    """Process the cpufreq cooling power actor data in a ftrace dump"""

    def __init__(self, path=None):
        super(InPower, self).__init__(
            basepath=path,
            unique_word="raw_cpu_power",
        )

    def get_cluster_data_frame(self, cluster):
        df = self.get_data_frame()

        return df[df["cluster"] == cluster]
