#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current
directory's trace.dat"""

from thermal import BaseThermal
from matplotlib import pyplot as plt

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

    def plot_cluster_load(self, cluster):
        df = self.get_cluster_data_frame(cluster)
        load_cols = [s for s in df.columns if s.startswith("load")]

        _, ax = plt.subplots()
        df[load_cols].plot(ax=ax)
        ax.set_ylim(0, 110)
