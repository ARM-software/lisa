#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current
directory's trace.dat"""

from matplotlib import pyplot as plt
import pandas as pd

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

    def get_load_data(self, mapping_label):
        """return a dataframe suitable for plot_load()

        mapping_label is a dictionary mapping cluster numbers to labels."""

        dfr = self.get_data_frame()
        load_cols = [s for s in dfr.columns if s.startswith("load")]

        load_series = dfr[load_cols[0]]
        for col in load_cols[1:]:
            load_series += dfr[col]

        load_dfr = pd.DataFrame({"cluster": dfr["cluster"], "load": load_series})
        cluster_numbers = set(dfr["cluster"])

        ret_dict = {}
        for num in cluster_numbers:
            label = mapping_label[num]
            cluster_load = load_dfr[dfr["cluster"] == num]["load"]

            ret_dict[label] = cluster_load

        return pd.DataFrame(ret_dict).fillna(method="pad")

    def plot_cluster_load(self, cluster):
        df = self.get_cluster_data_frame(cluster)
        load_cols = [s for s in df.columns if s.startswith("load")]

        self.pre_plot_setup()
        df[load_cols].plot(ax=self.ax)
        self.post_plot_setup(ylim=(0, 110))

    def plot_load(self, mapping_label, title="Utilisation", width=None, height=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """

        load_data = self.get_load_data(mapping_label)

        self.pre_plot_setup(width=width, height=height)
        load_data.plot(ax=self.ax)
        self.post_plot_setup(title=title)
