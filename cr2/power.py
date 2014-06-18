#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current
directory's trace.dat"""

from matplotlib import pyplot as plt
import pandas as pd

from thermal import BaseThermal
from plot_utils import normalize_title

def pivot_with_labels(dfr, data_col_name, new_col_name, mapping_label):
    """Pivot a DataFrame row into columns

    dfr is the DataFrame to operate on.  data_col_name is the name of
    the column in the DataFrame which contains the values.
    new_col_name is the name of the column in the DataFrame that will
    became the new columns.  mapping_label is a dictionary whose keys
    are the values in new_col_name and whose values are their
    corresponding name in the DataFrame to be returned.

    There has to be a more "pandas" way of doing this.

    Example: XXX

    In [8]: dfr_in = pd.DataFrame({'cpus': ["000000f0", "0000000f", "000000f0", "0000000f"], 'freq': [1, 3, 2, 6]})

    In [9]: dfr_in
    Out[9]:
           cpus  freq
    0  000000f0     1
    1  0000000f     3
    2  000000f0     2
    3  0000000f     6

    [4 rows x 2 columns]

    In [10]: map_label = {"000000f0": "A15", "0000000f": "A7"}

    In [11]: power.pivot_with_labels(dfr_in, "freq", "cpus", map_label)
    Out[11]:
       A15  A7
    0    1 NaN
    1    1   3
    2    2   3
    3    2   6

    [4 rows x 2 columns]
    """

    col_set = set(dfr[new_col_name])

    ret_series = {}
    for col in col_set:
        label = mapping_label[col]
        data = dfr[dfr[new_col_name] == col][data_col_name]

        ret_series[label] = data

    return pd.DataFrame(ret_series).fillna(method="pad")

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

        return pivot_with_labels(load_dfr, "load", "cluster", mapping_label)

    def plot_cluster_load(self, cluster):
        df = self.get_cluster_data_frame(cluster)
        load_cols = [s for s in df.columns if s.startswith("load")]

        self.pre_plot_setup()
        df[load_cols].plot(ax=self.ax)
        self.post_plot_setup(ylim=(0, 110))

    def plot_load(self, mapping_label, title="", width=None, height=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """

        load_data = self.get_load_data(mapping_label)
        title = normalize_title("Utilisation", title)

        self.pre_plot_setup(width=width, height=height)
        load_data.plot(ax=self.ax)
        self.post_plot_setup(title=title)
