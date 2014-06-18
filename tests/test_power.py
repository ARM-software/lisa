#!/usr/bin/python

import pandas as pd

from test_thermal import TestThermalBase
from cr2 import OutPower, InPower
import power

class TestPower(TestThermalBase):
    def test_pivot_with_labels(self):
        """Test pivot_with_labels()"""
        dfr_in = pd.DataFrame({'cpus': ["000000f0", "0000000f", "000000f0", "0000000f"],
                               'freq': [1, 3, 2, 6]})
        map_label = {"000000f0": "A15", "0000000f": "A7"}

        dfr_out = power.pivot_with_labels(dfr_in, "freq", "cpus", map_label)

        self.assertEquals(dfr_out["A15"].iloc[0], 1)
        self.assertEquals(dfr_out["A15"].iloc[1], 1)
        self.assertEquals(dfr_out["A15"].iloc[2], 2)
        self.assertEquals(dfr_out["A7"].iloc[1], 3)

    def test_outpower_get_dataframe(self):
        """Test OutPower.get_data_frame()"""
        df = OutPower().get_data_frame()

        self.assertEquals(df["power"].iloc[0], 5036)
        print df.columns
        self.assertTrue("cdev_state" in df.columns)

    def test_inpower_get_dataframe(self):
        """Test InPower.get_data_frame()"""
        df = InPower().get_data_frame()

        self.assertEquals(df["load0"].iloc[0], 2)
        self.assertTrue("load0" in df.columns)

    def test_inpower_percluster_dataframe(self):
        """Test InPower.get_cluster_data_frame()"""
        df = InPower().get_cluster_data_frame(0)

        self.assertEquals(df["raw_cpu_power"].iloc[0], 36)
        self.assertTrue("load0" in df.columns)

    def test_inpower_get_load_data(self):
        """Test InPower.get_load_data()"""
        load_data = InPower().get_load_data({0: "A7", 1: "A15"})

        self.assertEquals(load_data["A15"].iloc[0], 2 + 6 + 0 + 1)
        self.assertEquals(load_data["A7"].iloc[3], 9 + 7 + 20 + 2)
        self.assertEquals(load_data["A15"].iloc[0], load_data["A15"].iloc[1])

    def test_inpower_plot_cluster_load(self):
        """Test that InPower.plot_cluster_load() doesn't explode"""
        InPower().plot_cluster_load(0)

    def test_inpower_plot_load(self):
        """Test that InPower.plot_load() doesn't explode"""
        InPower().plot_load({0: "A7", 1: "A15"}, title="Util")
