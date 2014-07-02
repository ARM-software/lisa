#!/usr/bin/python

import pandas as pd

from test_thermal import TestThermalBase
from cr2 import OutPower, InPower
import power

class TestPower(TestThermalBase):
    def __init__(self, *args, **kwargs):
        super(TestPower, self).__init__(*args, **kwargs)
        self.map_label = {"000000f0": "A15", "0000000f": "A7"}

    def test_pivot_with_labels(self):
        """Test pivot_with_labels()"""
        map_label = {"000000f0": "A15", "0000000f": "A7"}
        dfr_in = pd.DataFrame({'cpus': ["000000f0", "0000000f", "000000f0", "0000000f"],
                               'freq': [1, 3, 2, 6]})

        dfr_out = power.pivot_with_labels(dfr_in, "freq", "cpus", map_label)

        self.assertEquals(dfr_out["A15"].iloc[0], 1)
        self.assertEquals(dfr_out["A15"].iloc[1], 1)
        self.assertEquals(dfr_out["A15"].iloc[2], 2)
        self.assertEquals(dfr_out["A7"].iloc[1], 3)

    def test_get_all_freqs_data(self):
        """Test get_all_freqs_data()"""

        inp = InPower()
        outp = OutPower()

        allfreqs = power.get_all_freqs_data(inp, outp, self.map_label)

        self.assertEquals(allfreqs["A7"]["A7_freq_out"].iloc[1], 1400)
        self.assertEquals(allfreqs["A7"]["A7_freq_in"].iloc[35], 1400)
        self.assertEquals(allfreqs["A15"]["A15_freq_out"].iloc[0], 1900)

        # Make sure there are no NaNs in the middle of the array
        self.assertTrue(allfreqs["A15"]["A15_freq_out"].notnull().all())

    def test_outpower_get_dataframe(self):
        """Test OutPower.get_data_frame()"""
        df = OutPower().get_data_frame()

        self.assertEquals(df["power"].iloc[0], 5036)
        self.assertTrue("cdev_state" in df.columns)

    def test_outpower_get_all_freqs(self):
        """Test OutPower.get_all_freqs()"""
        dfr = OutPower().get_all_freqs(self.map_label)

        self.assertEquals(dfr["A15"].iloc[0], 1900)
        self.assertEquals(dfr["A7"].iloc[1], 1400)

    def test_inpower_get_dataframe(self):
        """Test InPower.get_data_frame()"""
        df = InPower().get_data_frame()

        self.assertTrue("load0" in df.columns)
        self.assertEquals(df["load0"].iloc[0], 1)

    def test_inpower_get_all_freqs(self):
        """Test InPower.get_all_freqs()"""
        dfr = InPower().get_all_freqs(self.map_label)

        self.assertEquals(dfr["A15"].iloc[0], 1900)
        self.assertEquals(dfr["A7"].iloc[1], 1400)
        self.assertEquals(dfr["A15"].iloc[13], 1800)

    def test_inpower_get_load_data(self):
        """Test InPower.get_load_data()"""
        load_data = InPower().get_load_data(self.map_label)

        self.assertEquals(load_data["A15"].iloc[0], 1 + 2 + 1 + 1)
        self.assertEquals(load_data["A7"].iloc[3], 100 + 100 + 100 + 100)
        self.assertEquals(load_data["A15"].iloc[0], load_data["A15"].iloc[1])

    def test_inpower_plot_load(self):
        """Test that InPower.plot_load() doesn't explode"""
        InPower().plot_load(self.map_label, title="Util")
