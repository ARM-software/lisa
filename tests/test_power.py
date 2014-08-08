#!/usr/bin/python

import pandas as pd

from test_thermal import BaseTestThermal
from cr2 import OutPower, InPower
import power

class TestPower(BaseTestThermal):
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

    def test_outpower_dataframe(self):
        """Test that OutPower() creates a proper data_frame"""
        outp = OutPower()

        self.assertEquals(outp.data_frame["power"].iloc[0], 5036)
        self.assertTrue("cdev_state" in outp.data_frame.columns)

    def test_outpower_get_all_freqs(self):
        """Test OutPower.get_all_freqs()"""
        dfr = OutPower().get_all_freqs(self.map_label)

        self.assertEquals(dfr["A15"].iloc[0], 1900)
        self.assertEquals(dfr["A7"].iloc[1], 1400)

    def test_inpower_get_dataframe(self):
        """Test that InPower() creates a proper data_frame()"""
        inp = InPower()

        self.assertTrue("load0" in inp.data_frame.columns)
        self.assertEquals(inp.data_frame["load0"].iloc[0], 2)

    def test_inpower_big_cpumask(self):
        """InPower()'s data_frame is not confused by 64-bit cpumasks"""
        in_data = """     kworker/2:2-679   [002]   676.256284: thermal_power_actor_cpu_get_dyn:  cpus=00000000,0000000f freq=261888 cdev_state=5 power=12
     kworker/2:2-679   [002]   676.276200: thermal_power_actor_cpu_get_dyn:  cpus=00000000,00000030 freq=261888 cdev_state=5 power=0
     kworker/2:2-679   [002]   676.416202: thermal_power_actor_cpu_get_dyn:  cpus=00000000,0000000f freq=261888 cdev_state=5 power=0
        """
        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        inp = InPower()
        self.assertEquals(round(inp.data_frame.index[0], 6), 676.256284)
        self.assertEquals(inp.data_frame["cpus"].iloc[1], "0000000000000030")

    def test_inpower_data_frame_asymmetric_clusters(self):
        """Test that InPower()'s data_frame can handle asymmetric clusters

        That is 2 cpus in one cluster and 4 in another, like Juno
        """
        in_data = """     kworker/2:2-679   [002]   676.256261: thermal_power_actor_cpu_get_dyn_power:   cpus=00000000,0000000f freq=450000 raw_cpu_power=36 load={1 2 1 3} power=9
     kworker/2:2-679   [002]   676.256271: thermal_power_actor_cpu_get_dyn_power:   cpus=00000000,00000030 freq=1900000 raw_cpu_power=1259 load={74 49} power=451
"""

        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        inp = InPower()

        self.assertEquals(inp.data_frame["load0"].iloc[0], 1)
        self.assertEquals(inp.data_frame["load1"].iloc[0], 2)
        self.assertEquals(inp.data_frame["load2"].iloc[0], 1)
        self.assertEquals(inp.data_frame["load3"].iloc[0], 3)
        self.assertEquals(inp.data_frame["load0"].iloc[1], 74)
        self.assertEquals(inp.data_frame["load1"].iloc[1], 49)
        self.assertEquals(inp.data_frame["load2"].iloc[1], 0)
        self.assertEquals(inp.data_frame["load3"].iloc[1], 0)

    def test_inpower_get_all_freqs(self):
        """Test InPower.get_all_freqs()"""
        dfr = InPower().get_all_freqs(self.map_label)

        self.assertEquals(dfr["A15"].iloc[0], 1900)
        self.assertEquals(dfr["A7"].iloc[1], 1400)
        self.assertEquals(dfr["A15"].iloc[55], 1800)

    def test_inpower_get_load_data(self):
        """Test InPower.get_load_data()"""
        load_data = InPower().get_load_data(self.map_label)

        self.assertEquals(load_data["A15"].iloc[0], 2 + 3 + 2 + 3)
        self.assertEquals(load_data["A7"].iloc[3], 100 + 100 + 100 + 100)
        self.assertEquals(load_data["A15"].iloc[0], load_data["A15"].iloc[1])

    def test_inpower_plot_load(self):
        """Test that InPower.plot_load() doesn't explode"""
        InPower().plot_load(self.map_label, title="Util")
