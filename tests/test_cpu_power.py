#    Copyright 2015-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import matplotlib
import pandas as pd

from test_thermal import BaseTestThermal
import trappy
import cpu_power

class TestCpuPower(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestCpuPower, self).__init__(*args, **kwargs)
        self.map_label = {"00000000,00000039": "A53", "00000000,00000006": "A57"}

    def test_pivot_with_labels(self):
        """Test pivot_with_labels()"""
        map_label = {"000000f0": "A15", "0000000f": "A7"}
        dfr_in = pd.DataFrame({'cpus': ["000000f0", "0000000f", "000000f0", "0000000f"],
                               'freq': [1, 3, 2, 6]})

        dfr_out = cpu_power.pivot_with_labels(dfr_in, "freq", "cpus", map_label)

        self.assertEquals(dfr_out["A15"].iloc[0], 1)
        self.assertEquals(dfr_out["A15"].iloc[1], 1)
        self.assertEquals(dfr_out["A15"].iloc[2], 2)
        self.assertEquals(dfr_out["A7"].iloc[1], 3)

    def test_num_cpus_in_mask(self):
        """num_cpus_in_mask() works with the masks we usually use"""
        mask = "000000f0"
        self.assertEquals(cpu_power.num_cpus_in_mask(mask), 4)

        mask = sorted(self.map_label)[0]
        self.assertEquals(cpu_power.num_cpus_in_mask(mask), 2)

        mask = sorted(self.map_label)[1]
        self.assertEquals(cpu_power.num_cpus_in_mask(mask), 4)

    def test_cpuoutpower_dataframe(self):
        """Test that CpuOutPower() creates a proper data_frame"""
        outp = trappy.FTrace().cpu_out_power

        self.assertEquals(outp.data_frame["power"].iloc[0], 1344)
        self.assertTrue("cdev_state" in outp.data_frame.columns)

    def test_cpuoutpower_get_all_freqs(self):
        """Test CpuOutPower.get_all_freqs()"""
        dfr = trappy.FTrace().cpu_out_power.get_all_freqs(self.map_label)

        self.assertEquals(dfr["A57"].iloc[0], 1100)
        self.assertEquals(dfr["A53"].iloc[1], 850)

    def test_cpuinpower_get_dataframe(self):
        """Test that CpuInPower() creates a proper data_frame()"""
        inp = trappy.FTrace().cpu_in_power

        self.assertTrue("load0" in inp.data_frame.columns)
        self.assertEquals(inp.data_frame["load0"].iloc[0], 24)

    def test_cpuinpower_big_cpumask(self):
        """CpuInPower()'s data_frame is not confused by 64-bit cpumasks"""
        in_data = """     kworker/2:2-679   [002]   676.256284: thermal_power_cpu_get:  cpus=00000000,0000000f freq=261888 cdev_state=5 power=12
     kworker/2:2-679   [002]   676.276200: thermal_power_cpu_get:  cpus=00000000,00000030 freq=261888 cdev_state=5 power=0
     kworker/2:2-679   [002]   676.416202: thermal_power_cpu_get:  cpus=00000000,0000000f freq=261888 cdev_state=5 power=0
        """
        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        inp = trappy.FTrace(normalize_time=False).cpu_in_power
        self.assertEquals(round(inp.data_frame.index[0], 6), 676.256284)
        self.assertEquals(inp.data_frame["cpus"].iloc[1], "00000000,00000030")

    def test_cpuinpower_data_frame_asymmetric_clusters(self):
        """Test that CpuInPower()'s data_frame can handle asymmetric clusters

        That is 2 cpus in one cluster and 4 in another, like Juno
        """
        in_data = """
     kworker/2:2-679   [002]   676.256261: thermal_power_cpu_get:   cpus=00000000,00000030 freq=1900000 raw_cpu_power=1259 load={74 49} power=451
     kworker/2:2-679   [002]   676.256271: thermal_power_cpu_get:   cpus=00000000,0000000f freq=450000 raw_cpu_power=36 load={1 2 1 3} power=9
"""

        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        inp = trappy.FTrace(normalize_time=False).cpu_in_power

        self.assertEquals(inp.data_frame["load0"].iloc[0], 74)
        self.assertEquals(inp.data_frame["load1"].iloc[0], 49)
        self.assertEquals(inp.data_frame["load2"].iloc[0], 0)
        self.assertEquals(inp.data_frame["load3"].iloc[0], 0)
        self.assertEquals(inp.data_frame["load0"].iloc[1], 1)
        self.assertEquals(inp.data_frame["load1"].iloc[1], 2)
        self.assertEquals(inp.data_frame["load2"].iloc[1], 1)
        self.assertEquals(inp.data_frame["load3"].iloc[1], 3)

    def test_cpuinpower_get_all_freqs(self):
        """Test CpuInPower.get_all_freqs()"""
        dfr = trappy.FTrace().cpu_in_power.get_all_freqs(self.map_label)

        self.assertEquals(dfr["A57"].iloc[0], 1100)
        self.assertEquals(dfr["A53"].iloc[1], 850)
        self.assertEquals(dfr["A57"].iloc[5], 1100)

    def test_cpuinpower_get_load_data(self):
        """Test CpuInPower.get_load_data()"""
        trace = trappy.FTrace()
        first_load = trace.cpu_in_power.data_frame["load0"].iloc[0]
        load_data = trace.cpu_in_power.get_load_data(self.map_label)

        self.assertEquals(load_data["A57"].iloc[0], 24 + 19)
        self.assertEquals(load_data["A53"].iloc[3], 32 + 28 + 46 + 44)
        self.assertEquals(load_data["A57"].iloc[0], load_data["A57"].iloc[1])

        self.assertEquals(trace.cpu_in_power.data_frame["load0"].iloc[0],
                          first_load)

    def test_cpuinpower_get_normalized_load_data(self):
        """Test CpuInPower.get_normalized_load_data()"""
        trace = trappy.FTrace()
        first_load = trace.cpu_in_power.data_frame["load0"].iloc[0]
        load_data = trace.cpu_in_power.get_normalized_load_data(self.map_label)

        # Ideally the trace should have an event in which the cpus are
        # not running at maximum frequency
        self.assertEquals(load_data["A57"].iloc[0],
                          (24. + 19) * 1100000 / (1100000 * 2))
        self.assertEquals(load_data["A53"].iloc[1],
                          (36. + 49 + 48 + 7) * 850000 / (850000 * 4))
        self.assertEquals(trace.cpu_in_power.data_frame["load0"].iloc[0],
                          first_load)
