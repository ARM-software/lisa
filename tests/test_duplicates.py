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


import unittest
import matplotlib
import pandas as pd
import utils_tests
import trappy
import shutil

from test_thermal import BaseTestThermal


class TestPlotterDupVals(BaseTestThermal):

    """Test Duplicate Entries in plotter"""

    def __init__(self, *args, **kwargs):
        super(TestPlotterDupVals, self).__init__(*args, **kwargs)

    def test_plotter_duplicates(self):
        """Test that plotter handles duplicates fine"""
        with open("trace.txt", "w") as fout:
            fout.write("""version = 6
cpus=6
       rcuos/2-22 [001] 0000.018510: sched_load_avg_sg: cpus=00000001 load=0 utilization=0
       rcuos/2-22 [001] 6550.018611: sched_load_avg_sg: cpus=00000002 load=1 utilization=1
       rcuos/2-22 [001] 6550.018611: sched_load_avg_sg: cpus=00000004 load=2 utilization=2
       rcuos/2-22 [001] 6550.018612: sched_load_avg_sg: cpus=00000001 load=2 utilization=3
       rcuos/2-22 [001] 6550.018624: sched_load_avg_sg: cpus=00000002 load=1 utilization=4
       rcuos/2-22 [001] 6550.018625: sched_load_avg_sg: cpus=00000002 load=2 utilization=5
       rcuos/2-22 [001] 6550.018626: sched_load_avg_sg: cpus=00000002 load=3 utilization=6
       rcuos/2-22 [001] 6550.018627: sched_load_avg_sg: cpus=00000002 load=1 utilization=7
       rcuos/2-22 [001] 6550.018628: sched_load_avg_sg: cpus=00000004 load=2 utilization=8\n""")
            fout.close()
        trace1 = trappy.FTrace(name="first")
        l = trappy.LinePlot(
            trace1,
            trappy.sched.SchedLoadAvgSchedGroup,
            column=['utilization'],
            filters={
                "load": [
                    1,
                    2]},
            pivot="cpus",
            marker='o',
            linestyle='none',
            per_line=3)
        l.view(test=True)

    def test_plotter_triplicates(self):

        """Test that plotter handles triplicates fine"""

        with open("trace.txt", "w") as fout:
            fout.write("""version = 6
cpus=6
       rcuos/2-22 [001] 0000.018510: sched_load_avg_sg: cpus=00000001 load=0 utilization=0
       rcuos/2-22 [001] 6550.018611: sched_load_avg_sg: cpus=00000002 load=1 utilization=1
       rcuos/2-22 [001] 6550.018611: sched_load_avg_sg: cpus=00000004 load=2 utilization=2
       rcuos/2-22 [001] 6550.018611: sched_load_avg_sg: cpus=00000004 load=2 utilization=2
       rcuos/2-22 [001] 6550.018612: sched_load_avg_sg: cpus=00000001 load=2 utilization=3
       rcuos/2-22 [001] 6550.018624: sched_load_avg_sg: cpus=00000002 load=1 utilization=4
       rcuos/2-22 [001] 6550.018625: sched_load_avg_sg: cpus=00000002 load=2 utilization=5
       rcuos/2-22 [001] 6550.018626: sched_load_avg_sg: cpus=00000002 load=3 utilization=6
       rcuos/2-22 [001] 6550.018627: sched_load_avg_sg: cpus=00000002 load=1 utilization=7
       rcuos/2-22 [001] 6550.018628: sched_load_avg_sg: cpus=00000004 load=2 utilization=8\n""")
            fout.close()

        trace1 = trappy.FTrace(name="first")
        l = trappy.LinePlot(
            trace1,
            trappy.sched.SchedLoadAvgSchedGroup,
            column=['utilization'],
            filters={
                "load": [
                    1,
                    2]},
            pivot="cpus",
            marker='o',
            linestyle='none',
            per_line=3)
        l.view(test=True)
