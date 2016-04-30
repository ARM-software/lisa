#    Copyright 2015-2016 ARM Limited
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
import numpy as np
import pandas as pd
import tempfile
import os
import warnings

from test_thermal import BaseTestThermal
import trappy


class TestPlotter(BaseTestThermal):

    """No Bombing testcases for plotter"""

    def __init__(self, *args, **kwargs):
        super(TestPlotter, self).__init__(*args, **kwargs)

    def test_plot_no_pivot(self):
        """Tests LinePlot with no pivot"""
        trace1 = trappy.FTrace(name="first")
        l = trappy.LinePlot(trace1, trappy.thermal.Thermal, column="temp")
        l.view(test=True)

    def test_plot_multi_trace(self):
        """Tests LinePlot with no Pivot multi traces"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot(
            [trace1, trace2], trappy.thermal.Thermal, column="temp")
        l.view(test=True)

    def test_plot_multi(self):
        """Tests LinePlot with no Pivot multi attrs"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot([trace1,
                          trace2],
                         [trappy.thermal.Thermal,
                          trappy.thermal.ThermalGovernor],
                         column=["temp",
                                 "power_range"])
        l.view(test=True)

    def test_plot_filter(self):
        """Tests LinePlot with no Pivot with filters"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot([trace1,
                          trace2],
                         [trappy.cpu_power.CpuOutPower],
                         column=["power"],
                         filters={"cdev_state": [0]})
        l.view(test=True)

    def test_plot_pivot(self):
        """Tests LinePlot with Pivot"""
        trace1 = trappy.FTrace(name="first")
        l = trappy.LinePlot(
            trace1,
            trappy.thermal.Thermal,
            column="temp",
            pivot="thermal_zone")
        l.view(test=True)

    def test_plot_multi_trace_pivot(self):
        """Tests LinePlot with Pivot multi traces"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot(
            [trace1, trace2], trappy.cpu_power.CpuOutPower, column="power", pivot="cpus")
        l.view(test=True)

    def test_plot_multi_pivot(self):
        """Tests LinePlot with Pivot with multi attrs"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot([trace1,
                          trace2],
                         [trappy.cpu_power.CpuInPower,
                          trappy.cpu_power.CpuOutPower],
                         column=["dynamic_power",
                                 "power"],
                         pivot="cpus")
        l.view(test=True)

    def test_plot_multi_pivot_filter(self):
        """Tests LinePlot with Pivot and filters"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot(
            trace1,
            trappy.cpu_power.CpuInPower,
            column=[
                "dynamic_power",
                "load1"],
            filters={
                "cdev_state": [
                    1,
                    0]},
            pivot="cpus")
        l.view(test=True)

    def test_plot_savefig(self):
        """Tests plotter: savefig"""
        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")
        l = trappy.LinePlot(
            trace1,
            trappy.cpu_power.CpuInPower,
            column=[
                "dynamic_power",
                "load1"],
            filters={
                "cdev_state": [
                    1,
                    0]},
            pivot="cpus")
        png_file = tempfile.mktemp(dir="/tmp", suffix=".png")
        l.savefig(png_file)
        self.assertTrue(os.path.isfile(png_file))
        os.remove(png_file)


    def test_signals(self):
        """Test signals input for LinePlot"""

        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")

        l = trappy.LinePlot([trace1,
                          trace2],
                         signals=["cpu_in_power:dynamic_power",
                                 "cpu_out_power:power"],
                         pivot="cpus")

        l.view(test=True)


    def test_signals_exceptions(self):
        """Test incorrect input combinations: signals"""

        trace1 = trappy.FTrace(name="first")
        trace2 = trappy.FTrace(name="second")

        with self.assertRaises(ValueError):
            l = trappy.LinePlot([trace1, trace2],
                            column=[
                                "dynamic_power",
                                "load1"],
                            signals=["cpu_in_power:dynamic_power",
                                 "cpu_out_power:power"],
                            pivot="cpus")

        with self.assertRaises(ValueError):
            l = trappy.LinePlot([trace1, trace2],
                            trappy.cpu_power.CpuInPower,
                            signals=["cpu_in_power:dynamic_power",
                                 "cpu_out_power:power"],
                            pivot="cpus")

        with self.assertRaises(ValueError):
            l = trappy.LinePlot([trace1, trace2],
                            trappy.cpu_power.CpuInPower,
                            column=[
                                "dynamic_power",
                                "load1"],
                            signals=["cpu_in_power:dynamic_power",
                                 "cpu_out_power:power"],
                            pivot="cpus")

    def test_lineplot_dataframe(self):
        """LinePlot plots DataFrames without exploding"""
        data = np.random.randn(4, 2)
        dfr = pd.DataFrame(data, columns=["tick", "tock"]).cumsum()
        trappy.LinePlot(dfr, column=["tick"]).view(test=True)

    def test_get_trace_event_data_corrupted_trace(self):
        """get_trace_event_data() works with a corrupted trace"""
        from trappy.plotter.Utils import get_trace_event_data

        trace = trappy.FTrace()

        # We create this trace:
        #
        # 1 15414 -> 15411
        # 2 15411 -> 15414
        # 3 15414 -> 15411 (corrupted, should be dropped)
        # 4 15413 -> 15411
        # 5 15411 -> 15413
        #
        # Which should plot like:
        #
        # CPU
        #    +-------+-------+
        #  0 | 15411 | 15414 |
        #    +-------+-------+       +-------+
        #  1                         | 15411 |
        #                            +-------+
        #    +-------+-------+-------+-------+
        #   0.1     0.2     0.3     0.4     0.5

        broken_trace = pd.DataFrame({
            '__comm': ["task2", "task1", "task2", "task3", "task1"],
            '__cpu':  [0, 0, 0, 1, 1],
            '__pid':  [15414, 15411, 15414, 15413, 15411],
            'next_comm': ["task1", "task2", "task1", "task1", "task3"],
            'next_pid':  [15411, 15414, 15411, 15411, 15413],
            'prev_comm': ["task2", "task1", "task2", "task3", "task1"],
            'prev_pid':  [15414, 15411, 15414, 15413, 15411],
            'prev_state': ["S", "R", "S", "S", "S"]},
            index=pd.Series(range(1, 6), name="Time"))

        trace.sched_switch.data_frame = broken_trace

        with warnings.catch_warnings(record=True) as warn:
            data, procs, window = get_trace_event_data(trace)
            self.assertEquals(len(warn), 1)

            warn_str = str(warn[-1])
            self.assertTrue("15411" in warn_str)
            self.assertTrue("4" in warn_str)

        zipped_comms = zip(broken_trace["next_comm"], broken_trace["next_pid"])
        expected_procs = set("-".join([comm, str(pid)]) for comm, pid in zipped_comms)

        self.assertTrue([1, 2, 0] in data["task1-15411"])
        self.assertTrue([2, 3, 0] in data["task2-15414"])
        self.assertTrue([4, 5, 1] in data["task1-15411"])
        self.assertEquals(procs, expected_procs)
        self.assertEquals(window, [1, 5])

class TestILinePlotter(unittest.TestCase):
    def test_simple_dfr(self):
        dfr1 = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        dfr2 = pd.DataFrame([2, 3, 4, 5], columns=["a"])

        trappy.ILinePlot([dfr1, dfr2], column=["a", "a"]).view(test=True)

    def test_duplicate_merging(self):
        dfr1 = pd.DataFrame([1, 2, 3, 4], index=[0., 0., 1., 2.], columns=["a"])
        dfr2 = pd.DataFrame([2, 3, 4, 5], index=[1., 1., 1., 2.], columns=["a"])

        trappy.ILinePlot([dfr1, dfr2], column=["a", "a"]).view(test=True)

    def test_independent_series_merging(self):
        """ILinePlot fixes indexes of independent series"""
        index1 = [0., 1., 2., 3.]
        s1 = pd.Series([1, 2, 3, 4], index=index1)
        index2 = [0.5, 1.5, 2.5, 3.5]
        s2 = pd.Series([2, 3, 4, 5], index=index2)

        dfr = pd.DataFrame([0, 1, 2, 3], columns=["a"])
        iplot = trappy.ILinePlot(dfr, column=["a"])
        s = {"s1": s1, "s2": s2}
        merged = iplot._fix_indexes(s)

        expected_index = index1 + index2
        expected_index.sort()
        self.assertEquals(expected_index, merged.index.tolist())

class TestBarPlot(unittest.TestCase):
    def setUp(self):
        self.dfr = pd.DataFrame({"foo": [1, 2, 3],
                                 "bar": [2, 3, 1],
                                 "baz": [3, 2, 1]})

    def test_barplot_dfr(self):
        """BarPlot plots dataframes without exploding"""
        trappy.BarPlot(self.dfr, column=["foo", "bar"]).view(test=True)

    def test_barplot_trace(self):
        """BarPlot plots traces without exploding"""
        trace = trappy.BareTrace()
        trace.add_parsed_event("event", self.dfr)

        trappy.BarPlot(trace, signals=["event:foo", "event:bar"]).view(test=True)
