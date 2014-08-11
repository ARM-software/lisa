#!/usr/bin/python

import unittest
import os, sys
import matplotlib, re, shutil, tempfile
import pandas as pd

import utils_tests
from cr2 import Thermal, ThermalGovernor
sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))
import thermal

class TestThermalMethods(unittest.TestCase):
    """Test simple methods that don't need to set up a directory"""
    def test_trace_parser_explode_array(self):
        """Basic test of trace_parser_explode_array()"""

        line = "cpus=0000000f freq=1400000 raw_cpu_power=189 load={3 2 12 2} power=14"
        expected = "cpus=0000000f freq=1400000 raw_cpu_power=189 load0=3 load1=2 load2=12 load3=2 power=14"
        array_lengths = {"load": 4}

        result = thermal.trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, expected)

    def test_trace_parser_explode_array_nop(self):
        """trace_parser_explode_array() returns the same string if there's no array in it"""

        line = "cpus=0000000f freq=1400000 raw_cpu_power=189 load0=3 load1=2 load2=12 load3=2 power=14"
        array_lengths = {"load": 0}

        result = thermal.trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, line)

    def test_trace_parser_explode_array_2(self):
        """trace_parser_explode_array() works if there's two arrays in the string"""

        line = "cpus=0000000f freq=1400000 load={3 2 12 2} power=14 req_power={10 7 2 34}"
        expected = "cpus=0000000f freq=1400000 load0=3 load1=2 load2=12 load3=2 power=14 req_power0=10 req_power1=7 req_power2=2 req_power3=34"
        array_lengths = {'load': 4, 'req_power': 4}

        result = thermal.trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, expected)

    def test_trace_parser_explode_array_diff_lengths(self):
        """trace_parser_explode_array() expands arrays that are shorter than
the expected length

        trace_parser_explode_array() has to be able to deal with an
        array of size 2 if we tell it in other parts of the trace it
        is four.

        """

        line = "cpus=0000000f freq=1400000 load={3 2} power=14"
        expected = "cpus=0000000f freq=1400000 load0=3 load1=2 load2=0 load3=0 power=14"
        array_lengths = {'load': 4}

        result = thermal.trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, expected)

class BaseTestThermal(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(BaseTestThermal, self).__init__(
             ["trace.dat"],
             *args,
             **kwargs)

class TestThermalBase(utils_tests.SetupDirectory):
    """Incomplete tests for the ThermalBase class"""

    def __init__(self, *args, **kwargs):
        super(TestThermalBase, self).__init__(
             [],
             *args,
             **kwargs)

    def test_get_trace_array_lengths(self):
        """Test InPower.get_trace_array_lengths()"""

        in_data = """     kworker/4:1-397   [004]   720.741315: thermal_power_actor_cpu_get_dyn_power: cpus=000000f0 freq=1900000 raw_cpu_power=1259 load={1 2} power=61
     kworker/4:1-397   [004]   720.741349: thermal_power_actor_cpu_get_dyn_power: cpus=0000000f freq=1400000 raw_cpu_power=189 load={1 3 4 89} power=14
     kworker/4:1-397   [004]   720.841315: thermal_power_actor_cpu_get_dyn_power: cpus=000000f0 freq=1900000 raw_cpu_power=1259 load={1 2} power=61
     kworker/4:1-397   [004]   720.841349: thermal_power_actor_cpu_get_dyn_power: cpus=0000000f freq=1400000 raw_cpu_power=189 load={} power=14
"""
        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        base = thermal.BaseThermal(".", "thermal_power_actor_cpu_get_dyn_power")
        lengths = base.get_trace_array_lengths("trace.txt")

        self.assertEquals(len(lengths), 1)
        self.assertEquals(lengths["load"], 4)

    def test_parse_into_csv_empty_array(self):
        """Test that trace that has an empty array creates valid csv"""

        in_data = """     kworker/4:1-397   [004]   720.741315: thermal_power_actor_cpu_get_dyn_power: cpus=000000f0 freq=1900000 raw_cpu_power=1259 load={} power=61
     kworker/4:1-397   [004]   720.741349: thermal_power_actor_cpu_get_dyn_power: cpus=0000000f freq=1400000 raw_cpu_power=189 load={} power=14"""
        expected_header = "Time,cpus,freq,raw_cpu_power,power"
        expected_first_data = "720.741315,000000f0,1900000,1259,61"

        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        base = thermal.BaseThermal(None, "thermal_power_actor_cpu_get_dyn_power")

        data_csv_lines = base.data_csv.split('\n')

        self.assertEquals(data_csv_lines[0], expected_header)
        self.assertEquals(data_csv_lines[1], expected_first_data)

class TestThermal(BaseTestThermal):
    def test_get_dataframe(self):
        dfr = Thermal().data_frame

        self.assertTrue("thermal_zone" in dfr.columns)
        self.assertEquals(dfr["temp"].iloc[0], 24000)

    def test_plot_temperature(self):
        """Test ThermalGovernor.plot_temperature()

        Can't check that the graph is ok, so just see that the method
        doesn't blow up

        """

        th_data = Thermal()
        dfr = th_data.data_frame
        ct_series = pd.Series([57, 57], index=(dfr.index[0], dfr.index[-1]))

        th_data.plot_temperature()
        matplotlib.pyplot.close('all')

        th_data.plot_temperature(title="Antutu", control_temperature=ct_series)
        matplotlib.pyplot.close('all')

        th_data.plot_temperature(title="Antutu", ylim=[0, 60])
        matplotlib.pyplot.close('all')

        _, ax = matplotlib.pyplot.subplots()
        th_data.plot_temperature(ax=ax)
        matplotlib.pyplot.close('all')

    def test_normalize_time(self):
        """BaseThermal.normalize_time() normalizes the time of the trace"""
        thrm = thermal.Thermal()

        last_prev_time = thrm.data_frame.index[-1]

        basetime = thrm.data_frame.index[0]
        thrm.normalize_time(basetime)

        last_time = thrm.data_frame.index[-1]
        expected_last_time = last_prev_time - basetime

        self.assertEquals(round(thrm.data_frame.index[0], 7), 0)
        self.assertEquals(round(last_time - expected_last_time, 7), 0)

class TestThermalGovernor(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestThermalGovernor, self).__init__(*args, **kwargs)
        self.actor_order = ["GPU", "A15", "A7"]

    def test_do_txt_if_not_there(self):
        c = ThermalGovernor()

        found = False
        with open("trace.txt") as f:
            for line in f:
                if re.search("thermal", line):
                    found = True
                    break

        self.assertTrue(found)

    def test_fail_if_no_trace_dat(self):
        """Raise an IOError if there's no trace.dat and trace.txt"""
        os.remove("trace.dat")
        self.assertRaises(IOError, ThermalGovernor)

    def test_get_thermal_csv(self):
        ThermalGovernor().write_thermal_csv()
        first_data_line = '523.424828,0,124,10,134,2718,5036,755,8509,8511,8511,47000,10000\n'

        with open("thermal.csv") as f:
            first_line = f.readline()
            self.assertTrue(first_line.startswith("Time,req_power0"))

            second_line = f.readline()
            self.assertEquals(second_line, first_data_line)

    def test_get_dataframe(self):
        dfr = ThermalGovernor().data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertEquals(dfr["current_temperature"].iloc[0], 47000)
        self.assertTrue("total_granted_power" in dfr.columns)
        self.assertFalse("time" in dfr.columns)

    def test_plot_input_power(self):
        """Test plot_input_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        ThermalGovernor().plot_input_power(self.actor_order)
        ThermalGovernor().plot_input_power(self.actor_order, title="Antutu")
        matplotlib.pyplot.close('all')

    def test_plot_output_power(self):
        """Test plot_output_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        ThermalGovernor().plot_output_power(self.actor_order)
        ThermalGovernor().plot_output_power(self.actor_order, title="Antutu")
        matplotlib.pyplot.close('all')

    def test_plot_inout_power(self):
        """Test plot_inout_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        ThermalGovernor().plot_inout_power()
        ThermalGovernor().plot_inout_power(title="Antutu")
        matplotlib.pyplot.close('all')

    def test_other_directory(self):
        """ThermalGovernor can grab the trace.dat from other directories"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        dfr = ThermalGovernor(self.out_dir).data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertEquals(os.getcwd(), other_random_dir)

class TestEmptyThermalGovernor(unittest.TestCase):
    def setUp(self):
        self.previous_dir = os.getcwd()
        self.out_dir = tempfile.mkdtemp()
        os.chdir(self.out_dir)
        with open("trace.txt", "w") as fout:
            fout.write("""version = 6
cpus=8
CPU:7 [204600 EVENTS DROPPED]
           <...>-3979  [007]   217.975284: sched_stat_runtime:   comm=Thread-103 pid=3979 runtime=5014167 [ns] vruntime=244334517704 [ns]
           <...>-3979  [007]   217.975298: sched_task_load_contrib: comm=Thread-103 pid=3979 load_contrib=2500
           <...>-3979  [007]   217.975314: sched_task_runnable_ratio: comm=Thread-103 pid=3979 ratio=1023
           <...>-3979  [007]   217.975332: sched_rq_runnable_ratio: cpu=7 ratio=1023
           <...>-3979  [007]   217.975345: sched_rq_runnable_load: cpu=7 load=127
           <...>-3979  [007]   217.975366: softirq_raise:        vec=7 [action=SCHED]
           <...>-3979  [007]   217.975446: irq_handler_exit:     irq=163 ret=handled
           <...>-3979  [007]   217.975502: softirq_entry:        vec=1 [action=TIMER]
           <...>-3979  [007]   217.975523: softirq_exit:         vec=1 [action=TIMER]
           <...>-3979  [007]   217.975535: softirq_entry:        vec=7 [action=SCHED]
           <...>-3979  [007]   217.975559: sched_rq_runnable_ratio: cpu=7 ratio=1023
           <...>-3979  [007]   217.975571: sched_rq_runnable_load: cpu=7 load=127
           <...>-3979  [007]   217.975584: softirq_exit:         vec=7 [action=SCHED]
           <...>-3979  [007]   217.980139: irq_handler_entry:    irq=163 name=mct_tick7
           <...>-3979  [007]   217.980216: softirq_raise:        vec=1 [action=TIMER]
           <...>-3979  [007]   217.980253: sched_stat_runtime:   comm=Thread-103 pid=3979 runtime=4990542 [ns] vruntime=244336561007 [ns]
           <...>-3979  [007]   217.980268: sched_task_load_contrib: comm=Thread-103 pid=3979 load_contrib=2500""")

    def tearDown(self):
        os.chdir(self.previous_dir)
        shutil.rmtree(self.out_dir)

    def test_empty_trace_txt(self):
        dfr = ThermalGovernor().data_frame
        self.assertEquals(len(dfr), 0)
