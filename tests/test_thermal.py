#!/usr/bin/python

import unittest
import os, sys
import matplotlib, re, shutil, tempfile

import utils_tests
from cr2 import Thermal, ThermalGovernor
sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))
import thermal

class TestThermalBase(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestThermalBase, self).__init__(
             ["trace.dat"],
             *args,
             **kwargs)

class TestThermal(TestThermalBase):
    def test_get_dataframe(self):
        df = Thermal().get_data_frame()

        self.assertTrue("thermal_zone" in df.columns)
        self.assertEquals(df["temp"].iloc[0], 44000)

    def test_plot_temperature(self):
        """Test ThermalGovernor.plot_temperature()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        Thermal().plot_temperature()
        matplotlib.pyplot.close('all')

        Thermal().plot_temperature(title="Antutu")
        matplotlib.pyplot.close('all')

        Thermal().plot_temperature(title="Antutu", ylim=[0, 60])
        matplotlib.pyplot.close('all')

class TestThermalGovernor(TestThermalBase):
    def test_do_txt_if_not_there(self):
        c = ThermalGovernor()

        found = False
        with open("trace.txt") as f:
            for line in f:
                print line
                if re.search("bprint", line):
                    found = True
                    break

        self.assertTrue(found)

    def test_fail_if_no_trace_dat(self):
        """Raise an IOError if there's no trace.dat and trace.txt"""
        os.remove("trace.dat")
        self.assertRaises(IOError, ThermalGovernor)

    def test_get_thermal_csv(self):
        ThermalGovernor().write_thermal_csv()
        first_data_line = '666.731837,3,124,10,137,2718,5036,755,8509,8511,8511,48000,9000\n'

        with open("thermal.csv") as f:
            first_line = f.readline()
            self.assertTrue(first_line.startswith("Time,Pgpu_in"))

            second_line = f.readline()
            self.assertEquals(second_line, first_data_line)

    def test_get_dataframe(self):
        df = ThermalGovernor().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertEquals(df["currT"].iloc[0], 48000)
        self.assertTrue("Ptot_out" in df.columns)
        self.assertFalse("time" in df.columns)

    def test_plot_temperature(self):
        """Test ThermalGovernor.plot_temperature()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        ThermalGovernor().plot_temperature()
        ThermalGovernor().plot_temperature(title="Antutu", ylim=(0, 72))
        matplotlib.pyplot.close('all')

    def test_plot_input_power(self):
        """Test plot_input_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        ThermalGovernor().plot_input_power()
        ThermalGovernor().plot_input_power(title="Antutu")
        matplotlib.pyplot.close('all')

    def test_plot_output_power(self):
        """Test plot_output_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        ThermalGovernor().plot_output_power()
        ThermalGovernor().plot_output_power(title="Antutu")
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

        t = ThermalGovernor(self.out_dir)
        df = t.get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertEquals(os.getcwd(), other_random_dir)

    def test_double_thermal(self):
        """Make sure that multiple invocations of ThermalGovernor work fine

        It should memoize the DataFrame() from the call to
        get_data_frame() to the plot_temperature() one and the latter
        shouldn't bomb

        """

        t = ThermalGovernor()
        dfr = t.get_data_frame()
        t.plot_temperature()

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
        df = ThermalGovernor().get_data_frame()
        self.assertEquals(len(df), 0)
