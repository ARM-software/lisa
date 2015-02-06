#!/usr/bin/python

import unittest
import matplotlib
import os
import pandas as pd
import shutil
import sys
import tempfile

import utils_tests
import cr2

sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))

class BaseTestThermal(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(BaseTestThermal, self).__init__(
             [("trace_thermal.txt", "trace.txt")],
             *args,
             **kwargs)

class TestThermal(BaseTestThermal):

    def test_plot_temperature(self):
        """Test ThermalGovernor.plot_temperature()

        Can't check that the graph is ok, so just see that the method
        doesn't blow up

        """

        th_data = cr2.Run().thermal
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

    def test_plot_temperature_hist(self):
        """Test that plot_temperature_hist() doesn't bomb"""

        _, ax = matplotlib.pyplot.subplots()
        cr2.Run().thermal.plot_temperature_hist(ax, "Foo")
        matplotlib.pyplot.close('all')

class TestThermalGovernor(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestThermalGovernor, self).__init__(*args, **kwargs)
        self.actor_order = ["GPU", "A15", "A7"]

    def test_get_dataframe(self):
        dfr = cr2.Run().thermal_governor.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertEquals(dfr["current_temperature"].iloc[0], 75724)
        self.assertTrue("total_granted_power" in dfr.columns)
        self.assertFalse("time" in dfr.columns)

    def test_plot_temperature(self):
        """Test ThermalGovernor.plot_temperature()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        gov = cr2.Run().thermal_governor

        gov.plot_temperature()
        gov.plot_temperature(legend_label="power allocator", ylim=(0, 72))
        matplotlib.pyplot.close('all')

    def test_plot_input_power(self):
        """plot_input_power() doesn't bomb"""
        gov = cr2.Run().thermal_governor

        gov.plot_input_power(self.actor_order)
        matplotlib.pyplot.close('all')

        gov.plot_input_power(self.actor_order, title="Antutu")
        matplotlib.pyplot.close('all')

        _, ax = matplotlib.pyplot.subplots()
        gov.plot_input_power(self.actor_order, ax=ax)
        matplotlib.pyplot.close('all')

    def test_plot_output_power(self):
        """Test plot_output_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        gov = cr2.Run().thermal_governor

        gov.plot_output_power(self.actor_order)
        matplotlib.pyplot.close('all')

        gov.plot_output_power(self.actor_order, title="Antutu")
        matplotlib.pyplot.close('all')

        _, ax = matplotlib.pyplot.subplots()
        gov.plot_output_power(self.actor_order, title="Antutu", ax=ax)
        matplotlib.pyplot.close('all')

    def test_plot_inout_power(self):
        """Test plot_inout_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        cr2.Run().thermal_governor.plot_inout_power()
        cr2.Run().thermal_governor.plot_inout_power(title="Antutu")
        matplotlib.pyplot.close('all')

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
        dfr = cr2.Run(normalize_time=False).thermal_governor.data_frame
        self.assertEquals(len(dfr), 0)

    def test_empty_plot_temperature(self):
        """run.thermal.plot_temperature() raises ValueError() on an empty
        thermal trace"""
        run = cr2.Run()
        self.assertRaises(ValueError, run.thermal.plot_temperature)
