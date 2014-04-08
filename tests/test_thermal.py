#!/usr/bin/python

import unittest
import os,sys
import re, shutil, tempfile

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(TESTS_DIRECTORY, "..", "cr2"))
import thermal

class TestThermalBase(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        os.chdir(self.out_dir)

        src_fname = os.path.join(TESTS_DIRECTORY, "trace.dat")
        shutil.copy(src_fname, self.out_dir)

    def tearDown(self):
        shutil.rmtree(self.out_dir)

class TestThermal(TestThermalBase):
    def test_do_txt_if_not_there(self):
        c = thermal.Thermal()

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
        self.assertRaises(IOError, thermal.Thermal)

    def test_get_thermal_csv(self):
        thermal.Thermal().write_thermal_csv()
        first_data_line = '328.473417,3,156,12,171,2898,2898,5252,6580,68,8934,48000,9000\n'

        with open("thermal.csv") as f:
            first_line = f.readline()
            self.assertTrue(first_line.startswith("time,Pgpu_in"))

            second_line = f.readline()
            self.assertEquals(second_line, first_data_line)

    def test_get_dataframe(self):
        df = thermal.Thermal().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertEquals(df["currT"][0], 48000)
        self.assertTrue("Ptot_out" in df.columns)
        self.assertFalse("time" in df.columns)

    def test_plot_temp(self):
        """Test plot_temp()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        thermal.Thermal().plot_temperature()

    def test_plot_input_power(self):
        """Test plot_input_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        thermal.Thermal().plot_input_power()

    def test_plot_output_power(self):
        """Test plot_output_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        thermal.Thermal().plot_output_power()

    def test_set_plot_size(self):
        """Test that Thermal.set_plot_size() doesn't bomb"""
        t = thermal.Thermal()
        t.set_plot_size(None, None)
        t.set_plot_size(height=9, width=None)
        t.set_plot_size(height=None, width=9)
        t.set_plot_size(3, 9)

    def test_other_directory(self):
        """Thermal can grab the trace.dat from other directories"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        t = thermal.Thermal(self.out_dir)
        df = t.get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertEquals(os.getcwd(), other_random_dir)

class TestEmptyThermal(unittest.TestCase):
    def setUp(self):
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
        shutil.rmtree(self.out_dir)

    def test_empty_trace_txt(self):
        df = thermal.Thermal().get_data_frame()
        self.assertEquals(len(df), 0)
