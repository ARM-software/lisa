#!/usr/bin/python

import unittest
import os,sys
import re, shutil, tempfile

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(TESTS_DIRECTORY, "..", "cr2"))
import thermal

class TestThermal(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        os.chdir(self.out_dir)

        src_fname = os.path.join(TESTS_DIRECTORY, "trace.dat")
        shutil.copy(src_fname, self.out_dir)

    def tearDown(self):
        shutil.rmtree(self.out_dir)

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
        os.remove("trace.dat")
        self.assertRaises(IOError, thermal.Thermal)

    def test_get_thermal_csv(self):
        thermal.Thermal().write_thermal_csv()
        first_data_line = "267.525370,0,1244,144,1388,0,0,0,0,50000,7000\n"

        with open("thermal.csv") as f:
            first_line = f.readline()
            self.assertTrue(first_line.startswith("time,Pa7_in"))

            second_line = f.readline()
            self.assertEquals(second_line, first_data_line)

    def test_get_dataframe(self):
        df = thermal.Thermal().get_data_frame()

        self.assertEquals(df["currT"][0], 50000)
        self.assertTrue("Ptot_out" in df.columns)
        self.assertFalse("time" in df.columns)
        self.assertEquals(len(df), 3)

    def test_plot_temp(self):
        """Test plot_temp()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        thermal.Thermal().plot_temperature()

    def test_plot_input_power(self):
        """Test plot_input_power()

        Can't check that the graph is ok, so just see that the method doesn't blow up"""
        thermal.Thermal().plot_input_power()
