#!/usr/bin/python

import os
import sys
import unittest
import utils_tests
import cr2
from cr2.base import trace_parser_explode_array

sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))

class TestBaseMethods(unittest.TestCase):
    """Test simple methods that don't need to set up a directory"""
    def test_trace_parser_explode_array(self):
        """TestBaseMethods: Basic test of trace_parser_explode_array()"""

        line = "cpus=0000000f freq=1400000 raw_cpu_power=189 load={3 2 12 2} power=14"
        expected = "cpus=0000000f freq=1400000 raw_cpu_power=189 load0=3 load1=2 load2=12 load3=2 power=14"
        array_lengths = {"load": 4}

        result = trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, expected)

    def test_trace_parser_explode_array_nop(self):
        """TestBaseMethods: trace_parser_explode_array() returns the same string if there's no array in it"""

        line = "cpus=0000000f freq=1400000 raw_cpu_power=189 load0=3 load1=2 load2=12 load3=2 power=14"
        array_lengths = {"load": 0}

        result = trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, line)

    def test_trace_parser_explode_array_2(self):
        """TestBaseMethods: trace_parser_explode_array() works if there's two arrays in the string"""

        line = "cpus=0000000f freq=1400000 load={3 2 12 2} power=14 req_power={10 7 2 34}"
        expected = "cpus=0000000f freq=1400000 load0=3 load1=2 load2=12 load3=2 power=14 req_power0=10 req_power1=7 req_power2=2 req_power3=34"
        array_lengths = {'load': 4, 'req_power': 4}

        result = trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, expected)

    def test_trace_parser_explode_array_diff_lengths(self):
        """TestBaseMethods: trace_parser_explode_array() expands arrays that are shorter than the expected length

        trace_parser_explode_array() has to be able to deal with an array of
        size 2 if we tell it in other parts of the trace it is four.

        """

        line = "cpus=0000000f freq=1400000 load={3 2} power=14"
        expected = "cpus=0000000f freq=1400000 load0=3 load1=2 load2=0 load3=0 power=14"
        array_lengths = {'load': 4}

        result = trace_parser_explode_array(line, array_lengths)
        self.assertEquals(result, expected)

class TestBase(utils_tests.SetupDirectory):
    """Incomplete tests for the Base class"""

    def __init__(self, *args, **kwargs):
        super(TestBase, self).__init__(
             [("trace_thermal.txt", "trace.txt")],
             *args,
             **kwargs)

    def test_parse_empty_array(self):
        """TestBase: Trace with empty array creates a valid DataFrame"""

        in_data = """     kworker/4:1-397   [004]   720.741315: thermal_power_cpu_get: cpus=000000f0 freq=1900000 raw_cpu_power=1259 load={} power=61
     kworker/4:1-397   [004]   720.741349: thermal_power_cpu_get: cpus=0000000f freq=1400000 raw_cpu_power=189 load={} power=14"""

        expected_columns = set(["cpus", "freq", "raw_cpu_power", "power"])

        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        run = cr2.Run()
        dfr = run.in_power.data_frame

        self.assertEquals(set(dfr.columns), expected_columns)
        self.assertEquals(dfr["power"].iloc[0], 61)

    def test_get_dataframe(self):
        """TestBase: Thermal.data_frame["thermal_zone"] exists and
           Thermal.data_frame["temp"][0] = 24000"""
        dfr = cr2.Run().thermal.data_frame

        self.assertTrue("thermal_zone" in dfr.columns)
        self.assertEquals(dfr["temp"].iloc[0], 75885)

    def test_write_csv(self):
        """TestBase: Base::write_csv() creates a valid csv"""
        from csv import DictReader

        fname = "thermal.csv"
        cr2.Run().thermal.write_csv(fname)

        with open(fname) as fin:
            csv_reader = DictReader(fin)

            self.assertTrue("Time" in csv_reader.fieldnames)
            self.assertTrue("temp" in csv_reader.fieldnames)

            first_data = csv_reader.next()
            self.assertEquals(first_data["Time"], "0.0")
            self.assertEquals(first_data["temp"], "75885")

    def test_normalize_time(self):
        """TestBase: Base::normalize_time() normalizes the time of the trace"""
        thrm = cr2.Run().thermal

        last_prev_time = thrm.data_frame.index[-1]

        basetime = thrm.data_frame.index[0]
        thrm.normalize_time(basetime)

        last_time = thrm.data_frame.index[-1]
        expected_last_time = last_prev_time - basetime

        self.assertEquals(round(thrm.data_frame.index[0], 7), 0)
        self.assertEquals(round(last_time - expected_last_time, 7), 0)
