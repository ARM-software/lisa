#!/usr/bin/python

import unittest
import os, sys
import tempfile

import utils_tests
import cr2

class TestCR2(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestCR2, self).__init__(
            ["results.csv"],
            *args, **kwargs)

    def test_get_results(self):
        results_frame = cr2.get_results()

        self.assertEquals(type(results_frame), cr2.CR2)
        self.assertEquals(len(results_frame.columns), 3)
        self.assertEquals(results_frame["antutu"][0], 2)
        self.assertEquals(results_frame["antutu"][1], 6)
        self.assertEquals(results_frame["antutu"][2], 3)
        self.assertEquals(results_frame["glbench_trex"][0], 740)
        self.assertEquals(results_frame["geekbench"][0], 3)
        self.assertEquals(results_frame["geekbench"][1], 4)

    def test_get_results_path(self):
        """cr2.get_results() can be given a directory for the results.csv"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        results_frame = cr2.get_results(self.out_dir)

        self.assertEquals(len(results_frame.columns), 3)

    def test_combine_results(self):
        res1 = cr2.get_results()
        res2 = cr2.get_results()

        res2["antutu"][0] = 42
        combined = cr2.combine_results([res1, res2], keys=["power_allocator", "ipa"])

        self.assertEquals(type(combined), cr2.CR2)
        self.assertEquals(combined["antutu"]["power_allocator"][0], 2)
        self.assertEquals(combined["antutu"]["ipa"][0], 42)
        self.assertEquals(combined["geekbench"]["power_allocator"][1], 4)
        self.assertEquals(combined["glbench_trex"]["ipa"][2], 920)

    def test_plot_results(self):
        """Test CR2.plot_results()

        Can't test it, so just check that it doens't bomb
        """
        results_frame = cr2.get_results()

        results_frame.plot_results("antutu")
        results_frame.plot_results("glbench_trex", title="Glbench TRex")

    def test_get_run_number(self):
        self.assertEquals(cr2.get_run_number("score_2"), (True, 2))
        self.assertEquals(cr2.get_run_number("score"), (True, 0))
        self.assertEquals(cr2.get_run_number("score 3"), (True, 3))
        self.assertEquals(cr2.get_run_number("FPS_1"), (True, 1))
        self.assertEquals(cr2.get_run_number("Memory_score")[0], False)
