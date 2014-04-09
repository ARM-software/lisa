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
        self.assertEquals(len(results_frame.columns), 3)
        self.assertEquals(results_frame["antutu"][0], 1)
        self.assertEquals(results_frame["glbench_egypt"][0], 2)
        self.assertEquals(results_frame["glbench_egypt"][1], 3)
        self.assertEquals(results_frame["geekbench"][0], 4)

    def test_get_results_path(self):
        """cr2.get_results() can be given a directory for the results.csv"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        results_frame = cr2.get_results(self.out_dir)

        self.assertEquals(len(results_frame.columns), 3)
