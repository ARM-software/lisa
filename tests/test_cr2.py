#!/usr/bin/python

import unittest
import os,sys

import utils_tests
import cr2

class TestCR2(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestCR2, self).__init__(
            ["results.csv"],
            *args, **kwargs)

    def test_show_results(self):
        cr = cr2.CR2()

        results_frame = cr.get_results()
        self.assertEquals(len(results_frame.columns), 3)
        self.assertEquals(results_frame["antutu"][0], 1)
        self.assertEquals(results_frame["glbench_egypt"][0], 2)
        self.assertEquals(results_frame["glbench_egypt"][1], 3)
        self.assertEquals(results_frame["geekbench"][0], 4)
