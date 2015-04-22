#!/usr/bin/python
# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        test_results.py
# ----------------------------------------------------------------
# $
#

import os, sys
import shutil
import tempfile
import matplotlib
import pandas as pd

import utils_tests
sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))
import results

class TestResults(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestResults, self).__init__(
            [("results.csv", "results.csv")],
            *args, **kwargs)

    def test_get_results(self):
        results_frame = results.get_results()

        self.assertEquals(type(results_frame), results.CR2)
        self.assertEquals(type(results_frame.columns), pd.core.index.MultiIndex)
        self.assertEquals(results_frame["antutu"]["power_allocator"][0], 5)
        self.assertEquals(results_frame["antutu"]["step_wise"][1], 9)
        self.assertEquals(results_frame["antutu"]["step_wise"][2], 7)
        self.assertEquals(results_frame["t-rex_offscreen"]["power_allocator"][0], 1777)
        self.assertEquals(results_frame["geekbench"]["step_wise"][0], 8)
        self.assertEquals(results_frame["geekbench"]["power_allocator"][1], 1)
        self.assertAlmostEquals(results_frame["thechase"]["step_wise"][0], 242.0522258138)

    def test_get_results_path(self):
        """results.get_results() can be given a directory for the results.csv"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        results_frame = results.get_results(self.out_dir)

        self.assertEquals(len(results_frame.columns), 10)

    def test_get_results_id(self):
        """get_results() optional id argument overrides the one in the results file"""
        res = results.get_results(id="malkovich")
        self.assertIsNotNone(res["antutu"]["malkovich"])

    def test_combine_results(self):
        res1 = results.get_results()
        res2 = results.get_results()

        # First split them
        res1.drop('step_wise', axis=1, level=1, inplace=True)
        res2.drop('power_allocator', axis=1, level=1, inplace=True)

        # Now combine them again
        combined = results.combine_results([res1, res2])

        self.assertEquals(type(combined), results.CR2)
        self.assertEquals(combined["antutu"]["step_wise"][0], 4)
        self.assertEquals(combined["antutu"]["power_allocator"][0], 5)
        self.assertEquals(combined["geekbench"]["power_allocator"][1], 1)
        self.assertEquals(combined["t-rex_offscreen"]["step_wise"][2], 424)

    def test_plot_results_benchmark(self):
        """Test CR2.plot_results_benchmark()

        Can't test it, so just check that it doens't bomb
        """

        res = results.get_results()

        res.plot_results_benchmark("antutu")
        res.plot_results_benchmark("t-rex_offscreen", title="Glbench TRex")

        (_, _, y_min, y_max) = matplotlib.pyplot.axis()

        trex_data = pd.concat(res["t-rex_offscreen"][s] for s in res["t-rex_offscreen"])
        data_min = min(trex_data)
        data_max = max(trex_data)

        # Fail if the axes are within the limits of the data.
        self.assertTrue(data_min > y_min)
        self.assertTrue(data_max < y_max)
        matplotlib.pyplot.close('all')

    def test_get_run_number(self):
        self.assertEquals(results.get_run_number("score_2"), (True, 2))
        self.assertEquals(results.get_run_number("score"), (True, 0))
        self.assertEquals(results.get_run_number("score 3"), (True, 3))
        self.assertEquals(results.get_run_number("FPS_1"), (True, 1))
        self.assertEquals(results.get_run_number("Overall_Score"), (True, 0))
        self.assertEquals(results.get_run_number("Overall_Score_2"), (True, 1))
        self.assertEquals(results.get_run_number("Memory_score")[0], False)

    def test_plot_results(self):
        """Test CR2.plot_results()

        Can't test it, so just check that it doens't bomb
        """

        res = results.get_results()

        res.plot_results()
        matplotlib.pyplot.close('all')

    def test_init_fig(self):
        r1 = results.get_results()
        r1.init_fig()
