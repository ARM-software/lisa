#!/usr/bin/python

import os, sys
import tempfile

import utils_tests
sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))
import results

class TestResults(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestResults, self).__init__(
            ["results.csv"],
            *args, **kwargs)

    def test_get_results(self):
        results_frame = results.get_results()

        self.assertEquals(type(results_frame), results.CR2)
        self.assertEquals(len(results_frame.columns), 3)
        self.assertEquals(results_frame["antutu"][0], 2)
        self.assertEquals(results_frame["antutu"][1], 6)
        self.assertEquals(results_frame["antutu"][2], 3)
        self.assertEquals(results_frame["glbench_trex"][0], 740)
        self.assertEquals(results_frame["geekbench"][0], 3)
        self.assertEquals(results_frame["geekbench"][1], 4)

    def test_get_results_path(self):
        """results.get_results() can be given a directory for the results.csv"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        results_frame = results.get_results(self.out_dir)

        self.assertEquals(len(results_frame.columns), 3)

    def test_combine_results(self):
        res1 = results.get_results()
        res2 = results.get_results()

        res2["antutu"][0] = 42
        combined = results.combine_results([res1, res2], keys=["power_allocator", "ipa"])

        self.assertEquals(type(combined), results.CR2)
        self.assertEquals(combined["antutu"]["power_allocator"][0], 2)
        self.assertEquals(combined["antutu"]["ipa"][0], 42)
        self.assertEquals(combined["geekbench"]["power_allocator"][1], 4)
        self.assertEquals(combined["glbench_trex"]["ipa"][2], 920)

    def test_plot_results_benchmark(self):
        """Test CR2.plot_results_benchmark()

        Can't test it, so just check that it doens't bomb
        """
        results_frame = results.get_results()

        results_frame.plot_results_benchmark("antutu")
        results_frame.plot_results_benchmark("glbench_trex", title="Glbench TRex")

    def test_get_run_number(self):
        self.assertEquals(results.get_run_number("score_2"), (True, 2))
        self.assertEquals(results.get_run_number("score"), (True, 0))
        self.assertEquals(results.get_run_number("score 3"), (True, 3))
        self.assertEquals(results.get_run_number("FPS_1"), (True, 1))
        self.assertEquals(results.get_run_number("Memory_score")[0], False)

    def test_plot_results(self):
        """Test CR2.plot_results()

        Can't test it, so just check that it doens't bomb
        """

        r1 = results.get_results()
        r2 = results.get_results()
        combined = results.combine_results([r1, r2], ["r1", "r2"])

        combined.plot_results()
