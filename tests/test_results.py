#    Copyright 2015-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os, sys
import shutil
import tempfile
import matplotlib
import pandas as pd

import utils_tests
sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "trappy"))
from trappy.wa import Result, get_results, combine_results

class TestResults(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestResults, self).__init__(
            [("results.csv", "results.csv")],
            *args, **kwargs)

    def test_get_results(self):
        results_frame = get_results()

        self.assertEquals(type(results_frame), Result)
        self.assertEquals(type(results_frame.columns), pd.core.index.MultiIndex)
        self.assertEquals(results_frame["antutu"]["power_allocator"][0], 5)
        self.assertEquals(results_frame["antutu"]["step_wise"][1], 9)
        self.assertEquals(results_frame["antutu"]["step_wise"][2], 7)
        self.assertEquals(results_frame["t-rex_offscreen"]["power_allocator"][0], 1777)
        self.assertEquals(results_frame["geekbench"]["step_wise"][0], 8)
        self.assertEquals(results_frame["geekbench"]["power_allocator"][1], 1)
        self.assertAlmostEquals(results_frame["thechase"]["step_wise"][0], 242.0522258138)

    def test_get_results_path(self):
        """get_results() can be given a directory for the results.csv"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        results_frame = get_results(self.out_dir)

        self.assertEquals(len(results_frame.columns), 10)

    def test_get_results_filename(self):
        """get_results() can be given a specific filename"""

        old_path = os.path.join(self.out_dir, "results.csv")
        new_path = os.path.join(self.out_dir, "new_results.csv")
        os.rename(old_path, new_path)

        results_frame = get_results(new_path)

        self.assertEquals(len(results_frame.columns), 10)

    def test_get_results_name(self):
        """get_results() optional name argument overrides the one in the results file"""
        res = get_results(name="malkovich")
        self.assertIsNotNone(res["antutu"]["malkovich"])

    def test_combine_results(self):
        res1 = get_results()
        res2 = get_results()

        # First split them
        res1.drop('step_wise', axis=1, level=1, inplace=True)
        res2.drop('power_allocator', axis=1, level=1, inplace=True)

        # Now combine them again
        combined = combine_results([res1, res2])

        self.assertEquals(type(combined), Result)
        self.assertEquals(combined["antutu"]["step_wise"][0], 4)
        self.assertEquals(combined["antutu"]["power_allocator"][0], 5)
        self.assertEquals(combined["geekbench"]["power_allocator"][1], 1)
        self.assertEquals(combined["t-rex_offscreen"]["step_wise"][2], 424)

    def test_plot_results_benchmark(self):
        """Test Result.plot_results_benchmark()

        Can't test it, so just check that it doens't bomb
        """

        res = get_results()

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
        from trappy.wa.results import get_run_number

        self.assertEquals(get_run_number("score_2"), (True, 2))
        self.assertEquals(get_run_number("score"), (True, 0))
        self.assertEquals(get_run_number("score 3"), (True, 3))
        self.assertEquals(get_run_number("FPS_1"), (True, 1))
        self.assertEquals(get_run_number("Overall_Score"), (True, 0))
        self.assertEquals(get_run_number("Overall_Score_2"), (True, 1))
        self.assertEquals(get_run_number("Memory_score")[0], False)

    def test_plot_results(self):
        """Test Result.plot_results()

        Can't test it, so just check that it doens't bomb
        """

        res = get_results()

        res.plot_results()
        matplotlib.pyplot.close('all')

    def test_init_fig(self):
        r1 = get_results()
        r1.init_fig()
