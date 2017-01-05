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


import os
import re
import matplotlib, tempfile

import trappy
from test_thermal import BaseTestThermal

class TestTrappy(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestTrappy, self).__init__(*args, **kwargs)
        self.map_label = {"00000000,00000039": "A53", "00000000,00000006": "A57"}
        self.actor_order = ["GPU", "A57", "A53"]

    def test_summary_plots(self):
        """Test summary_plots()

        Can't check that the graphs are ok, so just see that the method doesn't blow up"""

        trappy.summary_plots(self.actor_order, self.map_label)
        matplotlib.pyplot.close('all')

        trappy.summary_plots(self.actor_order, self.map_label, width=14,
                          title="Foo")
        matplotlib.pyplot.close('all')

    def test_summary_plots_bad_parameters(self):
        """When summary_plots() receives bad parameters, it offers an understandable error"""

        self.assertRaises(TypeError, trappy.summary_plots,
                          (self.map_label, self.actor_order))

        try:
            trappy.summary_plots(self.map_label, self.actor_order)
            self.fail()
        except TypeError as exception:
            pass

        self.assertTrue("actor_order" in str(exception))

        try:
            trappy.summary_plots(self.actor_order, self.actor_order)
            self.fail()
        except TypeError as exception:
            pass

        self.assertTrue("map_label" in str(exception))

    def test_summary_other_dir(self):
        """Test summary_plots() with another directory"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        trappy.summary_plots(self.actor_order, self.map_label, path=self.out_dir)
        matplotlib.pyplot.close('all')

        # Sanity check that the test actually ran from another directory
        self.assertEquals(os.getcwd(), other_random_dir)

    def test_summary_plots_only_power_allocator_trace(self):
        """Test that summary_plots() work if there is only power allocator
        trace"""

        # Strip out "thermal_temperature" from the trace
        trace_out = ""
        with open("trace.txt") as fin:
            for line in fin:
                if not re.search("thermal_temperature:", line):
                    trace_out += line

        with open("trace.txt", "w") as fout:
            fout.write(trace_out)

        trappy.summary_plots(self.actor_order, self.map_label)
        matplotlib.pyplot.close('all')

    def test_summary_plots_no_gpu(self):
        """summary_plots() works if there is no GPU trace"""

        # Strip out devfreq traces
        trace_out = ""
        with open("trace.txt") as fin:
            for line in fin:
                if ("thermal_power_devfreq_get_power:" not in line) and \
                   ("thermal_power_devfreq_limit:" not in line):
                    trace_out += line

        with open("trace.txt", "w") as fout:
            fout.write(trace_out)

        trappy.summary_plots(self.actor_order, self.map_label)
        matplotlib.pyplot.close('all')

    def test_summary_plots_one_actor(self):
        """summary_plots() works if there is only one actor"""

        # Strip out devfreq and little traces
        trace_out = ""
        with open("trace.txt") as fin:
            for line in fin:
                if ("thermal_power_devfreq_get_power:" not in line) and \
                   ("thermal_power_devfreq_limit:" not in line) and \
                   ("thermal_power_cpu_get_power: cpus=00000000,00000039" not in line) and \
                   ("thermal_power_cpu_limit: cpus=00000000,00000039" not in line):
                    trace_out += line

        with open("trace.txt", "w") as fout:
            fout.write(trace_out)

        map_label = {"00000000,00000006": "A57"}
        trappy.summary_plots(self.actor_order, map_label)
        matplotlib.pyplot.close('all')

    def test_compare_runs(self):
        """Basic compare_runs() functionality"""

        trappy.compare_runs(self.actor_order, self.map_label,
                        runs=[("new", "."), ("old", self.out_dir)])
        matplotlib.pyplot.close('all')
