# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import os

from test import LisaTest

import trappy
from bart.common.Analyzer import Analyzer

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
TESTS_CONF = os.path.join(TESTS_DIRECTORY, "smoke_test_ramp.config")

class STune(LisaTest):
    """Tests for SchedTune framework"""

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(STune, cls)._init(TESTS_CONF, *args, **kwargs)

    def test_boosted_utilization_signal(self):
        """The boosted utilization signal is appropriately boosted

        The margin should match the formula
        (sched_load_scale - util) * boost"""

        for tc in self.conf["confs"]:
            test_id = tc["tag"]

            wload_idx = self.conf["wloads"].keys()[0]
            run_dir = os.path.join(self.te.res_dir,
                                   "rtapp:{}:{}".format(test_id, wload_idx),
                                   "1")

            ftrace_events = ["sched_boost_task"]
            ftrace = trappy.FTrace(run_dir, scope="custom",
                                   events=ftrace_events)

            first_task_params = self.conf["wloads"][wload_idx]["conf"]["params"]
            first_task_name = first_task_params.keys()[0]
            rta_task_name = "task_{}".format(first_task_name)

            sbt_dfr = ftrace.sched_boost_task.data_frame
            boost_task_rtapp = sbt_dfr[sbt_dfr.comm == rta_task_name]
            ftrace.add_parsed_event("boost_task_rtapp", boost_task_rtapp)

            # Avoid the first period as the task starts with a very
            # high load and it overutilizes the CPU
            rtapp_period = first_task_params[first_task_name]["params"]["period_ms"]
            task_start = boost_task_rtapp.index[0]
            after_first_period = task_start + (rtapp_period / 1000.)

            boost = tc["cgroups"]["conf"]["schedtune"]["/stune"]["boost"]
            analyzer_const = {
                "SCHED_LOAD_SCALE": 1024,
                "BOOST": boost,
            }
            analyzer = Analyzer(ftrace, analyzer_const,
                                window=(after_first_period, None))
            statement = "(((SCHED_LOAD_SCALE - boost_task_rtapp:util) * BOOST) // 100) == boost_task_rtapp:margin"
            error_msg = "task was not boosted to the expected margin: {:.2f}"\
                        .format(boost / 100.)
            self.assertTrue(analyzer.assertStatement(statement), msg=error_msg)

# vim :set tabstop=4 shiftwidth=4 expandtab
