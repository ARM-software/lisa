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
    """
    Goal
    ====

    Verify that a task in a SchedTune cgroup is boosted

    Detailed Description
    ====================

    The test runs a ramp task that has increasing load as time passes.
    The load increases from 5% to 60% over 1 second.  It is run in
    seven different configurations: -15% boost, -30% boost,
    -60% boost, no boost, 15% boost, 30% boost and 60% boost.

    Expected Behaviour
    ==================

    The margin of the task should match the formula

    .. math::

          (sched\_load\_scale - util) \\times boost

    for configurations greater than zero

          -((-util) \\times boost)

    for configurations lesser than zero.

    """

    test_conf = TESTS_CONF
    experiments_conf = TESTS_CONF

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(STune, cls).runExperiments(*args, **kwargs)

    def test_boosted_utilization_signal(self):
        """Tasks in stune groups are boosted"""

        for tc in self.confs:
            conf_id = tc["tag"]

            wload_id = self.wloads.keys()[0]
            run_dir = os.path.join(self.te.res_dir,
                                   "rtapp:{}:{}".format(conf_id, wload_id),
                                   "1")

            ftrace_events = ["sched_boost_task"]
            ftrace = trappy.FTrace(run_dir, scope="custom",
                                   events=ftrace_events)

            first_task_params = self.wloads[wload_id]["conf"]["params"]
            first_task_name = first_task_params.keys()[0]
            rta_task_name = "task_{}".format(first_task_name)

            # Avoid the first period as the task starts with a very
            # high load and it overutilizes the CPU
            rtapp_period = first_task_params[first_task_name]["params"]["period_ms"]
            sbt_dfr = ftrace.sched_boost_task.data_frame
            task_start = sbt_dfr[sbt_dfr.comm == rta_task_name].index[0]
            after_first_period = task_start + (rtapp_period / 1000.)

            boost = tc["cgroups"]["conf"]["schedtune"]["/stune"]["boost"]
            analyzer_const = {
                "SCHED_LOAD_SCALE": 1024,
                "BOOST": boost,
            }
            analyzer = Analyzer(ftrace, analyzer_const,
                                window=(after_first_period, None),
                                filters={"comm": rta_task_name})
            if boost == 0:
                statement = "sched_boost_task:margin == 0"
            elif boost > 0:
                statement = "(((SCHED_LOAD_SCALE - sched_boost_task:util) * BOOST) // 100) == sched_boost_task:margin"
            else:
                statement = "-((-sched_boost_task:util * BOOST) // 100) == sched_boost_task:margin"

            error_msg = "task was not boosted to the expected margin: {:.2f}"\
                        .format(boost / 100.)
            self.assertTrue(analyzer.assertStatement(statement), msg=error_msg)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
