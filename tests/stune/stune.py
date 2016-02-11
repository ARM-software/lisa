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

import os
import unittest

from conf import JsonConf
from executor import Executor

import trappy
from bart.common.Analyzer import Analyzer

# Configure logging (we want verbose logging to monitor test progress)
import logging
reload(logging)
logging.basicConfig(
    format='%(asctime)-9s %(levelname)-8s: %(message)s',
    level=logging.INFO, datefmt='%I:%M:%S')

class STune(unittest.TestCase):
    """Tests for SchedTune framework"""

    @classmethod
    def setUpClass(cls):

        # Get the base path for this test
        cls.basepath = os.path.dirname(os.path.realpath(__file__))
        cls.basepath = cls.basepath.replace('/libs/utils', '')
        # Test configuration file
        cls.tests_conf_file = os.path.join(
                cls.basepath, "stune.config")

        logging.info("%14s - Using configuration: %s",
                     "STune", cls.tests_conf_file)

        # Load test specific configuration
        json_conf = JsonConf(cls.tests_conf_file)
        cls.conf = json_conf.load()

        # Check for mandatory configurations
        assert 'confs' in cls.conf, \
            "Configuration file missing target configurations ('confs' attribute)"
        assert cls.conf['confs'], \
            "Configuration file with empty set of target configurations ('confs' attribute)"
        assert 'wloads' in cls.conf, \
            "Configuration file missing workload configurations ('wloads' attribute)"
        assert cls.conf['wloads'], \
            "Configuration file with empty set of workloads ('wloads' attribute)"

        logging.info("%14s - Target setup...", "STune")
        cls.executor = Executor(tests_conf = cls.tests_conf_file)

        # Alias executor objects to simplify following tests
        cls.te = cls.executor.te
        cls.target = cls.executor.target

        logging.info("%14s - Experiments execution...", "STune")
        cls.executor.run()

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

            # Avoid the first period as the task starts with a very
            # high load and it overutilizes the CPU
            rtapp_period = first_task_params[first_task_name]["params"]["period_ms"]
            task_start = boost_task_rtapp.index[0]
            after_first_period = task_start + rtapp_period
            boost_task_rtapp = boost_task_rtapp.ix[after_first_period:]

            sched_load_scale = 1024
            boost = tc["cgroups"]["conf"]["schedtune"]["/stune"]["boost"] / 100.
            util = boost_task_rtapp["util"]
            expected_margin = (sched_load_scale - util) * boost
            expected_margin = expected_margin.astype(int)
            boost_task_rtapp["expected_margin"] = expected_margin
            ftrace.add_parsed_event("boost_task_rtapp", boost_task_rtapp)

            analyzer = Analyzer(ftrace, {})
            statement = "boost_task_rtapp:margin == boost_task_rtapp:expected_margin"
            error_msg = "task was not boosted to the expected margin: {}".\
                        format(boost)
            self.assertTrue(analyzer.assertStatement(statement), msg=error_msg)

# vim :set tabstop=4 shiftwidth=4 expandtab
