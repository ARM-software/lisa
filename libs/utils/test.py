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
import unittest

from bart.sched.SchedAssert import SchedAssert
from bart.sched.SchedMultiAssert import SchedMultiAssert
from devlib.utils.misc import memoized
import wrapt

from env import TestEnv
from executor import Executor

class LisaTest(unittest.TestCase):
    """A base class for LISA defined tests"""

    @classmethod
    def _init(cls, conf, *args, **kwargs):
        """
        Base class to run LISA test experiments
        """

        cls.logger = logging.getLogger('test')
        cls.logger.setLevel(logging.INFO)
        if 'loglevel' in kwargs:
            cls.logger.setLevel(kwargs['loglevel'])
            kwargs.pop('loglevel')

        cls.conf = conf

        cls._runExperiments()

    @classmethod
    def _runExperiments(cls):
        """
        Default experiments execution engine
        """

        cls.logger.info("%14s - Setup tests execution engine...", "LisaTest")
        test_env = TestEnv()

        cls.executor = Executor(test_env, tests_conf=cls.conf);

        # Alias executor objects to make less verbose tests code
        cls.te = cls.executor.te
        cls.target = cls.executor.target

        # Execute pre-experiments code defined by the test
        cls._experimentsInit()

        cls.logger.info("%14s - Experiments execution...", "LisaTest")
        cls.executor.run()

        # Execute post-experiments code defined by the test
        cls._experimentsFinalize()

    @classmethod
    def _experimentsInit(cls):
        """
        Code executed before running the experiments
        """

    @classmethod
    def _experimentsFinalize(cls):
        """
        Code executed after running the experiments
        """

    @memoized
    def get_sched_assert(self, experiment, task):
        """
        Return a SchedAssert over the task provided
        """
        return SchedAssert(experiment.out_dir, self.te.topology, execname=task)

    @memoized
    def get_multi_assert(self, experiment, task_filter=""):
        """
        Return a SchedMultiAssert over the tasks whose names contain task_filter

        By default, this includes _all_ the tasks that were executed for the
        experiment.
        """
        tasks = experiment.wload.tasks.keys()
        return SchedMultiAssert(experiment.out_dir,
                                self.te.topology,
                                [t for t in tasks if task_filter in t])

    def get_start_time(self, experiment):
        """
        Get the time at which the experiment workload began executing
        """
        start_times_dict = self.get_multi_assert(experiment).getStartTime()
        return min([t["starttime"] for t in start_times_dict.itervalues()])

    def get_end_time(self, experiment):
        """
        Get the time at which the experiment workload finished executing
        """
        end_times_dict = self.get_multi_assert(experiment).getEndTime()
        return max([t["endtime"] for t in end_times_dict.itervalues()])

    def get_window(self, experiment):
        return (self.get_start_time(experiment), self.get_end_time(experiment))

    def get_end_times(self, experiment):
        """
        Get the time at which each task in the workload finished

        Returned as a dict; {"task_name": finish_time, ...}
        """

        end_times = {}
        for task in experiment.wload.tasks.keys():
            sched_assert = SchedAssert(experiment.out_dir, self.te.topology,
                                       execname=task)
            end_times[task] = sched_assert.getEndTime()

        return end_times


@wrapt.decorator
def experiment_test(wrapped_test, instance, args, kwargs):
    """
    Convert a LisaTest test method to be automatically called for each experiment

    The method will be passed the experiment object and a list of the names of
    tasks that were run as the experiment's workload.
    """
    for experiment in instance.executor.experiments:
        tasks = experiment.wload.tasks.keys()
        try:
            wrapped_test(experiment, tasks, *args, **kwargs)
        except AssertionError as e:
            trace_relpath = os.path.join(experiment.out_dir, "trace.dat")
            add_msg = "\n\tCheck trace file: " + os.path.abspath(trace_relpath)
            orig_msg = e.args[0] if len(e.args) else ""
            e.args = (orig_msg + add_msg,) + e.args[1:]
            raise

# Prevent nosetests from running experiment_test directly as a test case
experiment_test.__test__ = False

# vim :set tabstop=4 shiftwidth=4 expandtab
