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

import wrapt

from conf import JsonConf
from executor import Executor

class LisaTest(unittest.TestCase):
    """A base class for LISA defined tests"""

    @classmethod
    def _init(cls, conf_file, *args, **kwargs):
        """
        Base class to run LISA test experiments
        """

        cls.logger = logging.getLogger('test')
        cls.logger.setLevel(logging.INFO)
        if 'loglevel' in kwargs:
            cls.logger.setLevel(kwargs['loglevel'])
            kwargs.pop('loglevel')

        cls.conf_file = conf_file
        cls.logger.info("%14s - Using configuration:",
                         "LisaTest")
        cls.logger.info("%14s -    %s",
                         "LisaTest", cls.conf_file)

        cls.logger.debug("%14s - Load test specific configuration...", "LisaTest")
        json_conf = JsonConf(cls.conf_file)
        cls.conf = json_conf.load()

        cls.logger.debug("%14s - Checking tests configuration...", "LisaTest")
        cls._checkConf()

        cls._runExperiments()

    @classmethod
    def _runExperiments(cls):
        """
        Default experiments execution engine
        """

        cls.logger.info("%14s - Setup tests execution engine...", "LisaTest")
        cls.executor = Executor(tests_conf = cls.conf_file);

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
    def _checkConf(cls):
        """
        Check for mandatory configuration options
        """
        assert 'confs' in cls.conf, \
            "Configuration file missing target configurations ('confs' attribute)"
        assert cls.conf['confs'], \
            "Configuration file with empty set of target configurations ('confs' attribute)"
        assert 'wloads' in cls.conf, \
            "Configuration file missing workload configurations ('wloads' attribute)"
        assert cls.conf['wloads'], \
            "Configuration file with empty set of workloads ('wloads' attribute)"

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
