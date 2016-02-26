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

from conf import JsonConf
from executor import Executor

class LisaTest(unittest.TestCase):
    """A base class for LISA defined tests"""

    @classmethod
    def _init(self, conf_file, *args, **kwargs):
        """
        Base class to run LISA test experiments
        """

        self.logger = logging.getLogger('test')
        self.logger.setLevel(logging.INFO)
        if 'loglevel' in kwargs:
            self.logger.setLevel(kwargs['loglevel'])
            kwargs.pop('loglevel')

        self.conf_file = conf_file
        self.logger.info("%14s - Using configuration:",
                         "LisaTest")
        self.logger.info("%14s -    %s",
                         "LisaTest", self.conf_file)

        self.logger.debug("%14s - Load test specific configuration...", "LisaTest")
        json_conf = JsonConf(self.conf_file)
        self.conf = json_conf.load()

        self.logger.debug("%14s - Checking tests configuration...", "LisaTest")
        self._checkConf()

        self._runExperiments()

    @classmethod
    def _runExperiments(self):
        """
        Default experiments execution engine
        """

        self.logger.info("%14s - Setup tests execution engine...", "LisaTest")
        self.executor = Executor(tests_conf = self.conf_file)

        # Alias executor objects to make less verbose tests code
        self.te = self.executor.te
        self.target = self.executor.target

        # Execute pre-experiments code defined by the test
        self._experimentsInit()

        self.logger.info("%14s - Experiments execution...", "LisaTest")
        self.executor.run()

        # Execute post-experiments code defined by the test
        self._experimentsFinalize()

    @classmethod
    def _checkConf(self):
        """
        Check for mandatory configuration options
        """
        assert 'confs' in self.conf, \
            "Configuration file missing target configurations ('confs' attribute)"
        assert self.conf['confs'], \
            "Configuration file with empty set of target configurations ('confs' attribute)"
        assert 'wloads' in self.conf, \
            "Configuration file missing workload configurations ('wloads' attribute)"
        assert self.conf['wloads'], \
            "Configuration file with empty set of workloads ('wloads' attribute)"

    @classmethod
    def _experimentsInit(self):
        """
        Code executed before running the experiments
        """

    @classmethod
    def _experimentsFinalize(self):
        """
        Code executed after running the experiments
        """

# vim :set tabstop=4 shiftwidth=4 expandtab
