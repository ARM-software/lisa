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

class SFreq(unittest.TestCase):
    """Tests for SchedFreq framework"""

    @classmethod
    def setUpClass(cls):

        # Get the base path for this test
        cls.basepath = os.path.dirname(os.path.realpath(__file__))
        cls.basepath = cls.basepath.replace('/libs/utils', '')
        # Test configuration file
        cls.tests_conf_file = os.path.join(
                cls.basepath, "sfreq.config")

        logging.info("%14s - Using configuration: %s",
                     "SFreq", cls.tests_conf_file)

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

        logging.info("%14s - Target setup...", "SFreq")
        cls.executor = Executor(tests_conf = cls.tests_conf_file)

        logging.info("%14s - Experiments execution...", "SFreq")
        cls.executor.run()

    def test_regression(self):
        """Check that there is not regression on energy"""
        # TODO

# vim :set tabstop=4 shiftwidth=4 expandtab
