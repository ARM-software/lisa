# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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
import shutil
import os
from unittest import TestCase

from lisa.env import TestEnv
from lisa.executor import Executor

class SetUpTarget(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._log = logging.getLogger('TestExecutor')

    def setUp(self):
        self.res_dir='test_{}'.format(self.__class__.__name__)
        self.te = TestEnv(
            target_conf={
                'platform': 'host',
                # With no cpufreq (see below), we won't be able to do
                # calibration. Provide dummy.
                'rtapp-calib': {c: 100 for c in range(64)}
            },
            test_conf={
                # Don't load cpufreq, it won't work when platform=host
                'exclude_modules': ['cpufreq'],
                # Empty list of events to avoid getting the default ones
                'ftrace': {
                    'events': []
                }
            },
            force_new=True)

class TestMagicSmoke(SetUpTarget):
    def test_files_created(self):
        """Test that we can run experiments and get output files"""
        conf_name = 'myconf'
        wl_name = 'mywl'

        results_dir = os.path.join(self.te.res_dir,
                                   'rtapp:{}:{}'.format(conf_name, wl_name))

        experiments_conf = {
            'confs': [{
                'tag': conf_name
            }],
            "wloads" : {
                wl_name : {
                    "type" : "rt-app",
                    "conf" : {
                        "class" : "profile",
                        "params" : {
                            "mytask" : {
                                "kind" : "Periodic",
                                "params" : {
                                    "duty_cycle_pct": 10,
                                    "duration_s": 0.2,
                                },
                            },
                        },
                    },
                },
            },
        }

        executor = Executor(self.te, experiments_conf)
        executor.run()

        self.assertTrue(
            os.path.isdir(results_dir),
            'Expected to find a directory at {}'.format(results_dir))

        result_1_dir = os.path.join(results_dir, '1')
        self.assertTrue(
            os.path.isdir(result_1_dir),
            'Expected to find a directory at {}'.format(result_1_dir))
