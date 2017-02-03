# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
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
from collections import namedtuple
import shutil
import os
from unittest import TestCase

from mock import patch, Mock, MagicMock, call

import devlib

from env import TestEnv
from executor import Executor
import wlgen

class SetUpTarget(TestCase):
    def setUp(self):
        self.res_dir='test_{}'.format(self.__class__.__name__)
        self.te = TestEnv(target_conf={
            'platform': 'host',
            # TODO: can't calibrate rt-app on local targets because they're
            # probably intel_pstate and/or no root. Calibration requires setting
            # performance governor.
            'rtapp-calib': {i: 100 for i in range(4)},
            'modules': ['cgroups'],
        },
        test_conf={
            'results_dir': self.res_dir
        }, force_new=True)

mock_freezer = namedtuple('MockController', ['name'])('freezer')
class MockCgroupsModule(devlib.module.Module):
    name = 'cgroups'
    list_subsystems = Mock(return_value=[mock_freezer])
    freeze = Mock(name='Cgroups_freeze')
    @staticmethod
    def probe(target):
        return True

devlib.module.register_module(MockCgroupsModule)

example_wl = {
    "type" : "rt-app",
    "conf" : {
        "class" : "profile",
        "params" : {
            "mytask" : {
                "kind" : "Periodic",
                "params" : {
                    "duty_cycle_pct": 10,
                    "duration_s": 1,
                },
            },
        },
    }
}

class TestMagicSmoke(SetUpTarget):
    def test_files_created(self):
        conf_name = 'myconf'
        wl_name = 'mywl'
        results_dir = os.path.join(
            self.te.LISA_HOME,
            'results',
            self.res_dir,
            'rtapp:{}:{}'.format(conf_name, wl_name))
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)

        experiments_conf = {
            'confs': [{
                'tag': conf_name
            }],
            "wloads" : {
                wl_name : example_wl,
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

class InterceptedRTA(wlgen.RTA):
    pre_callback = None
    def run(self, *args, **kwargs):
        self.pre_callback()
        super(InterceptedRTA, self).run(*args, **kwargs)

class BrokenRTAException(Exception):
    pass

class BrokenRTA(wlgen.RTA):
    pre_callback = None
    def run(self, *args, **kwargs):
        self.pre_callback()
        self._log.warning('\n\nInjecting workload failure\n')
        raise BrokenRTAException('INJECTED WORKLOAD FAILURE')

class TestFreezeUserspace(SetUpTarget):
    def _do_freezer_test(self):
        experiments_conf = {
            'confs': [{
                'tag': 'with_freeze',
                'flags': ['freeze_userspace'],
            }],
            "wloads" : {
                'my_wl' : example_wl,
            },
        }

        freezer_mock = self.te.target.cgroups.freeze
        freezer_mock.reset_mock()

        def assert_frozen(rta):
            freezer_mock.assert_called_once_with(
                ['init', 'systemd', 'sh', 'ssh'])
            freezer_mock.reset_mock()

        print wlgen.RTA
        wlgen.RTA.pre_callback = assert_frozen

        executor = Executor(self.te, experiments_conf)
        executor.run()

        freezer_mock.assert_called_once_with(thaw=True)

    @patch('wlgen.RTA', InterceptedRTA)
    def test_freeze_userspace(self):
        self._do_freezer_test()

    @patch('wlgen.RTA', BrokenRTA)
    def test_freeze_userspace_broken(self):
        self._do_freezer_test()
