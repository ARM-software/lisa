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

from collections import OrderedDict, namedtuple
import json
import os

from lisa.perf_analysis import PerfAnalysis
from lisa.wlgen.rta import RTA, Periodic, Ramp, Step, RunAndSync

from lisa.tests.lisa.utils import StorageTestCase, create_local_testenv

class RTABase(StorageTestCase):
    """
    Common functionality for testing RTA

    Doesn't have "Test" in the name so that nosetests doesn't try to run it
    directly
    """

    tools = ['rt-app']

    def get_expected_command(self, rta_wload):
        """Return the rt-app command we should execute when `run` is called"""
        rta_path = self.te.target.which('rt-app')
        json_path = os.path.join(rta_wload.run_dir, rta_wload.json)
        return '{} {} 2>&1'.format(rta_path, json_path)

    def setUp(self):
        super().setUp()
        self.te = create_local_testenv()

    def assert_output_file_exists(self, path):
        """Assert that a file was created"""
        path = os.path.join(self.res_dir, path)
        self.assertTrue(os.path.isfile(path),
                        'No output file {} from rt-app'.format(path))

    def assert_can_read_logfile(self, exp_tasks):
        """Assert that the perf_analysis module understands the log output"""
        pa = PerfAnalysis(self.res_dir)
        self.assertSetEqual(set(exp_tasks), set(pa.tasks()))

class TestRTAProfile(RTABase):
    def _do_test(self, profile, exp_phases):
        rtapp = RTA.by_profile(
            self.te, name='test', profile=profile, res_dir=self.res_dir,
            calibration=self.te.get_rtapp_calibration())

        with open(rtapp.local_json) as f:
            conf = json.load(f, object_pairs_hook=OrderedDict)

        # Check that the configuration looks like we expect it to
        phases = list(conf['tasks']['test_task']['phases'].values())
        self.assertEqual(len(phases), len(exp_phases), 'Wrong number of phases')
        for phase, exp_phase in zip(phases, exp_phases):
            self.assertDictEqual(phase, exp_phase)

        # Try running the workload and check that it produces the expected log
        # files
        rtapp.run()

        # rtapp_cmds = [c for c in self.te.target.executed_commands if 'rt-app' in c]
        # self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

        self.assert_output_file_exists('output.log')
        self.assert_output_file_exists('test.json')
        self.assert_output_file_exists('rt-app-test_task-0.log')
        self.assert_can_read_logfile(exp_tasks=['test_task'])

    def test_profile_periodic_smoke(self):
        """
        Smoketest Periodic rt-app workload

        Creates a workload using Periodic, tests that the JSON has the expected
        content, then tests that it can be run.
        """

        profile = {
            "test_task" : Periodic(period_ms=100, duty_cycle_pct=20, duration_s=1)
        }

        exp_phases = [
            {
                'loop': 10,
                'run': 20000,
                'timer': {
                    'period': 100000,
                    'ref': 'test_task'
                }
            }
        ]

        self._do_test(profile, exp_phases)

    def test_profile_step_smoke(self):
        """
        Smoketest Step rt-app workload

        Creates a workload using Step, tests that the JSON has the expected
        content, then tests that it can be run.
        """

        profile = {"test_task" : Step(start_pct=100, end_pct=0, time_s=1)}

        exp_phases = [
            {
                'run': 1000000,
                'loop': 1
            },
            {
                'sleep': 1000000,
                'loop': 1
            },
        ]

        self._do_test(profile, exp_phases)

    def test_profile_run_and_sync_smoke(self):
        profile = {"test_task" : RunAndSync('my_barrier', time_s=1)}
        exp_phases = [
            OrderedDict([
                ('loop', 1),
                ('run', 1000000),
                ('barrier', 'my_barrier')
            ])
        ]

        self._do_test(profile, exp_phases)

    def test_composition(self):
        """
        Test RTA task composition with __add__

        Creates a composed workload by +-ing RTATask objects, tests that the
        JSON has the expected content, then tests running the workload
        """
        light = Periodic(duty_cycle_pct=10, duration_s=1.0, period_ms=10)

        start_pct = 10
        end_pct = 90
        delta_pct = 20
        ramp = Ramp(start_pct=start_pct, end_pct=end_pct, delta_pct=delta_pct,
                    time_s=1, period_ms=50)

        heavy = Periodic(duty_cycle_pct=90, duration_s=0.1, period_ms=100)

        profile = {"test_task" : light + ramp + heavy}

        exp_phases = [
            # Light phase:
            {
                "loop": 100,
                "run": 1000,
                "timer": {
                    "period": 10000,
                    "ref": "test_task"
                }
            },
            # Ramp phases:
            {
                "loop": 20,
                "run": 5000,
                "timer": {
                    "period": 50000,
                    "ref": "test_task"
                }
            },
            {
                "loop": 20,
                "run": 15000,
                "timer": {
                    "period": 50000,
                    "ref": "test_task"
                }
            },
            {
                "loop": 20,
                "run": 25000,
                "timer": {
                    "period": 50000,
                    "ref": "test_task"
                }
            },
            {
                "loop": 20,
                "run": 35000,
                "timer": {
                    "period": 50000,
                    "ref": "test_task"
                }
            },
            {
                "loop": 20,
                "run": 45000,
                "timer": {
                    "period": 50000,
                    "ref": "test_task"
                }
            },
            # Heavy phase:
            {
                "loop": 1,
                "run": 90000,
                "timer": {
                    "period": 100000,
                    "ref": "test_task"
                }
            }]


        self._do_test(profile, exp_phases)

    def test_invalid_composition(self):
        """Test that you can't compose tasks with a delay in the second task"""
        t1 = Periodic()
        t2 = Periodic(delay_s=1)

        # Should work fine if delayed task is the first one
        try:
            t3 = t2 + t1
        except Exception as e:
            raise AssertionError("Couldn't compose tasks: {}".format(e))

        # But not the other way around
        with self.assertRaises(ValueError):
            t3 = t1 + t2


class TestRTACustom(RTABase):
    def _test_custom_smoke(self, calibration):
        """
        Test RTA custom workload

        Creates an rt-app workload using 'custom' and checks that the json
        roughly matches the file we provided. If we have root, attempts to run
        the workload.
        """

        #TODO: update the path to mp3-short.json
        json_path = os.path.join(os.getenv('LISA_HOME'),
                                 'lisa', 'assets', 'mp3-short.json')

        with open(json_path, 'r') as fh:
            str_conf = fh.read()

        rtapp = RTA.by_str(
            self.te, name='test', str_conf=str_conf, res_dir=self.res_dir,
            max_duration_s=5, calibration=calibration)

        with open(rtapp.local_json, 'r') as fh:
            conf = json.load(fh)

        # Convert k to str because the json loader gives us unicode strings
        tasks = set([str(k) for k in list(conf['tasks'].keys())])
        self.assertSetEqual(
            tasks,
            set(['AudioTick', 'AudioOut', 'AudioTrack',
                 'mp3.decoder', 'OMXCall']))

        # Would like to try running the workload but mp3-short.json has nonzero
        # 'priority' fields, and we probably don't have permission for that
        # unless we're root.
        if self.te.target.is_rooted:
            rtapp.run()

            # rtapp_cmds = [c for c in self.target.executed_commands
            #               if 'rt-app' in c]
            # self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

            self.assert_output_file_exists('output.log')
            self.assert_output_file_exists('test.json')

    def test_custom_smoke_calib(self):
        """Test RTA custom workload (providing calibration)"""
        self._test_custom_smoke(self.te.get_rtapp_calibration())

    def test_custom_smoke_no_calib(self):
        """Test RTA custom workload (providing no calibration)"""
        self._test_custom_smoke(None)


DummyBlModule = namedtuple('bl', ['bigs'])

class TestRTACalibrationConf(RTABase):
    """Test setting the "calibration" field of rt-app config"""
    def _get_calib_conf(self, calibration):
        profile = {"test_task" : Periodic()}
        rtapp = RTA.by_profile(
            self.te, name='test', res_dir=self.res_dir, profile=profile,
            calibration=calibration)

        with open(rtapp.local_json) as fh:
            return json.load(fh)['global']['calibration']

    def test_calibration_conf_pload(self):
        """Test that the smallest pload value is used, if provided"""
        conf = self._get_calib_conf(self.te.get_rtapp_calibration())
        self.assertEqual(conf, 100, 'Calibration not set to minimum pload value')

    def test_calibration_conf_bl(self):
        """Test that a big CPU is used if big.LITTLE data is available"""
        self.te.target.modules.append('bl')
        self.te.target.bl = DummyBlModule([1, 2])
        conf = self._get_calib_conf(None)
        self.assertIn(conf, ['CPU{}'.format(c) for c in self.te.target.bl.bigs],
                      'Calibration not set to use a big CPU')

    def test_calibration_conf_nodata(self):
        """Test that the last CPU is used if no data is available"""
        conf = self._get_calib_conf(None)
        cpu = self.te.target.number_of_cpus - 1
        self.assertEqual(conf, 'CPU{}'.format(cpu),
                         'Calibration not set to highest numbered CPU')

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
