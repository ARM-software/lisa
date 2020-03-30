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
import re
import copy

from lisa.wlgen.rta import RTA, Periodic, Ramp, Step, RunAndSync

from .utils import StorageTestCase, create_local_target, ASSET_DIR


# Disable DeprecationWarning since PerfAnalysis is deprecated
from lisa.analysis.rta import PerfAnalysis
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class RTABase(StorageTestCase):
    """
    Common functionality for testing RTA

    Doesn't have "Test" in the name so that nosetests doesn't try to run it
    directly
    """

    tools = ['rt-app']

    def get_expected_command(self, rta_wload):
        """Return the rt-app command we should execute when `run` is called"""
        rta_path = self.target.which('rt-app')
        json_path = os.path.join(rta_wload.run_dir, rta_wload.json)
        return '{} {} 2>&1'.format(rta_path, json_path)

    def setUp(self):
        super().setUp()
        self.target = create_local_target()

    def assert_output_file_exists(self, path):
        """Assert that a file was created"""
        path = os.path.join(self.res_dir, path)
        self.assertTrue(os.path.isfile(path),
                        'No output file {} from rt-app'.format(path))

    def assert_can_read_logfile(self, exp_tasks):
        """Assert that the perf_analysis module understands the log output"""
        analysis = PerfAnalysis.from_dir(self.res_dir)
        exp_tasks = [re.sub(r'-[0-9]+', '', task) for task in exp_tasks]
        self.assertSetEqual(set(exp_tasks), set(analysis.tasks))


class TestRTAProfile(RTABase):
    def _do_test(self, profile, exp_phases):
        rtapp = RTA.by_profile(
            self.target, name='test', profile=profile, res_dir=self.res_dir,
            calibration=None, log_stats=True)

        with open(rtapp.local_json) as f:
            conf = json.load(f, object_pairs_hook=OrderedDict)

        # Check that the configuration looks like we expect it to
        phases = list(conf['tasks']['test']['phases'].values())
        self.assertEqual(len(phases), len(exp_phases), 'Wrong number of phases')
        for phase, exp_phase in zip(phases, exp_phases):
            if 'cpus' not in exp_phase:
                exp_phase = copy.copy(exp_phase)
                exp_phase.update(
                    cpus=sorted(range(self.target.plat_info['cpus-count'])),
                    nodes_membind=sorted(range(self.target.plat_info['numa-nodes-count'])),
                )
            self.assertDictEqual(phase, exp_phase)

        # Try running the workload and check that it produces the expected log
        # files
        rtapp.run()

        # rtapp_cmds = [c for c in self.target.executed_commands if 'rt-app' in c]
        # self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

        self.assert_output_file_exists('output.log')
        self.assert_output_file_exists('test.json')
        self.assert_output_file_exists('rt-app-test-0.log')
        self.assert_can_read_logfile(exp_tasks=['test-0'])

    def test_profile_periodic_smoke(self):
        """
        Smoketest Periodic rt-app workload

        Creates a workload using Periodic, tests that the JSON has the expected
        content, then tests that it can be run.
        """

        profile = {
            "test": Periodic(period_ms=100, duty_cycle_pct=20, duration_s=1)
        }

        exp_phases = [
            {
                'loop': 10,
                'run': 20000,
                'timer': {
                    'period': 100000,
                    'ref': 'test'
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

        profile = {"test": Step(start_pct=100, end_pct=0, time_s=1)}

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
        profile = {"test": RunAndSync('my_barrier', time_s=1)}
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

        profile = {"test": light + ramp + heavy}

        exp_phases = [
            # Light phase:
            {
                "loop": 100,
                "run": 1000,
                "timer": {
                    "period": 10000,
                    "ref": "test"
                }
            },
            # Ramp phases:
            {
                "loop": 20,
                "run": 5000,
                "timer": {
                    "period": 50000,
                    "ref": "test"
                }
            },
            {
                "loop": 20,
                "run": 15000,
                "timer": {
                    "period": 50000,
                    "ref": "test"
                }
            },
            {
                "loop": 20,
                "run": 25000,
                "timer": {
                    "period": 50000,
                    "ref": "test"
                }
            },
            {
                "loop": 20,
                "run": 35000,
                "timer": {
                    "period": 50000,
                    "ref": "test"
                }
            },
            {
                "loop": 20,
                "run": 45000,
                "timer": {
                    "period": 50000,
                    "ref": "test"
                }
            },
            # Heavy phase:
            {
                "loop": 1,
                "run": 90000,
                "timer": {
                    "period": 100000,
                    "ref": "test"
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

        json_path = os.path.join(ASSET_DIR, 'mp3-short.json')

        with open(json_path, 'r') as fh:
            str_conf = fh.read()

        rtapp = RTA.by_str(
            self.target, name='test', str_conf=str_conf, res_dir=self.res_dir,
            max_duration_s=5, calibration=calibration)

        with open(rtapp.local_json, 'r') as fh:
            conf = json.load(fh)

        self.assertSetEqual(
            set(conf['tasks'].keys()),
            {'AudioTick', 'AudioOut', 'AudioTrack',
             'mp3.decoder', 'OMXCall'})

        # Would like to try running the workload but mp3-short.json has nonzero
        # 'priority' fields, and we probably don't have permission for that
        # unless we're root.
        if self.target.is_rooted:
            rtapp.run(as_root=True)

            # rtapp_cmds = [c for c in self.target.executed_commands
            #               if 'rt-app' in c]
            # self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

            self.assert_output_file_exists('output.log')
            self.assert_output_file_exists('test.json')

    def test_custom_smoke_calib(self):
        """Test RTA custom workload (providing calibration)"""
        calibration = min(self.target.plat_info['rtapp']['calib'].values())
        self._test_custom_smoke(calibration)

    def test_custom_smoke_no_calib(self):
        """Test RTA custom workload (providing no calibration)"""
        self._test_custom_smoke(None)


class TestRTACalibrationConf(RTABase):
    """Test setting the "calibration" field of rt-app config"""

    def _get_calib_conf(self, calibration):
        profile = {"test": Periodic()}
        rtapp = RTA.by_profile(
            self.target, name='test', res_dir=self.res_dir, profile=profile,
            calibration=calibration)

        with open(rtapp.local_json) as fh:
            return json.load(fh)['global']['calibration']

    def test_calibration_conf_pload_nodata(self):
        """Test that the smallest pload value is used"""
        conf = self._get_calib_conf(None)
        self.assertEqual(conf, 100, 'Calibration not set to minimum pload value')

    def test_calibration_conf_pload_int(self):
        """Test that the calibration value is returned as expected"""
        conf = self._get_calib_conf(666)
        self.assertEqual(conf, 666, 'Calibration value modified')

    def test_calibration_conf_pload_str(self):
        """Test that the calibration value is returned as expected"""
        conf = self._get_calib_conf('CPU0')
        self.assertEqual(conf, 'CPU0', 'Calibration value modified')

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
