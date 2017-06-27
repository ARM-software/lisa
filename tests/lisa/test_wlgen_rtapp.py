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

from collections import OrderedDict
import json
import os

from perf_analysis import PerfAnalysis
from wlgen import RTA, Periodic, Ramp

from test_wlgen import WlgenSelfBase

class RTABase(WlgenSelfBase):
    """
    Common functionality for testing RTA

    Doesn't have "Test" in the name so that nosetests doesn't try to run it
    directly
    """

    tools = ['rt-app']

    def get_expected_command(self, rta_wload):
        """Return the rt-app command we should execute when `run` is called"""
        rta_path = os.path.join(self.target.executables_directory, 'rt-app')
        json_path = os.path.join(rta_wload.run_dir, rta_wload.json)
        return '{} {} 2>&1'.format(rta_path, json_path)

    def setUp(self):
        super(RTABase, self).setUp()

        # Can't calibrate rt-app because:
        # - Need to set performance governor
        # - Need to use SCHED_FIFO + high priority
        # We probably don't have permissions so use a dummy calibration.
        self.calibration = {c: 100
                           for c in range(len(self.target.cpuinfo.cpu_names))}

        os.makedirs(self.host_out_dir)

    def assert_output_file_exists(self, path):
        """Assert that a file was created in host_out_dir"""
        path = os.path.join(self.host_out_dir, path)
        self.assertTrue(os.path.isfile(path),
                        'No output file {} from rt-app'.format(path))

    def assert_can_read_logfile(self, exp_tasks):
        """Assert that the perf_analysis module understands the log output"""
        pa = PerfAnalysis(self.host_out_dir)
        self.assertSetEqual(set(exp_tasks), set(pa.tasks()))

class TestRTAProfile(RTABase):
    def test_profile_periodic_smoke(self):
        """
        Smoketest Periodic rt-app workload

        Creates a workload using Periodic, tests that the JSON has the expected
        content, then tests that it can be run.
        """
        rtapp = RTA(self.target, name='test', calibration=self.calibration)

        rtapp.conf(
            kind = 'profile',
            params = {
                'task_p20': Periodic(
                    period_ms      = 100,
                    duty_cycle_pct = 20,
                    duration_s     = 1,
                ).get(),
            },
            run_dir=self.target_run_dir
        )

        with open(rtapp.json) as f:
            conf = json.load(f)

        [phase] = conf['tasks']['task_p20']['phases'].values()
        self.assertDictEqual(phase, {
            'loop': 10,
            'run': 20000,
            'timer': {
                'period': 100000,
                'ref': 'task_p20'
            }
        })
        rtapp.run(out_dir=self.host_out_dir)

        rtapp_cmds = [c for c in self.target.executed_commands if 'rt-app' in c]
        self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

        self.assert_output_file_exists('output.log')
        self.assert_output_file_exists('test_00.json')
        self.assert_output_file_exists('rt-app-task_p20-0.log')
        self.assert_can_read_logfile(exp_tasks=['task_p20'])

class TestRTAComposition(RTABase):
    def test_composition(self):
        """
        Test RTA task composition with __add__

        Creates a composed workload by +-ing RTATask objects, tests that the
        JSON has the expected content, then tests running the workload
        """
        rtapp = RTA(self.target, name='test', calibration=self.calibration)

        light  = Periodic(duty_cycle_pct=10, duration_s=1.0, period_ms=10)

        start_pct = 10
        end_pct = 90
        delta_pct = 20
        num_ramp_phases = ((end_pct - start_pct) / delta_pct) + 1
        ramp = Ramp(start_pct=start_pct, end_pct=end_pct, delta_pct=delta_pct,
                    time_s=1, period_ms=50)

        heavy = Periodic(duty_cycle_pct=90, duration_s=0.1, period_ms=100)

        lrh_task = light + ramp + heavy

        rtapp.conf(
            kind = 'profile',
            params = {
                'task_ramp': lrh_task.get()
            },
            run_dir=self.target_run_dir
        )

        with open(rtapp.json) as f:
            conf = json.load(f, object_pairs_hook=OrderedDict)

        phases = conf['tasks']['task_ramp']['phases'].values()

        exp_phases = [
            # Light phase:
            {
                "loop": 100,
                "run": 1000,
                "timer": {
                    "period": 10000,
                    "ref": "task_ramp"
                }
            },
            # Ramp phases:
            {
                "loop": 20,
                "run": 5000,
                "timer": {
                    "period": 50000,
                    "ref": "task_ramp"
                }
            },
            {
                "loop": 20,
                "run": 15000,
                "timer": {
                    "period": 50000,
                    "ref": "task_ramp"
                }
            },
            {
                "loop": 20,
                "run": 25000,
                "timer": {
                    "period": 50000,
                    "ref": "task_ramp"
                }
            },
            {
                "loop": 20,
                "run": 35000,
                "timer": {
                    "period": 50000,
                    "ref": "task_ramp"
                }
            },
            {
                "loop": 20,
                "run": 45000,
                "timer": {
                    "period": 50000,
                    "ref": "task_ramp"
                }
            },
            # Heavy phase:
            {
                "loop": 1,
                "run": 90000,
                "timer": {
                    "period": 100000,
                    "ref": "task_ramp"
                }
            }]

        self.assertListEqual(phases, exp_phases)

        rtapp.run(out_dir=self.host_out_dir)

        rtapp_cmds = [c for c in self.target.executed_commands if 'rt-app' in c]
        self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

        self.assert_output_file_exists('output.log')
        self.assert_output_file_exists('test_00.json')
        self.assert_output_file_exists('rt-app-task_ramp-0.log')
        self.assert_can_read_logfile(exp_tasks=['task_ramp'])


class TestRTACustom(RTABase):
    def _test_custom_smoke(self, calibration):
        """
        Test RTA custom workload

        Creates an rt-app workload using 'custom' and checks that the json
        roughly matches the file we provided. If we have root, attempts to run
        the workload.
        """

        json_path = os.path.join(os.getenv('LISA_HOME'),
                                 'assets', 'mp3-short.json')
        rtapp = RTA(self.target, name='test', calibration=calibration)

        # Configure this RTApp instance to:
        rtapp.conf(kind='custom', params=json_path, duration=5,
                   run_dir=self.target_run_dir)

        with open(rtapp.json) as f:
            conf = json.load(f)

        # Convert k to str because the json loader gives us unicode strings
        tasks = set([str(k) for k in conf['tasks'].keys()])
        self.assertSetEqual(
            tasks,
            set(['AudioTick', 'AudioOut', 'AudioTrack',
                 'mp3.decoder', 'OMXCall']))

        # Would like to try running the workload but mp3-short.json has nonzero
        # 'priority' fields, and we probably don't have permission for that
        # unless we're root.
        if self.target.is_rooted:
            rtapp.run(out_dir=self.host_out_dir)

            rtapp_cmds = [c for c in self.target.executed_commands
                          if 'rt-app' in c]
            self.assertListEqual(rtapp_cmds, [self.get_expected_command(rtapp)])

            self.assert_output_file_exists('output.log')
            self.assert_output_file_exists('test_00.json')

    def test_custom_smoke_calib(self):
        """Test RTA custom workload (providing calibration)"""
        self._test_custom_smoke(self.calibration)

    def test_custom_smoke_no_calib(self):
        """Test RTA custom workload (providing no calibration)"""
        self._test_custom_smoke(None)
