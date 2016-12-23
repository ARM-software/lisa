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

from collections import OrderedDict
import json
import os
from unittest import TestCase

from wlgen import RTA, Periodic, Ramp, Step, Pulse # todo remove unused

dummy_calibration = {}

class TestProfile(TestCase):
    def test_profile_periodic_smoke(self):
        """
        Smoke test for a RTA workload using kind='profile' and Periodic
        """
        rtapp = RTA(None, 'test', calibration=dummy_calibration)

        rtapp.conf(
            kind = 'profile',
            params = {
                'task_p20': Periodic(
                    period_ms      = 100,
                    duty_cycle_pct = 20,
                    duration_s     = 5,
                ).get(),
            },
            run_dir='test',
        )

        with open(rtapp.json) as f:
            conf = json.load(f)

        [phase] = conf['tasks']['task_p20']['phases'].values()
        self.assertDictEqual(phase, {
            "loop": 50,
            "run": 20000,
            "timer": {
                "period": 100000,
                "ref": "task_p20"
            }
        })

class TestComposition(TestCase):
    def test_composition_smoke(self):
        """
        Smoke test for a RTA workload using __add__ to compose phases
        """
        rtapp = RTA(None, 'test', calibration=dummy_calibration)

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

class TestCustom(TestCase):
    def test_custom_smoke(self):
        """
        Smoke test for a custom workload
        """

        json_path = os.path.join(os.getenv('LISA_HOME'),
                                 'assets', 'mp3-short.json')
        rtapp = RTA(None, 'test', calibration=dummy_calibration)

        # Configure this RTApp instance to:
        rtapp.conf(kind='custom', params=json_path, duration=5, run_dir='test')

        with open(rtapp.json) as f:
            conf = json.load(f)

        # Convert to str because unicode
        tasks = set([str(k) for k in conf['tasks'].keys()])
        self.assertSetEqual(
            tasks,
            set(['AudioTick', 'AudioOut', 'AudioTrack',
                 'mp3.decoder', 'OMXCall']))
