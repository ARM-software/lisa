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

import json
import os

from lisa.wlgen import PerfMessaging, PerfPipe
from lisa.self_tests.wlgen import WlgenSelfBase

class PerfBenchBase(WlgenSelfBase):
    """Base class for common testing of PerfBench workloads"""
    def _do_test_performance_json(self, fields):
        """Test performance.json was created with the required fields"""
        json_path = os.path.join(self.host_out_dir, 'performance.json')
        try:
            with open(json_path) as fh:
                perf_json = json.load(fh)
        except IOError:
            raise AssertionError(
                "Perf workload didn't create performance report file")

        for field in fields:
            msg = 'Perf performance report missing "{}" field'.format(field)
            self.assertIn(field, perf_json, msg)

class TestPerfMessaging(PerfBenchBase):
    tools = ['perf']

    def test_perf_messaging_smoke(self):
        """
        Test PerfMessaging workload

        Runs a perf messaging workload and tests that the expected output was
        produced.
        """
        perf = PerfMessaging(self.target, 'perf_messaing')
        perf.conf(group=1, loop=100, pipe=True, thread=True,
                  run_dir=self.target_run_dir)

        os.makedirs(self.host_out_dir)
        perf.run(out_dir=self.host_out_dir)

        self._do_test_performance_json(['ctime', 'performance'])

class TestPerfPipe(PerfBenchBase):
    tools = ['perf']

    def test_perf_pipe_smoke(self):
        """
        Test PerfPipe workload

        Runs a 'PerfPipe' workload and tests that the expected output was
        produced.
        """
        perf = PerfPipe(self.target, 'perfpipe')
        perf.conf(loop=100000)

        os.makedirs(self.host_out_dir)
        perf.run(out_dir=self.host_out_dir)

        self._do_test_performance_json(
            ['ctime', 'performance', 'usec/op', 'ops/sec'])

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
