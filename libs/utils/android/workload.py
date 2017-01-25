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
import logging


class Workload(object):
    """
    Base class for Android related workloads
    """

    def __init__(self, test_env):
        """
        Initialized workloads available on the specified test environment

        test_env: target test environmen
        """
        self._te = test_env
        self._target = test_env.target
        self.te = test_env
        self.target = test_env.target
        self._log = logging.getLogger('Workload')

        # Set of data reported in output of each run
        self.trace_file = None
        self.nrg_report = None
        wloads = Workload.availables(self.target)
        self._log.info('Workloads available on target:')
        self._log.info('  %s', wloads)

    def _adb(self, cmd):
        return 'adb -s {} {}'.format(self.target.adb_name, cmd)


    def run(self, out_dir, collect='',
            **kwargs):
        raise RuntimeError('Not implemeted')

    def tracingStart(self):
        # Initialize energy meter results
        if 'energy' in self.collect and self._te.emeter:
            self._te.emeter.reset()
            self._log.info('Energy meter STARTED')

    def tracingStop(self):
        # Collect energy meter results
        if 'energy' in self.collect and self._te.emeter:
            self.nrg_report = self._te.emeter.report(self.out_dir)
            self._log.info('Energy meter STOPPED')
# vim :set tabstop=4 shiftwidth=4 expandtab
