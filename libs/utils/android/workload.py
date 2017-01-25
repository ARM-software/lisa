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
import re

from . import System

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
        self._log = logging.getLogger('Workload')

        # Set of data reported in output of each run
        self.trace_file = None
        self.nrg_report = None
        wloads = Workload.availables(self.target)
        self._log.info('Workloads available on target:')
        self._log.info('  %s', wloads)

    def _adb(self, cmd):
        return 'adb -s {} {}'.format(self._target.adb_name, cmd)


    def run(self, out_dir, collect='',
            **kwargs):
        raise RuntimeError('Not implemeted')

    def tracingStart(self):
        if 'ftrace' in self.collect and 'systrace' in self.collect:
            msg = 'ftrace and systrace cannot be used at the same time'
            raise ValueError(msg)
        # Start FTrace
        if 'ftrace' in self.collect:
            self.trace_file = os.path.join(self.out_dir, 'trace.dat')
            self._log.info('FTrace START')
            self._te.ftrace.start()
        # Start Systrace (mutually exclusive with ftrace)
        elif 'systrace' in self.collect:
            self.trace_file = os.path.join(self.out_dir, 'trace.html')
            # Get the systrace time
            match = re.search(r'systrace_([0-9]+)', self.collect)
            if match:
                self._trace_time = match.group(1)
            else:
                # TODO: must implement a CTRL+C based systrace stopping
                self._log.warning("Systrace time NOT defined, tracing for 10[s]")
                self._trace_time = 10
            self._log.info('Systrace START')
            self._systrace_output = System.systrace_start(
                self._te, self.trace_file, self._trace_time)
        # Initialize energy meter results
        if 'energy' in self.collect and self._te.emeter:
            self._te.emeter.reset()
            self._log.info('Energy meter STARTED')

    def tracingStop(self):
        # Collect energy meter results
        if 'energy' in self.collect and self._te.emeter:
            self.nrg_report = self._te.emeter.report(self.out_dir)
            self._log.info('Energy meter STOPPED')
        # Stop FTrace
        if 'ftrace' in self.collect:
            self._te.ftrace.stop()
            self._log.info('FTrace STOP')
            self._te.ftrace.get_trace(self.trace_file)
        # Stop Systrace (mutually exclusive with ftrace)
        elif 'systrace' in self.collect:
            if not self.systrace_output:
                self._log.warning('Systrace is not running!')
            else:
                self._log.info('Waiting systrace report [%s]...',
                                 self.trace_file)
                self.systrace_output.wait()
        # Dump a platform description
        self._te.platform_dump(self.out_dir)

# vim :set tabstop=4 shiftwidth=4 expandtab
