#    Copyright 2018 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import shutil

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.utils.android import LogcatMonitor

class LogcatCollector(CollectorBase):

    def __init__(self, target, regexps=None, logcat_format=None):
        super(LogcatCollector, self).__init__(target)
        self.regexps = regexps
        self.logcat_format = logcat_format
        self.output_path = None
        self._collecting = False
        self._prev_log = None
        self._monitor = None

    def reset(self):
        """
        Clear Collector data but do not interrupt collection
        """
        if not self._monitor:
            return

        if self._collecting:
            self._monitor.clear_log()
        elif self._prev_log:
            os.remove(self._prev_log)
            self._prev_log = None

    def start(self):
        """
        Start collecting logcat lines
        """
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")
        self._monitor = LogcatMonitor(self.target, self.regexps, logcat_format=self.logcat_format)
        if self._prev_log:
            # Append new data collection to previous collection
            self._monitor.start(self._prev_log)
        else:
            self._monitor.start(self.output_path)

        self._collecting = True

    def stop(self):
        """
        Stop collecting logcat lines
        """
        if not self._collecting:
            raise RuntimeError('Logcat monitor not running, nothing to stop')

        self._monitor.stop()
        self._collecting = False
        self._prev_log = self._monitor.logfile

    def set_output(self, output_path):
        self.output_path = output_path

    def get_data(self):
        if self.output_path is None:
            raise RuntimeError("No data collected.")
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
