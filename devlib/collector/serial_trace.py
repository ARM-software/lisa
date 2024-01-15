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

import shutil
from tempfile import NamedTemporaryFile
from pexpect.exceptions import TIMEOUT

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.utils.serial_port import get_connection


class SerialTraceCollector(CollectorBase):

    @property
    def collecting(self):
        return self._collecting

    def __init__(self, target, serial_port, baudrate, timeout=20):
        super(SerialTraceCollector, self).__init__(target)
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.output_path = None

        self._serial_target = None
        self._conn = None
        self._outfile_fh = None
        self._collecting = False

    def reset(self):
        if self._collecting:
            raise RuntimeError("reset was called whilst collecting")

        if self._outfile_fh:
            self._outfile_fh.close()
            self._outfile_fh = None

    def start(self):
        if self._collecting:
            raise RuntimeError("start was called whilst collecting")
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")

        self._outfile_fh = open(self.output_path, 'wb')
        start_marker = "-------- Starting serial logging --------\n"
        self._outfile_fh.write(start_marker.encode('utf-8'))

        self._serial_target, self._conn = get_connection(port=self.serial_port,
                                                         baudrate=self.baudrate,
                                                         timeout=self.timeout,
                                                         logfile=self._outfile_fh,
                                                         init_dtr=0)
        self._collecting = True

    def stop(self):
        if not self._collecting:
            raise RuntimeError("stop was called whilst not collecting")

        # We expect the below to fail, but we need to get pexpect to
        # do something so that it interacts with the serial device,
        # and hence updates the logfile.
        try:
            self._serial_target.expect(".", timeout=1)
        except TIMEOUT:
            pass

        self._serial_target.close()
        del self._conn

        stop_marker = "-------- Stopping serial logging --------\n"
        self._outfile_fh.write(stop_marker.encode('utf-8'))
        self._outfile_fh.flush()
        self._outfile_fh.close()
        self._outfile_fh = None

        self._collecting = False

    def set_output(self, output_path):
        self.output_path = output_path

    def get_data(self):
        if self._collecting:
            raise RuntimeError("get_data was called whilst collecting")
        if self.output_path is None:
            raise RuntimeError("No data collected.")
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
