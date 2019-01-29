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

from devlib.trace import TraceCollector
from devlib.utils.serial_port import get_connection


class SerialTraceCollector(TraceCollector):

    @property
    def collecting(self):
        return self._collecting

    def __init__(self, target, serial_port, baudrate, timeout=20):
        super(SerialTraceCollector, self).__init__(target)
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout = timeout

        self._serial_target = None
        self._conn = None
        self._tmpfile = None
        self._collecting = False

    def reset(self):
        if self._collecting:
            raise RuntimeError("reset was called whilst collecting")

        if self._tmpfile:
            self._tmpfile.close()
            self._tmpfile = None

    def start(self):
        if self._collecting:
            raise RuntimeError("start was called whilst collecting")


        self._tmpfile = NamedTemporaryFile()
        start_marker = "-------- Starting serial logging --------\n"
        self._tmpfile.write(start_marker.encode('utf-8'))

        self._serial_target, self._conn = get_connection(port=self.serial_port,
                                                         baudrate=self.baudrate,
                                                         timeout=self.timeout,
                                                         logfile=self._tmpfile,
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
        self._tmpfile.write(stop_marker.encode('utf-8'))

        self._collecting = False

    def get_trace(self, outfile):
        if self._collecting:
            raise RuntimeError("get_trace was called whilst collecting")

        self._tmpfile.flush()

        shutil.copy(self._tmpfile.name, outfile)

        self._tmpfile.close()
        self._tmpfile = None
