#    Copyright 2013-2015 ARM Limited
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


import time
from contextlib import contextmanager
from distutils.version import StrictVersion as V

import serial

# pylint: disable=ungrouped-imports
import pexpect
if V(pexpect.__version__) < V('4.0.0'):
    import fdpexpect  # pylint: disable=import-error
else:
    from pexpect import fdpexpect

from wlauto.utils.log import LogWriter
from devlib.utils.serial_port import pulse_dtr, get_connection, open_serial_connection


class PexpectLogger(LogWriter):

    def __init__(self, kind):
        """
        File-like object class designed to be used for logging with pexpect or
        fdpexect. Each complete line (terminated by new line character) gets logged
        at DEBUG level. In complete lines are buffered until the next new line.

        :param kind: This specified which of pexpect logfile attributes this logger
                    will be set to. It should be "read" for logfile_read, "send" for
                    logfile_send, and "" (emtpy string) for logfile.

        """
        if kind not in ('read', 'send', ''):
            raise ValueError('kind must be "read", "send" or ""; got {}'.format(kind))
        self.kind = kind
        logger_name = 'serial_{}'.format(kind) if kind else 'serial'
        super(PexpectLogger, self).__init__(logger_name)
