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

# Adding pexpect exceptions into this module's namespace
from pexpect import EOF, TIMEOUT  # NOQA pylint: disable=W0611

from wlauto.exceptions import HostError
from wlauto.utils.log import LogWriter


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


def pulse_dtr(conn, state=True, duration=0.1):
    """Set the DTR line of the specified serial connection to the specified state
    for the specified duration (note: the initial state of the line is *not* checked."""
    conn.setDTR(state)
    time.sleep(duration)
    conn.setDTR(not state)


def get_connection(timeout, init_dtr=None, *args, **kwargs):
    if init_dtr is not None:
        kwargs['dsrdtr'] = True
    try:
        conn = serial.Serial(*args, **kwargs)
    except serial.SerialException as e:
        raise HostError(e.message)
    if init_dtr is not None:
        conn.setDTR(init_dtr)
    conn.nonblocking()
    conn.flushOutput()
    target = fdpexpect.fdspawn(conn.fileno(), timeout=timeout)
    target.logfile_read = PexpectLogger('read')
    target.logfile_send = PexpectLogger('send')

    # Monkey-patching sendline to introduce a short delay after
    # chacters are sent to the serial. If two sendline s are issued
    # one after another the second one might start putting characters
    # into the serial device before the first one has finished, causing
    # corruption. The delay prevents that.
    tsln = target.sendline

    def sendline(x):
        tsln(x)
        time.sleep(0.1)

    target.sendline = sendline
    return target, conn


@contextmanager
def open_serial_connection(timeout, get_conn=False, init_dtr=None, *args, **kwargs):
    """
    Opens a serial connection to a device.

    :param timeout: timeout for the fdpexpect spawn object.
    :param conn: ``bool`` that specfies whether the underlying connection
                 object should be yielded as well.
    :param init_dtr: specifies the initial DTR state stat should be set.

    All arguments are passed into the __init__ of serial.Serial. See
    pyserial documentation for details:

        http://pyserial.sourceforge.net/pyserial_api.html#serial.Serial

    :returns: a pexpect spawn object connected to the device.
              See: http://pexpect.sourceforge.net/pexpect.html

    """
    target, conn = get_connection(timeout, init_dtr=init_dtr, *args, **kwargs)

    if get_conn:
        yield target, conn
    else:
        yield target

    target.close()  # Closes the file descriptor used by the conn.
    del conn
