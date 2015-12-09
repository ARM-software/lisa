#    Copyright 2014-2015 ARM Limited
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


"""
This module contains utilities for implemening device hard reset
using Netio 230 series power switches. This utilizes the KSHELL connection.

"""

import telnetlib
import socket
import re
import time
import logging


logger = logging.getLogger('NetIO')


class NetioError(Exception):
    pass


class KshellConnection(object):

    response_regex = re.compile(r'^(\d+) (.*?)\r\n')
    delay = 0.5

    def __init__(self, host='ippowerbar', port=1234, timeout=None):
        """Parameters are passed into ``telnetlib.Telnet`` -- see Python docs."""
        self.host = host
        self.port = port
        self.conn = telnetlib.Telnet(host, port, timeout)
        time.sleep(self.delay)  # give time to respond
        output = self.conn.read_very_eager()
        if 'HELLO' not in output:
            raise NetioError('Could not connect: did not see a HELLO. Got: {}'.format(output))

    def login(self, user, password):
        code, out = self.send_command('login {} {}\r\n'.format(user, password))
        if code != 250:
            raise NetioError('Login failed. Got: {} {}'.format(code, out))

    def enable_port(self, port):
        """Enable the power supply at the specified port."""
        self.set_port(port, 1)

    def disable_port(self, port):
        """Enable the power supply at the specified port."""
        self.set_port(port, 0)

    def set_port(self, port, value):
        code, out = self.send_command('port {} {}'.format(port, value))
        if code != 250:
            raise NetioError('Could not set {} on port {}. Got: {} {}'.format(value, port, code, out))

    def send_command(self, command):
        try:
            if command.startswith('login'):
                parts = command.split()
                parts[2] = '*' * len(parts[2])
                logger.debug(' '.join(parts))
            else:
                logger.debug(command)
            self.conn.write('{}\n'.format(command))
            time.sleep(self.delay)  # give time to respond
            out = self.conn.read_very_eager()
            match = self.response_regex.search(out)
            if not match:
                raise NetioError('Invalid response: {}'.format(out.strip()))
            logger.debug('response: {} {}'.format(match.group(1), match.group(2)))
            return int(match.group(1)), match.group(2)
        except socket.error as err:
            try:
                time.sleep(self.delay)  # give time to respond
                out = self.conn.read_very_eager()
                if out.startswith('130 CONNECTION TIMEOUT'):
                    raise NetioError('130 Timed out.')
            except EOFError:
                pass
            raise err

    def close(self):
        self.conn.close()
