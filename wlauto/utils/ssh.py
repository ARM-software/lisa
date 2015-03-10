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


import logging
import subprocess
import re

import pxssh
from pexpect import EOF, TIMEOUT, spawn

from wlauto.exceptions import HostError, DeviceError, TimeoutError, ConfigError
from wlauto.utils.misc import which, strip_bash_colors, escape_single_quotes, check_output


ssh = None
scp = None
sshpass = None

logger = logging.getLogger('ssh')


def ssh_get_shell(host, username, password=None, keyfile=None, port=None, timeout=10, telnet=False):
    _check_env()
    if telnet:
        if keyfile:
            raise ConfigError('keyfile may not be used with a telnet connection.')
        conn = TelnetConnection()
    else:  # ssh
        conn = pxssh.pxssh()
    try:
        if keyfile:
            conn.SSH_OPTS += ' -i {}'.format(keyfile)
            conn.login(host, username, port=port, login_timeout=timeout)
        else:
            conn.login(host, username, password, port=port, login_timeout=timeout)
    except EOF:
        raise DeviceError('Could not connect to {}; is the host name correct?'.format(host))
    return conn


class TelnetConnection(pxssh.pxssh):
    # pylint: disable=arguments-differ

    def login(self, server, username, password='', original_prompt=r'[#$]', login_timeout=10,
              auto_prompt_reset=True, sync_multiplier=1):
        cmd = 'telnet -l {} {}'.format(username, server)

        spawn._spawn(self, cmd)  # pylint: disable=protected-access
        i = self.expect('(?i)(?:password)', timeout=login_timeout)
        if i == 0:
            self.sendline(password)
            i = self.expect([original_prompt, 'Login incorrect'], timeout=login_timeout)
        else:
            raise pxssh.ExceptionPxssh('could not log in: did not see a password prompt')

        if i:
            raise pxssh.ExceptionPxssh('could not log in: password was incorrect')

        if not self.sync_original_prompt(sync_multiplier):
            self.close()
            raise pxssh.ExceptionPxssh('could not synchronize with original prompt')

        if auto_prompt_reset:
            if not self.set_unique_prompt():
                self.close()
                message = 'could not set shell prompt (recieved: {}, expected: {}).'
                raise pxssh.ExceptionPxssh(message.format(self.before, self.PROMPT))
        return True


class SshShell(object):

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.conn = None

    def login(self, host, username, password=None, keyfile=None, port=None, timeout=None, telnet=False):
        # pylint: disable=attribute-defined-outside-init
        logger.debug('Logging in {}@{}'.format(username, host))
        self.host = host
        self.username = username
        self.password = password
        self.keyfile = keyfile
        self.port = port
        timeout = self.timeout if timeout is None else timeout
        self.conn = ssh_get_shell(host, username, password, keyfile, port, timeout, telnet)

    def push_file(self, source, dest, timeout=30):
        dest = '{}@{}:{}'.format(self.username, self.host, dest)
        return self._scp(source, dest, timeout)

    def pull_file(self, source, dest, timeout=30):
        source = '{}@{}:{}'.format(self.username, self.host, source)
        return self._scp(source, dest, timeout)

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
        port_string = '-p {}'.format(self.port) if self.port else ''
        keyfile_string = '-i {}'.format(self.keyfile) if self.keyfile else ''
        command = '{} {} {} {}@{} {}'.format(ssh, keyfile_string, port_string, self.username, self.host, command)
        logger.debug(command)
        if self.password:
            command = _give_password(self.password, command)
        return subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=True)

    def execute(self, command, timeout=None, check_exit_code=True, as_root=False, strip_colors=True):
        output = self._execute_and_wait_for_prompt(command, timeout, as_root, strip_colors)
        if check_exit_code:
            exit_code = int(self._execute_and_wait_for_prompt('echo $?', strip_colors=strip_colors, log=False))
            if exit_code:
                message = 'Got exit code {}\nfrom: {}\nOUTPUT: {}'
                raise DeviceError(message.format(exit_code, command, output))
        return output

    def logout(self):
        logger.debug('Logging out {}@{}'.format(self.username, self.host))
        self.conn.logout()

    def _execute_and_wait_for_prompt(self, command, timeout=None, as_root=False, strip_colors=True, log=True):
        timeout = self.timeout if timeout is None else timeout
        if as_root:
            command = "sudo -- sh -c '{}'".format(escape_single_quotes(command))
            if log:
                logger.debug(command)
            self.conn.sendline(command)
            index = self.conn.expect_exact(['[sudo] password', TIMEOUT], timeout=0.5)
            if index == 0:
                self.conn.sendline(self.password)
            timed_out = not self.conn.prompt(timeout)
            output = re.sub(r'.*?{}'.format(re.escape(command)), '', self.conn.before, 1).strip()
        else:
            if log:
                logger.debug(command)
            self.conn.sendline(command)
            timed_out = not self.conn.prompt(timeout)
            # the regex removes line breaks potentiall introduced when writing
            # command to shell.
            command_index = re.sub(r' \r([^\n])', r'\1', self.conn.before).find(command)
            while not timed_out and command_index == -1:
                # In case of a "premature" timeout (i.e. timeout, but no hang,
                # so command completes afterwards), there may be a prompt from
                # the previous command completion in the serial output. This
                # checks for this case by making sure that the original command
                # is present in the serial output and waiting for the next
                # prompt if it is not.
                timed_out = not self.conn.prompt(timeout)
                command_index = re.sub(r' \r([^\n])', r'\1', self.conn.before).find(command)
            output = self.conn.before[command_index + len(command):].strip()
        if timed_out:
            raise TimeoutError(command, output)
        if strip_colors:
            output = strip_bash_colors(output)
        return output

    def _scp(self, source, dest, timeout=30):
        port_string = '-P {}'.format(self.port) if self.port else ''
        keyfile_string = '-i {}'.format(self.keyfile) if self.keyfile else ''
        command = '{} -r {} {} {} {}'.format(scp, keyfile_string, port_string, source, dest)
        pass_string = ''
        logger.debug(command)
        if self.password:
            command = _give_password(self.password, command)
        try:
            check_output(command, timeout=timeout, shell=True)
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(e.returncode, e.cmd.replace(pass_string, ''), e.output)
        except TimeoutError as e:
            raise TimeoutError(e.command.replace(pass_string, ''), e.output)


def _give_password(password, command):
    if not sshpass:
        raise HostError('Must have sshpass installed on the host in order to use password-based auth.')
    pass_string = "sshpass -p '{}' ".format(password)
    return pass_string + command


def _check_env():
    global ssh, scp, sshpass  # pylint: disable=global-statement
    if not ssh:
        ssh = which('ssh')
        scp = which('scp')
        sshpass = which('sshpass')
    if not (ssh and scp):
        raise HostError('OpenSSH must be installed on the host.')

