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


import os
import stat
import logging
import subprocess
import re
import threading
import tempfile
import shutil
import time

import pexpect
from distutils.version import StrictVersion as V
if V(pexpect.__version__) < V('4.0.0'):
    import pxssh
else:
    from pexpect import pxssh
from pexpect import EOF, TIMEOUT, spawn

from devlib.exception import HostError, TargetError, TimeoutError
from devlib.utils.misc import which, strip_bash_colors, escape_single_quotes, check_output


ssh = None
scp = None
sshpass = None

logger = logging.getLogger('ssh')


def ssh_get_shell(host, username, password=None, keyfile=None, port=None, timeout=10, telnet=False, original_prompt=None):
    _check_env()
    start_time = time.time()
    while True:
        if telnet:
            if keyfile:
                raise ValueError('keyfile may not be used with a telnet connection.')
            conn = TelnetPxssh(original_prompt=original_prompt)
        else:  # ssh
            conn = pxssh.pxssh()

        try:
            if keyfile:
                conn.login(host, username, ssh_key=keyfile, port=port, login_timeout=timeout)
            else:
                conn.login(host, username, password, port=port, login_timeout=timeout)
            break
        except EOF:
            timeout -= time.time() - start_time
            if timeout <= 0:
                message = 'Could not connect to {}; is the host name correct?'
                raise TargetError(message.format(host))
            time.sleep(5)

    conn.setwinsize(500,200)
    conn.sendline('')
    conn.prompt()
    conn.setecho(False)
    return conn


class TelnetPxssh(pxssh.pxssh):
    # pylint: disable=arguments-differ

    def __init__(self, original_prompt):
        super(TelnetPxssh, self).__init__()
        self.original_prompt = original_prompt or r'[#$]'

    def login(self, server, username, password='', login_timeout=10,
              auto_prompt_reset=True, sync_multiplier=1, port=23):
        args = ['telnet']
        if username is not None:
            args += ['-l', username]
        args += [server, str(port)]
        cmd = ' '.join(args)

        spawn._spawn(self, cmd)  # pylint: disable=protected-access

        if password is None:
            i = self.expect([self.original_prompt, 'Login timed out'], timeout=login_timeout)
        else:
            i = self.expect('(?i)(?:password)', timeout=login_timeout)
            if i == 0:
                self.sendline(password)
                i = self.expect([self.original_prompt, 'Login incorrect'], timeout=login_timeout)
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


def check_keyfile(keyfile):
    """
    keyfile must have the right access premissions in order to be useable. If the specified
    file doesn't, create a temporary copy and set the right permissions for that.

    Returns either the ``keyfile`` (if the permissions on it are correct) or the path to a
    temporary copy with the right permissions.
    """
    desired_mask = stat.S_IWUSR | stat.S_IRUSR
    actual_mask = os.stat(keyfile).st_mode & 0xFF
    if actual_mask != desired_mask:
        tmp_file = os.path.join(tempfile.gettempdir(), os.path.basename(keyfile))
        shutil.copy(keyfile, tmp_file)
        os.chmod(tmp_file, desired_mask)
        return tmp_file
    else:  # permissions on keyfile are OK
        return keyfile


class SshConnection(object):

    default_password_prompt = '[sudo] password'
    max_cancel_attempts = 5
    default_timeout=10

    @property
    def name(self):
        return self.host

    def __init__(self,
                 host,
                 username,
                 password=None,
                 keyfile=None,
                 port=None,
                 timeout=None,
                 telnet=False,
                 password_prompt=None,
                 original_prompt=None,
                 ):
        self.host = host
        self.username = username
        self.password = password
        self.keyfile = check_keyfile(keyfile) if keyfile else keyfile
        self.port = port
        self.lock = threading.Lock()
        self.password_prompt = password_prompt if password_prompt is not None else self.default_password_prompt
        logger.debug('Logging in {}@{}'.format(username, host))
        timeout = timeout if timeout is not None else self.default_timeout
        self.conn = ssh_get_shell(host, username, password, self.keyfile, port, timeout, False, None)

    def push(self, source, dest, timeout=30):
        dest = '{}@{}:{}'.format(self.username, self.host, dest)
        return self._scp(source, dest, timeout)

    def pull(self, source, dest, timeout=30):
        source = '{}@{}:{}'.format(self.username, self.host, source)
        return self._scp(source, dest, timeout)

    def execute(self, command, timeout=None, check_exit_code=True, as_root=False, strip_colors=True):
        try:
            with self.lock:
                output = self._execute_and_wait_for_prompt(command, timeout, as_root, strip_colors)
                if check_exit_code:
                    exit_code_text = self._execute_and_wait_for_prompt('echo $?', strip_colors=strip_colors, log=False)
                    try:
                        exit_code = int(exit_code_text.split()[0])
                        if exit_code:
                            message = 'Got exit code {}\nfrom: {}\nOUTPUT: {}'
                            raise TargetError(message.format(exit_code, command, output))
                    except (ValueError, IndexError):
                        logger.warning('Could not get exit code for "{}",\ngot: "{}"'.format(command, exit_code_text))
                return output
        except EOF:
            raise TargetError('Connection lost.')

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        try:
            port_string = '-p {}'.format(self.port) if self.port else ''
            keyfile_string = '-i {}'.format(self.keyfile) if self.keyfile else ''
            if as_root:
                command = "sudo -- sh -c '{}'".format(command)
            command = '{} {} {} {}@{} {}'.format(ssh, keyfile_string, port_string, self.username, self.host, command)
            logger.debug(command)
            if self.password:
                command = _give_password(self.password, command)
            return subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=True)
        except EOF:
            raise TargetError('Connection lost.')

    def close(self):
        logger.debug('Logging out {}@{}'.format(self.username, self.host))
        self.conn.logout()

    def cancel_running_command(self):
        # simulate impatiently hitting ^C until command prompt appears
        logger.debug('Sending ^C')
        for _ in xrange(self.max_cancel_attempts):
            self.conn.sendline(chr(3))
            if self.conn.prompt(0.1):
                return True
        return False

    def _execute_and_wait_for_prompt(self, command, timeout=None, as_root=False, strip_colors=True, log=True):
        self.conn.prompt(0.1)  # clear an existing prompt if there is one.
        if as_root:
            command = "sudo -- sh -c '{}'".format(escape_single_quotes(command))
            if log:
                logger.debug(command)
            self.conn.sendline(command)
            if self.password:
                index = self.conn.expect_exact([self.password_prompt, TIMEOUT], timeout=0.5)
                if index == 0:
                    self.conn.sendline(self.password)
        else:  # not as_root
            if log:
                logger.debug(command)
            self.conn.sendline(command)
        timed_out = self._wait_for_prompt(timeout)
        # the regex removes line breaks potential introduced when writing
        # command to shell.
        output = process_backspaces(self.conn.before)
        output = re.sub(r'\r([^\n])', r'\1', output)
        if '\r\n' in output: # strip the echoed command
            output = output.split('\r\n', 1)[1]
        if timed_out:
            self.cancel_running_command()
            raise TimeoutError(command, output)
        if strip_colors:
            output = strip_bash_colors(output)
        return output

    def _wait_for_prompt(self, timeout=None):
        if timeout:
            return not self.conn.prompt(timeout)
        else:  # cannot timeout; wait forever
            while not self.conn.prompt(1):
                pass
            return False

    def _scp(self, source, dest, timeout=30):
        # NOTE: the version of scp in Ubuntu 12.04 occasionally (and bizarrely)
        # fails to connect to a device if port is explicitly specified using -P
        # option, even if it is the default port, 22. To minimize this problem,
        # only specify -P for scp if the port is *not* the default.
        port_string = '-P {}'.format(self.port) if (self.port and self.port != 22) else ''
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


class TelnetConnection(SshConnection):

    def __init__(self,
                 host,
                 username,
                 password=None,
                 port=None,
                 timeout=None,
                 password_prompt=None,
                 original_prompt=None,
                 ):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.keyfile = None
        self.lock = threading.Lock()
        self.password_prompt = password_prompt if password_prompt is not None else self.default_password_prompt
        logger.debug('Logging in {}@{}'.format(username, host))
        timeout = timeout if timeout is not None else self.default_timeout
        self.conn = ssh_get_shell(host, username, password, None, port, timeout, True, original_prompt)


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


def process_backspaces(text):
    chars = []
    for c in text:
        if c == chr(8) and chars:  # backspace
            chars.pop()
        else:
            chars.append(c)
    return ''.join(chars)
