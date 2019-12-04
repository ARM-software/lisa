#    Copyright 2014-2018 ARM Limited
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
import socket
import sys
import time
import atexit
from pipes import quote
from future.utils import raise_from

# pylint: disable=import-error,wrong-import-position,ungrouped-imports,wrong-import-order
import pexpect
from distutils.version import StrictVersion as V
if V(pexpect.__version__) < V('4.0.0'):
    import pxssh
else:
    from pexpect import pxssh
from pexpect import EOF, TIMEOUT, spawn

# pylint: disable=redefined-builtin,wrong-import-position
from devlib.exception import (HostError, TargetStableError, TargetNotRespondingError,
                              TimeoutError, TargetTransientError)
from devlib.utils.misc import (which, strip_bash_colors, check_output,
                               sanitize_cmd_template, memoized)
from devlib.utils.types import boolean


ssh = None
scp = None
sshpass = None


logger = logging.getLogger('ssh')
gem5_logger = logging.getLogger('gem5-connection')

def ssh_get_shell(host,
                  username,
                  password=None,
                  keyfile=None,
                  port=None,
                  timeout=10,
                  telnet=False,
                  original_prompt=None,
                  options=None):
    _check_env()
    start_time = time.time()
    while True:
        if telnet:
            if keyfile:
                raise ValueError('keyfile may not be used with a telnet connection.')
            conn = TelnetPxssh(original_prompt=original_prompt)
        else:  # ssh
            conn = pxssh.pxssh(options=options,
                               echo=False)

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
                raise TargetTransientError(message.format(host))
            time.sleep(5)

    conn.setwinsize(500, 200)
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

        try:
            i = self.expect('(?i)(?:password)', timeout=login_timeout)
            if i == 0:
                self.sendline(password)
                i = self.expect([self.original_prompt, 'Login incorrect'], timeout=login_timeout)
            if i:
                raise pxssh.ExceptionPxssh('could not log in: password was incorrect')
        except TIMEOUT:
            if not password:
                # No password promt before TIMEOUT & no password provided
                # so assume everything is okay
                pass
            else:
                raise pxssh.ExceptionPxssh('could not log in: did not see a password prompt')

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
    default_timeout = 10

    @property
    def name(self):
        return self.host

    @property
    def connected_as_root(self):
        if self._connected_as_root is None:
            # Execute directly to prevent deadlocking of connection
            result = self._execute_and_wait_for_prompt('id', as_root=False)
            self._connected_as_root = 'uid=0(' in result
        return self._connected_as_root

    @connected_as_root.setter
    def connected_as_root(self, state):
        self._connected_as_root = state

    # pylint: disable=unused-argument,super-init-not-called
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
                 platform=None,
                 sudo_cmd="sudo -- sh -c {}",
                 options=None
                 ):
        self._connected_as_root = None
        self.host = host
        self.username = username
        self.password = password
        self.keyfile = check_keyfile(keyfile) if keyfile else keyfile
        self.port = port
        self.lock = threading.Lock()
        self.password_prompt = password_prompt if password_prompt is not None else self.default_password_prompt
        self.sudo_cmd = sanitize_cmd_template(sudo_cmd)
        logger.debug('Logging in {}@{}'.format(username, host))
        timeout = timeout if timeout is not None else self.default_timeout
        self.options = options if options is not None else {}
        self.conn = ssh_get_shell(host,
                                  username,
                                  password,
                                  self.keyfile,
                                  port,
                                  timeout,
                                  False,
                                  None,
                                  self.options)
        atexit.register(self.close)

    def push(self, source, dest, timeout=30):
        dest = '{}@{}:{}'.format(self.username, self.host, dest)
        return self._scp(source, dest, timeout)

    def pull(self, source, dest, timeout=30):
        source = '{}@{}:{}'.format(self.username, self.host, source)
        return self._scp(source, dest, timeout)

    def execute(self, command, timeout=None, check_exit_code=True,
                as_root=False, strip_colors=True, will_succeed=False): #pylint: disable=unused-argument
        if command == '':
            # Empty command is valid but the __devlib_ec stuff below will
            # produce a syntax error with bash. Treat as a special case.
            return ''
        try:
            with self.lock:
                _command = '({}); __devlib_ec=$?; echo; echo $__devlib_ec'.format(command)
                full_output = self._execute_and_wait_for_prompt(_command, timeout, as_root, strip_colors)
                split_output = full_output.rsplit('\r\n', 2)
                try:
                    output, exit_code_text, _ = split_output
                except ValueError as e:
                    raise TargetStableError(
                        "cannot split reply (target misconfiguration?):\n'{}'".format(full_output))
                if check_exit_code:
                    try:
                        exit_code = int(exit_code_text)
                        if exit_code:
                            message = 'Got exit code {}\nfrom: {}\nOUTPUT: {}'
                            raise TargetStableError(message.format(exit_code, command, output))
                    except (ValueError, IndexError):
                        logger.warning(
                            'Could not get exit code for "{}",\ngot: "{}"'\
                            .format(command, exit_code_text))
                return output
        except EOF:
            raise TargetNotRespondingError('Connection lost.')
        except TargetStableError as e:
            if will_succeed:
                raise TargetTransientError(e)
            else:
                raise

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        try:
            port_string = '-p {}'.format(self.port) if self.port else ''
            keyfile_string = '-i {}'.format(self.keyfile) if self.keyfile else ''
            if as_root and not self.connected_as_root:
                command = self.sudo_cmd.format(command)
            options = " ".join([ "-o {}={}".format(key,val)
                                for key,val in self.options.items()])
            command = '{} {} {} {} {}@{} {}'.format(ssh,
                                                    options,
                                                    keyfile_string,
                                                    port_string,
                                                    self.username,
                                                    self.host,
                                                    command)
            logger.debug(command)
            if self.password:
                command, _ = _give_password(self.password, command)
            return subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=True)
        except EOF:
            raise TargetNotRespondingError('Connection lost.')

    def close(self):
        logger.debug('Logging out {}@{}'.format(self.username, self.host))
        try:
            self.conn.logout()
        except:
            logger.debug('Connection lost.')
            self.conn.close(force=True)

    def cancel_running_command(self):
        # simulate impatiently hitting ^C until command prompt appears
        logger.debug('Sending ^C')
        for _ in range(self.max_cancel_attempts):
            self._sendline(chr(3))
            if self.conn.prompt(0.1):
                return True
        return False

    def _execute_and_wait_for_prompt(self, command, timeout=None, as_root=False, strip_colors=True, log=True):
        self.conn.prompt(0.1)  # clear an existing prompt if there is one.
        if as_root and self.connected_as_root:
            # As we're already root, there is no need to use sudo.
            as_root = False
        if as_root:
            command = self.sudo_cmd.format(quote(command))
            if log:
                logger.debug(command)
            self._sendline(command)
            if self.password:
                index = self.conn.expect_exact([self.password_prompt, TIMEOUT], timeout=0.5)
                if index == 0:
                    self._sendline(self.password)
        else:  # not as_root
            if log:
                logger.debug(command)
            self._sendline(command)
        timed_out = self._wait_for_prompt(timeout)
        if sys.version_info[0] == 3:
            output = process_backspaces(self.conn.before.decode(sys.stdout.encoding or 'utf-8', 'replace'))
        else:
            output = process_backspaces(self.conn.before)

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
        port_string = '-P {}'.format(quote(str(self.port))) if (self.port and self.port != 22) else ''
        keyfile_string = '-i {}'.format(quote(self.keyfile)) if self.keyfile else ''
        options = " ".join(["-o {}={}".format(key,val)
                            for key,val in self.options.items()])
        command = '{} {} -r {} {} {} {}'.format(scp,
                                                options,
                                                keyfile_string,
                                                port_string,
                                                quote(source),
                                                quote(dest))
        command_redacted = command
        logger.debug(command)
        if self.password:
            command, command_redacted = _give_password(self.password, command)
        try:
            check_output(command, timeout=timeout, shell=True)
        except subprocess.CalledProcessError as e:
            raise_from(HostError("Failed to copy file with '{}'. Output:\n{}".format(
                command_redacted, e.output)), None)
        except TimeoutError as e:
            raise TimeoutError(command_redacted, e.output)

    def _sendline(self, command):
        # Workaround for https://github.com/pexpect/pexpect/issues/552
        if len(command) == self._get_window_size()[1] - self._get_prompt_length():
            command += ' '
        self.conn.sendline(command)

    @memoized
    def _get_prompt_length(self):
        self.conn.sendline()
        self.conn.prompt()
        return len(self.conn.after)

    @memoized
    def _get_window_size(self):
        return self.conn.getwinsize()

class TelnetConnection(SshConnection):

    # pylint: disable=super-init-not-called
    def __init__(self,
                 host,
                 username,
                 password=None,
                 port=None,
                 timeout=None,
                 password_prompt=None,
                 original_prompt=None,
                 platform=None):
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


class Gem5Connection(TelnetConnection):

    # pylint: disable=super-init-not-called
    def __init__(self,
                 platform,
                 host=None,
                 username=None,
                 password=None,
                 port=None,
                 timeout=None,
                 password_prompt=None,
                 original_prompt=None,
                 strip_echoed_commands=False,
                 ):
        if host is not None:
            host_system = socket.gethostname()
            if host_system != host:
                raise TargetStableError("Gem5Connection can only connect to gem5 "
                                  "simulations on your current host {}, which "
                                  "differs from the one given {}!"
                                  .format(host_system, host))
        if username is not None and username != 'root':
            raise ValueError('User should be root in gem5!')
        if password is not None and password != '':
            raise ValueError('No password needed in gem5!')
        self.username = 'root'
        self.is_rooted = True
        self.password = None
        self.port = None
        # Flag to indicate whether commands are echoed by the simulated system
        self.strip_echoed_commands = strip_echoed_commands
        # Long timeouts to account for gem5 being slow
        # Can be overriden if the given timeout is longer
        self.default_timeout = 3600
        if timeout is not None:
            if timeout > self.default_timeout:
                logger.info('Overwriting the default timeout of gem5 ({})'
                                 ' to {}'.format(self.default_timeout, timeout))
                self.default_timeout = timeout
            else:
                logger.info('Ignoring the given timeout --> gem5 needs longer timeouts')
        self.ready_timeout = self.default_timeout * 3
        # Counterpart in gem5_interact_dir
        self.gem5_input_dir = '/mnt/host/'
        # Location of m5 binary in the gem5 simulated system
        self.m5_path = None
        # Actual telnet connection to gem5 simulation
        self.conn = None
        # Flag to indicate the gem5 device is ready to interact with the
        # outer world
        self.ready = False
        # Lock file to prevent multiple connections to same gem5 simulation
        # (gem5 does not allow this)
        self.lock_directory = '/tmp/'
        self.lock_file_name = None # Will be set once connected to gem5

        # These parameters will be set by either the method to connect to the
        # gem5 platform or directly to the gem5 simulation
        # Intermediate directory to push things to gem5 using VirtIO
        self.gem5_interact_dir = None
        # Directory to store output  from gem5 on the host
        self.gem5_out_dir = None
        # Actual gem5 simulation
        self.gem5simulation = None

        # Connect to gem5
        if platform:
            self._connect_gem5_platform(platform)

        # Wait for boot
        self._wait_for_boot()

        # Mount the virtIO to transfer files in/out gem5 system
        self._mount_virtio()

    def set_hostinteractdir(self, indir):
        logger.info('Setting hostinteractdir  from {} to {}'
                    .format(self.gem5_input_dir, indir))
        self.gem5_input_dir = indir

    def push(self, source, dest, timeout=None):
        """
        Push a file to the gem5 device using VirtIO

        The file to push to the device is copied to the temporary directory on
        the host, before being copied within the simulation to the destination.
        Checks, in the form of 'ls' with error code checking, are performed to
        ensure that the file is copied to the destination.
        """
        # First check if the connection is set up to interact with gem5
        self._check_ready()

        filename = os.path.basename(source)
        logger.debug("Pushing {} to device.".format(source))
        logger.debug("gem5interactdir: {}".format(self.gem5_interact_dir))
        logger.debug("dest: {}".format(dest))
        logger.debug("filename: {}".format(filename))

        # We need to copy the file to copy to the temporary directory
        self._move_to_temp_dir(source)

        # Dest in gem5 world is a file rather than directory
        if os.path.basename(dest) != filename:
            dest = os.path.join(dest, filename)
        # Back to the gem5 world
        filename = quote(self.gem5_input_dir + filename)
        self._gem5_shell("ls -al {}".format(filename))
        self._gem5_shell("cat {} > {}".format(filename, quote(dest)))
        self._gem5_shell("sync")
        self._gem5_shell("ls -al {}".format(quote(dest)))
        self._gem5_shell("ls -al {}".format(quote(self.gem5_input_dir)))
        logger.debug("Push complete.")

    def pull(self, source, dest, timeout=0): #pylint: disable=unused-argument
        """
        Pull a file from the gem5 device using m5 writefile

        The file is copied to the local directory within the guest as the m5
        writefile command assumes that the file is local. The file is then
        written out to the host system using writefile, prior to being moved to
        the destination on the host.
        """
        # First check if the connection is set up to interact with gem5
        self._check_ready()

        result = self._gem5_shell("ls {}".format(source))
        files = strip_bash_colors(result).split()

        for filename in files:
            dest_file = os.path.basename(filename)
            logger.debug("pull_file {} {}".format(filename, dest_file))
            # writefile needs the file to be copied to be in the current
            # working directory so if needed, copy to the working directory
            # We don't check the exit code here because it is non-zero if the
            # source and destination are the same. The ls below will cause an
            # error if the file was not where we expected it to be.
            if os.path.isabs(source):
                if os.path.dirname(source) != self.execute('pwd',
                                              check_exit_code=False):
                    self._gem5_shell("cat {} > {}".format(quote(filename),
                                                              quote(dest_file)))
            self._gem5_shell("sync")
            self._gem5_shell("ls -la {}".format(dest_file))
            logger.debug('Finished the copy in the simulator')
            self._gem5_util("writefile {}".format(dest_file))

            if 'cpu' not in filename:
                while not os.path.exists(os.path.join(self.gem5_out_dir,
                                                      dest_file)):
                    time.sleep(1)

            # Perform the local move
            if os.path.exists(os.path.join(dest, dest_file)):
                logger.warning(
                            'Destination file {} already exists!'\
                            .format(dest_file))
            else:
                shutil.move(os.path.join(self.gem5_out_dir, dest_file), dest)
            logger.debug("Pull complete.")

    def execute(self, command, timeout=1000, check_exit_code=True,
                as_root=False, strip_colors=True, will_succeed=False):
        """
        Execute a command on the gem5 platform
        """
        # First check if the connection is set up to interact with gem5
        self._check_ready()

        try:
            output = self._gem5_shell(command,
                                      check_exit_code=check_exit_code,
                                      as_root=as_root)
        except TargetStableError as e:
            if will_succeed:
                raise TargetTransientError(e)
            else:
                raise

        if strip_colors:
            output = strip_bash_colors(output)
        return output

    def background(self, command, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, as_root=False):
        # First check if the connection is set up to interact with gem5
        self._check_ready()

        # Create the logfile for stderr/stdout redirection
        command_name = command.split(' ')[0].split('/')[-1]
        redirection_file = 'BACKGROUND_{}.log'.format(command_name)
        trial = 0
        while os.path.isfile(redirection_file):
            # Log file already exists so add to name
            redirection_file = 'BACKGROUND_{}{}.log'.format(command_name, trial)
            trial += 1

        # Create the command to pass on to gem5 shell
        complete_command = '{} >> {} 2>&1 &'.format(command, redirection_file)
        output = self._gem5_shell(complete_command, as_root=as_root)
        output = strip_bash_colors(output)
        gem5_logger.info('STDERR/STDOUT of background command will be '
                         'redirected to {}. Use target.pull() to '
                         'get this file'.format(redirection_file))
        return output

    def close(self):
        """
        Close and disconnect from the gem5 simulation. Additionally, we remove
        the temporary directory used to pass files into the simulation.
        """
        gem5_logger.info("Gracefully terminating the gem5 simulation.")
        try:
            # Unmount the virtio device BEFORE we kill the
            # simulation. This is done to simplify checkpointing at
            # the end of a simulation!
            self._unmount_virtio()
            self._gem5_util("exit")
            self.gem5simulation.wait()
        except EOF:
            pass
        gem5_logger.info("Removing the temporary directory")
        try:
            shutil.rmtree(self.gem5_interact_dir)
        except OSError:
            gem5_logger.warning("Failed to remove the temporary directory!")

        # Delete the lock file
        os.remove(self.lock_file_name)

    # Functions only to be called by the Gem5 connection itself
    def _connect_gem5_platform(self, platform):
        port = platform.gem5_port
        gem5_simulation = platform.gem5
        gem5_interact_dir = platform.gem5_interact_dir
        gem5_out_dir = platform.gem5_out_dir

        self.connect_gem5(port, gem5_simulation, gem5_interact_dir, gem5_out_dir)

    # Handle the EOF exception raised by pexpect
    # pylint: disable=no-self-use
    def _gem5_EOF_handler(self, gem5_simulation, gem5_out_dir, err):
        # If we have reached the "EOF", it typically means
        # that gem5 crashed and closed the connection. Let's
        # check and actually tell the user what happened here,
        # rather than spewing out pexpect errors.
        if gem5_simulation.poll():
            message = "The gem5 process has crashed with error code {}!\n\tPlease see {} for details."
            raise TargetNotRespondingError(message.format(gem5_simulation.poll(), gem5_out_dir))
        else:
            # Let's re-throw the exception in this case.
            raise err

    # This function connects to the gem5 simulation
    # pylint: disable=too-many-statements
    def connect_gem5(self, port, gem5_simulation, gem5_interact_dir,
                      gem5_out_dir):
        """
        Connect to the telnet port of the gem5 simulation.

        We connect, and wait for the prompt to be found. We do not use a timeout
        for this, and wait for the prompt in a while loop as the gem5 simulation
        can take many hours to reach a prompt when booting the system. We also
        inject some newlines periodically to try and force gem5 to show a
        prompt. Once the prompt has been found, we replace it with a unique
        prompt to ensure that we are able to match it properly. We also disable
        the echo as this simplifies parsing the output when executing commands
        on the device.
        """
        host = socket.gethostname()
        gem5_logger.info("Connecting to the gem5 simulation on port {}".format(port))

        # Check if there is no on-going connection yet
        lock_file_name = '{}{}_{}.LOCK'.format(self.lock_directory, host, port)
        if os.path.isfile(lock_file_name):
            # There is already a connection to this gem5 simulation
            raise TargetStableError('There is already a connection to the gem5 '
                              'simulation using port {} on {}!'
                              .format(port, host))

        # Connect to the gem5 telnet port. Use a short timeout here.
        attempts = 0
        while attempts < 10:
            attempts += 1
            try:
                self.conn = TelnetPxssh(original_prompt=None)
                self.conn.login(host, self.username, port=port,
                                login_timeout=10, auto_prompt_reset=False)
                break
            except pxssh.ExceptionPxssh:
                pass
            except EOF as err:
                self._gem5_EOF_handler(gem5_simulation, gem5_out_dir, err)
        else:
            gem5_simulation.kill()
            raise TargetNotRespondingError("Failed to connect to the gem5 telnet session.")

        gem5_logger.info("Connected! Waiting for prompt...")

        # Create the lock file
        self.lock_file_name = lock_file_name
        open(self.lock_file_name, 'w').close() # Similar to touch
        gem5_logger.info("Created lock file {} to prevent reconnecting to "
                         "same simulation".format(self.lock_file_name))

        # We need to find the prompt. It might be different if we are resuming
        # from a checkpoint. Therefore, we test multiple options here.
        prompt_found = False
        while not prompt_found:
            try:
                self._login_to_device()
            except TIMEOUT:
                pass
            except EOF as err:
                self._gem5_EOF_handler(gem5_simulation, gem5_out_dir, err)

            try:
                # Try and force a prompt to be shown
                self.conn.send('\n')
                self.conn.expect([r'# ', r'\$ ', self.conn.UNIQUE_PROMPT, r'\[PEXPECT\][\\\$\#]+ '], timeout=60)
                prompt_found = True
            except TIMEOUT:
                pass
            except EOF as err:
                self._gem5_EOF_handler(gem5_simulation, gem5_out_dir, err)

        gem5_logger.info("Successfully logged in")
        gem5_logger.info("Setting unique prompt...")

        self.conn.set_unique_prompt()
        self.conn.prompt()
        gem5_logger.info("Prompt found and replaced with a unique string")

        # We check that the prompt is what we think it should be. If not, we
        # need to update the regex we use to match.
        self._find_prompt()

        self.conn.setecho(False)
        self._sync_gem5_shell()

        # Fully connected to gem5 simulation
        self.gem5_interact_dir = gem5_interact_dir
        self.gem5_out_dir = gem5_out_dir
        self.gem5simulation = gem5_simulation

        # Ready for interaction now
        self.ready = True

    def _login_to_device(self):
        """
        Login to device, will be overwritten if there is an actual login
        """
        pass

    def _find_prompt(self):
        prompt = r'\[PEXPECT\][\\\$\#]+ '
        synced = False
        while not synced:
            self.conn.send('\n')
            i = self.conn.expect([prompt, self.conn.UNIQUE_PROMPT, r'[\$\#] '], timeout=self.default_timeout)
            if i == 0:
                synced = True
            elif i == 1:
                prompt = self.conn.UNIQUE_PROMPT
                synced = True
            else:
                prompt = re.sub(r'\$', r'\\\$', self.conn.before.strip() + self.conn.after.strip())
                prompt = re.sub(r'\#', r'\\\#', prompt)
                prompt = re.sub(r'\[', r'\[', prompt)
                prompt = re.sub(r'\]', r'\]', prompt)

        self.conn.PROMPT = prompt

    def _sync_gem5_shell(self):
        """
        Synchronise with the gem5 shell.

        Write some unique text to the gem5 device to allow us to synchronise
        with the shell output. We actually get two prompts so we need to match
        both of these.
        """
        gem5_logger.debug("Sending Sync")
        self.conn.send("echo \*\*sync\*\*\n")
        self.conn.expect(r"\*\*sync\*\*", timeout=self.default_timeout)
        self.conn.expect([self.conn.UNIQUE_PROMPT, self.conn.PROMPT], timeout=self.default_timeout)
        self.conn.expect([self.conn.UNIQUE_PROMPT, self.conn.PROMPT], timeout=self.default_timeout)

    def _gem5_util(self, command):
        """ Execute a gem5 utility command using the m5 binary on the device """
        if self.m5_path is None:
            raise TargetStableError('Path to m5 binary on simulated system  is not set!')
        self._gem5_shell('{} {}'.format(self.m5_path, command))

    def _gem5_shell(self, command, as_root=False, timeout=None, check_exit_code=True, sync=True):  # pylint: disable=R0912
        """
        Execute a command in the gem5 shell

        This wraps the telnet connection to gem5 and processes the raw output.

        This method waits for the shell to return, and then will try and
        separate the output from the command from the command itself. If this
        fails, warn, but continue with the potentially wrong output.

        The exit code is also checked by default, and non-zero exit codes will
        raise a TargetStableError.
        """
        if sync:
            self._sync_gem5_shell()

        gem5_logger.debug("gem5_shell command: {}".format(command))

        if as_root:
            command = 'echo {} | su'.format(quote(command))

        # Send the actual command
        self.conn.send("{}\n".format(command))

        # Wait for the response. We just sit here and wait for the prompt to
        # appear, as gem5 might take a long time to provide the output. This
        # avoids timeout issues.
        command_index = -1
        while command_index == -1:
            if self.conn.prompt():
                output = re.sub(r' \r([^\n])', r'\1', self.conn.before)
                output = re.sub(r'[\b]', r'', output)
                # Deal with line wrapping
                output = re.sub(r'[\r].+?<', r'', output)
                command_index = output.find(command)

                # If we have -1, then we cannot match the command, but the
                # prompt has returned. Hence, we have a bit of an issue. We
                # warn, and return the whole output.
                if command_index == -1:
                    gem5_logger.warning("gem5_shell: Unable to match command in "
                                     "command output. Expect parsing errors!")
                    command_index = 0

        output = output[command_index + len(command):].strip()

        # If the gem5 system echoes the executed command, we need to remove that too!
        if self.strip_echoed_commands:
            command_index = output.find(command)
            if command_index != -1:
                output = output[command_index + len(command):].strip()

        gem5_logger.debug("gem5_shell output: {}".format(output))

        # We get a second prompt. Hence, we need to eat one to make sure that we
        # stay in sync. If we do not do this, we risk getting out of sync for
        # slower simulations.
        self.conn.expect([self.conn.UNIQUE_PROMPT, self.conn.PROMPT], timeout=self.default_timeout)

        if check_exit_code:
            exit_code_text = self._gem5_shell('echo $?', as_root=as_root,
                                             timeout=timeout, check_exit_code=False,
                                             sync=False)
            try:
                exit_code = int(exit_code_text.split()[0])
                if exit_code:
                    message = 'Got exit code {}\nfrom: {}\nOUTPUT: {}'
                    raise TargetStableError(message.format(exit_code, command, output))
            except (ValueError, IndexError):
                gem5_logger.warning('Could not get exit code for "{}",\ngot: "{}"'.format(command, exit_code_text))

        return output

    def _mount_virtio(self):
        """
        Mount the VirtIO device in the simulated system.
        """
        gem5_logger.info("Mounting VirtIO device in simulated system")

        self._gem5_shell('mkdir -p {}'.format(self.gem5_input_dir), as_root=True)
        mount_command = "mount -t 9p -o trans=virtio,version=9p2000.L,aname={} gem5 {}".format(self.gem5_interact_dir, self.gem5_input_dir)
        self._gem5_shell(mount_command, as_root=True)

    def _unmount_virtio(self):
        """
        Unmount the VirtIO device in the simulated system.
        """
        gem5_logger.info("Unmounting VirtIO device in simulated system")

        unmount_command = "umount {}".format(self.gem5_input_dir)
        self._gem5_shell(unmount_command, as_root=True)

    def take_checkpoint(self):
        """
        Take a checkpoint of the simulated system.

        In order to take a checkpoint we first unmount the virtio
        device, take then checkpoint, and then remount the device to
        allow us to continue the current run. This needs to be done to
        ensure that future gem5 simulations are able to utilise the
        virtio device (i.e., we need to drop the current state
        information that the device has).
        """
        self._unmount_virtio()
        self._gem5_util("checkpoint")
        self._mount_virtio()

    def _move_to_temp_dir(self, source):
        """
        Move a file to the temporary directory on the host for copying to the
        gem5 device
        """
        command = "cp {} {}".format(source, self.gem5_interact_dir)
        gem5_logger.debug("Local copy command: {}".format(command))
        subprocess.call(command.split())
        subprocess.call("sync".split())

    def _check_ready(self):
        """
        Check if the gem5 platform is ready
        """
        if not self.ready:
            raise TargetTransientError('Gem5 is not ready to interact yet')

    def _wait_for_boot(self):
        pass

    def _probe_file(self, filepath):
        """
        Internal method to check if the target has a certain file
        """
        filepath = quote(filepath)
        command = 'if [ -e {} ]; then echo 1; else echo 0; fi'
        output = self.execute(command.format(filepath), as_root=self.is_rooted)
        return boolean(output.strip())


class LinuxGem5Connection(Gem5Connection):

    def _login_to_device(self):
        gem5_logger.info("Trying to log in to gem5 device")
        login_prompt = ['login:', 'AEL login:', 'username:', 'aarch64-gem5 login:']
        login_password_prompt = ['password:']
        # Wait for the login prompt
        prompt = login_prompt + [self.conn.UNIQUE_PROMPT]
        i = self.conn.expect(prompt, timeout=10)
        # Check if we are already at a prompt, or if we need to log in.
        if i < len(prompt) - 1:
            self.conn.sendline("{}".format(self.username))
            password_prompt = login_password_prompt + [r'# ', self.conn.UNIQUE_PROMPT]
            j = self.conn.expect(password_prompt, timeout=self.default_timeout)
            if j < len(password_prompt) - 2:
                self.conn.sendline("{}".format(self.password))
                self.conn.expect([r'# ', self.conn.UNIQUE_PROMPT], timeout=self.default_timeout)



class AndroidGem5Connection(Gem5Connection):

    def _wait_for_boot(self):
        """
        Wait for the system to boot

        We monitor the sys.boot_completed and service.bootanim.exit system
        properties to determine when the system has finished booting. In the
        event that we cannot coerce the result of service.bootanim.exit to an
        integer, we assume that the boot animation was disabled and do not wait
        for it to finish.

        """
        gem5_logger.info("Waiting for Android to boot...")
        while True:
            booted = False
            anim_finished = True  # Assume boot animation was disabled on except
            try:
                booted = (int('0' + self._gem5_shell('getprop sys.boot_completed', check_exit_code=False).strip()) == 1)
                anim_finished = (int(self._gem5_shell('getprop service.bootanim.exit', check_exit_code=False).strip()) == 1)
            except ValueError:
                pass
            if booted and anim_finished:
                break
            time.sleep(60)

        gem5_logger.info("Android booted")

def _give_password(password, command):
    if not sshpass:
        raise HostError('Must have sshpass installed on the host in order to use password-based auth.')
    pass_template = "sshpass -p {} "
    pass_string = pass_template.format(quote(password))
    redacted_string = pass_template.format(quote('<redacted>'))
    return (pass_string + command, redacted_string + command)


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
