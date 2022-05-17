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


import glob
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
import contextlib
import weakref
import select
import copy
import functools
from pipes import quote
from future.utils import raise_from

from paramiko.client import SSHClient, AutoAddPolicy, RejectPolicy
import paramiko.ssh_exception
from scp import SCPClient
# By default paramiko is very verbose, including at the INFO level
logging.getLogger("paramiko").setLevel(logging.WARNING)

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
                              TimeoutError, TargetTransientError,
                              TargetCalledProcessError,
                              TargetTransientCalledProcessError,
                              TargetStableCalledProcessError)
from devlib.utils.misc import (which, strip_bash_colors, check_output,
                               sanitize_cmd_template, memoized, redirect_streams)
from devlib.utils.types import boolean
from devlib.connection import (ConnectionBase, ParamikoBackgroundCommand, PopenBackgroundCommand,
                               SSHTransferManager)


DEFAULT_SSH_SUDO_COMMAND = "sudo -k -p ' ' -S -- sh -c {}"


ssh = None
scp = None
sshpass = None


logger = logging.getLogger('ssh')
gem5_logger = logging.getLogger('gem5-connection')


@contextlib.contextmanager
def _handle_paramiko_exceptions(command=None):
    try:
        yield
    except paramiko.ssh_exception.NoValidConnectionsError as e:
        raise TargetNotRespondingError('Connection lost: {}'.format(e))
    except paramiko.ssh_exception.AuthenticationException as e:
        raise TargetStableError('Could not authenticate: {}'.format(e))
    except paramiko.ssh_exception.BadAuthenticationType as e:
        raise TargetStableError('Bad authentication type: {}'.format(e))
    except paramiko.ssh_exception.BadHostKeyException as e:
        raise TargetStableError('Bad host key: {}'.format(e))
    except paramiko.ssh_exception.ChannelException as e:
        raise TargetStableError('Could not open an SSH channel: {}'.format(e))
    except paramiko.ssh_exception.PasswordRequiredException as e:
        raise TargetStableError('Please unlock the private key file: {}'.format(e))
    except paramiko.ssh_exception.ProxyCommandFailure as e:
        raise TargetStableError('Proxy command failure: {}'.format(e))
    except paramiko.ssh_exception.SSHException as e:
        raise TargetTransientError('SSH logic error: {}'.format(e))
    except socket.timeout:
        raise TimeoutError(command, output=None)


def _read_paramiko_streams(stdout, stderr, select_timeout, callback, init, chunk_size=int(1e42)):
    try:
        return _read_paramiko_streams_internal(stdout, stderr, select_timeout, callback, init, chunk_size)
    finally:
        # Close the channel to make sure the remove process will receive
        # SIGPIPE when writing on its streams. That could happen if the
        # user closed the out_streams but the remote process has not
        # finished yet.
        assert stdout.channel is stderr.channel
        stdout.channel.close()


def _read_paramiko_streams_internal(stdout, stderr, select_timeout, callback, init, chunk_size):
    channel = stdout.channel
    assert stdout.channel is stderr.channel

    def read_channel(callback_state):
        read_list, _, _ = select.select([channel], [], [], select_timeout)
        for desc in read_list:
            for ready, recv, name in (
                (desc.recv_ready(), desc.recv, 'stdout'),
                (desc.recv_stderr_ready(), desc.recv_stderr, 'stderr')
            ):
                if ready:
                    chunk = recv(chunk_size)
                    if chunk:
                        try:
                            callback_state = callback(callback_state, name, chunk)
                        except Exception as e:
                            return (e, callback_state)

        return (None, callback_state)

    def read_all_channel(callback=None, callback_state=None):
        for stream, name in ((stdout, 'stdout'), (stderr, 'stderr')):
            try:
                chunk = stream.read()
            except Exception:
                continue

            if callback is not None and chunk:
                callback_state = callback(callback_state, name, chunk)

        return callback_state

    callback_excep = None
    try:
        callback_state = init
        while not channel.exit_status_ready():
            callback_excep, callback_state = read_channel(callback_state)
            if callback_excep is not None:
                raise callback_excep
    # Make sure to always empty the streams to unblock the remote process on
    # the way to exit, in case something bad happened. For example, the
    # callback could raise an exception to signal it does not want to do
    # anything anymore, or only reading from one of the stream might have
    # raised an exception, leaving the other one non-empty.
    except Exception as e:
        if callback_excep is None:
            # Only call the callback if there was no exception originally, as
            # we don't want to reenter it if it raised an exception
            read_all_channel(callback, callback_state)
        raise e
    else:
        # Finish emptying the buffers
        callback_state = read_all_channel(callback, callback_state)
        exit_code = channel.recv_exit_status()
        return (callback_state, exit_code)


def telnet_get_shell(host,
                  username,
                  password=None,
                  port=None,
                  timeout=10,
                  original_prompt=None):
    _check_env()
    start_time = time.time()
    while True:
        conn = TelnetPxssh(original_prompt=original_prompt)

        try:
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


class SshConnectionBase(ConnectionBase):
    """
    Base class for SSH connections.
    """

    default_timeout = 10

    @property
    def name(self):
        return self.host

    @property
    def connected_as_root(self):
        if self._connected_as_root is None:
            try:
                result = self.execute('id', as_root=False)
            except TargetStableError:
                is_root = False
            else:
                is_root = 'uid=0(' in result
            self._connected_as_root = is_root
        return self._connected_as_root

    @connected_as_root.setter
    def connected_as_root(self, state):
        self._connected_as_root = state

    def __init__(self,
                 host,
                 username,
                 password=None,
                 keyfile=None,
                 port=None,
                 platform=None,
                 sudo_cmd=DEFAULT_SSH_SUDO_COMMAND,
                 strict_host_check=True,
                 ):
        super().__init__()
        self._connected_as_root = None
        self.host = host
        self.username = username
        self.password = password
        self.keyfile = check_keyfile(keyfile) if keyfile else keyfile
        self.port = port
        self.sudo_cmd = sanitize_cmd_template(sudo_cmd)
        self.platform = platform
        self.strict_host_check = strict_host_check
        logger.debug('Logging in {}@{}'.format(username, host))


class SshConnection(SshConnectionBase):
    # pylint: disable=unused-argument,super-init-not-called
    def __init__(self,
                 host,
                 username,
                 password=None,
                 keyfile=None,
                 port=22,
                 timeout=None,
                 platform=None,
                 sudo_cmd=DEFAULT_SSH_SUDO_COMMAND,
                 strict_host_check=True,
                 use_scp=False,
                 poll_transfers=False,
                 start_transfer_poll_delay=30,
                 total_transfer_timeout=3600,
                 transfer_poll_period=30,
                 ):

        super().__init__(
            host=host,
            username=username,
            password=password,
            keyfile=keyfile,
            port=port,
            platform=platform,
            sudo_cmd=sudo_cmd,
            strict_host_check=strict_host_check,
        )
        self.timeout = timeout if timeout is not None else self.default_timeout

        # Allow using scp for file transfer if sftp is not supported
        self.use_scp = use_scp
        self.poll_transfers=poll_transfers
        if poll_transfers:
            transfer_opts = {'start_transfer_poll_delay': start_transfer_poll_delay,
                            'total_timeout': total_transfer_timeout,
                            'poll_period': transfer_poll_period,
                            }

        if self.use_scp:
            logger.debug('Using SCP for file transfer')
        else:
            logger.debug('Using SFTP for file transfer')

        self.transfer_mgr = SSHTransferManager(self, **transfer_opts) if poll_transfers else None
        self.client = self._make_client()
        atexit.register(self.close)

        # Use a marker in the output so that we will be able to differentiate
        # target connection issues with "password needed".
        # Also, sudo might not be installed at all on the target (but
        # everything will work as long as we login as root). If sudo is still
        # needed, it will explode when someone tries to use it. After all, the
        # user might not be interested in being root at all.
        self._sudo_needs_password = (
            'NEED_PASSWORD' in
            self.execute(
                # sudo -n is broken on some versions on MacOSX, revisit that if
                # someone ever cares
                'sudo -n true || echo NEED_PASSWORD',
                as_root=False,
                check_exit_code=False,
            )
        )

    def _make_client(self):
        if self.strict_host_check:
            policy = RejectPolicy
        else:
            policy = AutoAddPolicy
        # Only try using SSH keys if we're not using a password
        check_ssh_keys = self.password is None

        with _handle_paramiko_exceptions():
            client = SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(policy)
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                key_filename=self.keyfile,
                timeout=self.timeout,
                look_for_keys=check_ssh_keys,
                allow_agent=check_ssh_keys
            )

            return client

    def _make_channel(self):
        with _handle_paramiko_exceptions():
            transport = self.client.get_transport()
            channel = transport.open_session()
            return channel

    def _get_progress_cb(self):
        return self.transfer_mgr.progress_cb if self.transfer_mgr is not None else None

    # Limit the number of opened channels to a low number, since some servers
    # will reject more connections request. For OpenSSH, this is controlled by
    # the MaxSessions config.
    @functools.lru_cache(maxsize=1)
    def _cached_get_sftp(self):
        try:
            sftp = self.client.open_sftp()
        except paramiko.ssh_exception.SSHException as e:
            if 'EOF during negotiation' in str(e):
                raise TargetStableError('The SSH server does not support SFTP. Please install and enable appropriate module.') from e
            else:
                raise
        return sftp

    def _get_sftp(self, timeout):
        sftp = self._cached_get_sftp()
        sftp.get_channel().settimeout(timeout)
        return sftp

    @functools.lru_cache()
    def _get_scp(self, timeout):
        return SCPClient(self.client.get_transport(), socket_timeout=timeout, progress=self._get_progress_cb())

    def _push_file(self, sftp, src, dst):
        sftp.put(src, dst, callback=self._get_progress_cb())

    @classmethod
    def _path_exists(cls, sftp, path):
        try:
            sftp.lstat(path)
        except FileNotFoundError:
            return False
        else:
            return True

    def _push_folder(self, sftp, src, dst):
        sftp.mkdir(dst)
        for entry in os.scandir(src):
            name = entry.name
            src_path = os.path.join(src, name)
            dst_path = os.path.join(dst, name)
            if entry.is_dir():
                push = self._push_folder
            else:
                push = self._push_file

            push(sftp, src_path, dst_path)

    def _push_path(self, sftp, src, dst):
        logger.debug('Pushing via sftp: {} -> {}'.format(src, dst))
        push = self._push_folder if os.path.isdir(src) else self._push_file
        push(sftp, src, dst)

    def _pull_file(self, sftp, src, dst):
        sftp.get(src, dst, callback=self._get_progress_cb())

    def _pull_folder(self, sftp, src, dst):
        os.makedirs(dst)
        for fileattr in sftp.listdir_attr(src):
            filename = fileattr.filename
            src_path = os.path.join(src, filename)
            dst_path = os.path.join(dst, filename)
            if stat.S_ISDIR(fileattr.st_mode):
                pull = self._pull_folder
            else:
                pull = self._pull_file

            pull(sftp, src_path, dst_path)

    def _pull_path(self, sftp, src, dst):
        logger.debug('Pulling via sftp: {} -> {}'.format(src, dst))
        try:
            self._pull_file(sftp, src, dst)
        except IOError:
            # Maybe that was a directory, so retry as such
            self._pull_folder(sftp, src, dst)

    def push(self, sources, dest, timeout=None):
        self._push_pull('push', sources, dest, timeout)

    def pull(self, sources, dest, timeout=None):
        self._push_pull('pull', sources, dest, timeout)

    def _push_pull(self, action, sources, dest, timeout):
        if action not in ['push', 'pull']:
            raise ValueError("Action must be either `push` or `pull`")

        # If timeout is set, or told not to poll
        if timeout is not None or not self.poll_transfers:
            if self.use_scp:
                scp = self._get_scp(timeout)
                scp_cmd = getattr(scp, 'put' if action == 'push' else 'get')
                scp_msg = '{}ing via scp: {} -> {}'.format(action, sources, dest)
                logger.debug(scp_msg.capitalize())
                scp_cmd(sources, dest, recursive=True)
            else:
                sftp = self._get_sftp(timeout)
                sftp_cmd = getattr(self, '_' + action + '_path')
                with _handle_paramiko_exceptions():
                    for source in sources:
                        sftp_cmd(sftp, source, dest)

        # No timeout, and polling is set
        elif self.use_scp:
            scp = self._get_scp(timeout)
            scp_cmd = getattr(scp, 'put' if action == 'push' else 'get')
            with _handle_paramiko_exceptions(), self.transfer_mgr.manage(sources, dest, action, scp):
                scp_msg = '{}ing via scp: {} -> {}'.format(action, sources, dest)
                logger.debug(scp_msg.capitalize())
                scp_cmd(sources, dest, recursive=True)
        else:
            sftp = self._get_sftp(timeout)
            sftp_cmd = getattr(self, '_' + action + '_path')
            with _handle_paramiko_exceptions(), self.transfer_mgr.manage(sources, dest, action, sftp):
                for source in sources:
                    sftp_cmd(sftp, source, dest)

    def execute(self, command, timeout=None, check_exit_code=True,
                as_root=False, strip_colors=True, will_succeed=False): #pylint: disable=unused-argument
        if command == '':
            return ''
        try:
            with _handle_paramiko_exceptions(command):
                exit_code, output = self._execute(command, timeout, as_root, strip_colors)
        except TargetCalledProcessError:
            raise
        except TargetStableError as e:
            if will_succeed:
                raise TargetTransientError(e)
            else:
                raise
        else:
            if check_exit_code and exit_code:
                cls = TargetTransientCalledProcessError if will_succeed else TargetStableCalledProcessError
                raise cls(
                    exit_code,
                    command,
                    output,
                    None,
                )
            return output

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        with _handle_paramiko_exceptions(command):
            bg_cmd = self._background(command, stdout, stderr, as_root)

        self._current_bg_cmds.add(bg_cmd)
        return bg_cmd

    def _background(self, command, stdout, stderr, as_root):
        orig_command = command
        stdout, stderr, command = redirect_streams(stdout, stderr, command)

        command = "printf '%s\n' $$; exec sh -c {}".format(quote(command))
        channel = self._make_channel()

        def executor(cmd, timeout):
            channel.exec_command(cmd)
            # Read are not buffered so we will always get the data as soon as
            # they arrive
            return (
                channel.makefile_stdin('w', 0),
                channel.makefile(),
                channel.makefile_stderr(),
            )

        stdin, stdout_in, stderr_in = self._execute_command(
            command,
            as_root=as_root,
            log=False,
            timeout=None,
            executor=executor,
        )
        pid = stdout_in.readline()
        if not pid:
            stderr = stderr_in.read()
            if channel.exit_status_ready():
                ret = channel.recv_exit_status()
            else:
                ret = 126
            raise subprocess.CalledProcessError(
                ret,
                command,
                b'',
                stderr,
            )
        pid = int(pid)

        def create_out_stream(stream_in, stream_out):
            """
            Create a pair of file-like objects. The first one is used to read
            data and the second one to write.
            """

            if stream_out == subprocess.DEVNULL:
                r, w = None, None
            # When asked for a pipe, we just give the file-like object as the
            # reading end and no writing end, since paramiko already writes to
            # it
            elif stream_out == subprocess.PIPE:
                r, w = os.pipe()
                r = os.fdopen(r, 'rb')
                w = os.fdopen(w, 'wb')
            # Turn a file descriptor into a file-like object
            elif isinstance(stream_out, int) and stream_out >= 0:
                r = os.fdopen(stream_in, 'rb')
                w = os.fdopen(stream_out, 'wb')
            # file-like object
            else:
                r = stream_in
                w = stream_out

            return (r, w)

        out_streams = {
            name: create_out_stream(stream_in, stream_out)
            for stream_in, stream_out, name in (
                (stdout_in, stdout, 'stdout'),
                (stderr_in, stderr, 'stderr'),
            )
        }

        def redirect_thread_f(stdout_in, stderr_in, out_streams, select_timeout):
            def callback(out_streams, name, chunk):
                try:
                    r, w = out_streams[name]
                except KeyError:
                    return out_streams

                try:
                    w.write(chunk)
                # Write failed
                except ValueError:
                    # Since that stream is now closed, stop trying to write to it
                    del out_streams[name]
                    # If that was the last open stream, we raise an
                    # exception so the thread can terminate.
                    if not out_streams:
                        raise

                return out_streams

            try:
                _read_paramiko_streams(stdout_in, stderr_in, select_timeout, callback, copy.copy(out_streams))
            # The streams closed while we were writing to it, the job is done here
            except ValueError:
                pass

            # Make sure the writing end are closed proper since we are not
            # going to write anything anymore
            for r, w in out_streams.values():
                w.flush()
                if r is not w and w is not None:
                    w.close()

        # If there is anything we need to redirect to, spawn a thread taking
        # care of that
        select_timeout = 1
        thread_out_streams = {
            name: (r, w)
            for name, (r, w) in out_streams.items()
            if w is not None
        }
        redirect_thread = threading.Thread(
            target=redirect_thread_f,
            args=(stdout_in, stderr_in, thread_out_streams, select_timeout),
            # The thread will die when the main thread dies
            daemon=True,
        )
        redirect_thread.start()

        return ParamikoBackgroundCommand(
            conn=self,
            as_root=as_root,
            chan=channel,
            pid=pid,
            stdin=stdin,
            # We give the reading end to the consumer of the data
            stdout=out_streams['stdout'][0],
            stderr=out_streams['stderr'][0],
            redirect_thread=redirect_thread,
            cmd=orig_command,
        )

    def _close(self):
        logger.debug('Logging out {}@{}'.format(self.username, self.host))
        with _handle_paramiko_exceptions():
            bg_cmds = set(self._current_bg_cmds)
            for bg_cmd in bg_cmds:
                bg_cmd.close()
            self.client.close()

    def _execute_command(self, command, as_root, log, timeout, executor):
        # As we're already root, there is no need to use sudo.
        log_debug = logger.debug if log else lambda msg: None
        use_sudo = as_root and not self.connected_as_root

        if use_sudo:
            if self._sudo_needs_password and not self.password:
                raise TargetStableError('Attempt to use sudo but no password was specified')

            command = self.sudo_cmd.format(quote(command))

            log_debug(command)
            streams = executor(command, timeout=timeout)
            if self._sudo_needs_password:
                stdin = streams[0]
                stdin.write(self.password + '\n')
                stdin.flush()
        else:
            log_debug(command)
            streams = executor(command, timeout=timeout)

        return streams

    def _execute(self, command, timeout=None, as_root=False, strip_colors=True, log=True):
        # Merge stderr into stdout since we are going without a TTY
        command = '({}) 2>&1'.format(command)

        stdin, stdout, stderr = self._execute_command(
            command,
            as_root=as_root,
            log=log,
            timeout=timeout,
            executor=self.client.exec_command,
        )
        stdin.close()

        # Empty the stdout buffer of the command, allowing it to carry on to
        # completion
        def callback(output_chunks, name, chunk):
            output_chunks.append(chunk)
            return output_chunks

        select_timeout = 1
        output_chunks, exit_code = _read_paramiko_streams(stdout, stderr, select_timeout, callback, [])
        # Join in one go to avoid O(N^2) concatenation
        output = b''.join(output_chunks)

        if sys.version_info[0] == 3:
            output = output.decode(sys.stdout.encoding or 'utf-8', 'replace')
        if strip_colors:
            output = strip_bash_colors(output)

        return (exit_code, output)


class TelnetConnection(SshConnectionBase):

    default_password_prompt = '[sudo] password'
    max_cancel_attempts = 5

    # pylint: disable=unused-argument,super-init-not-called
    def __init__(self,
                 host,
                 username,
                 password=None,
                 port=None,
                 timeout=None,
                 password_prompt=None,
                 original_prompt=None,
                 sudo_cmd="sudo -- sh -c {}",
                 strict_host_check=True,
                 platform=None):

        super().__init__(
            host=host,
            username=username,
            password=password,
            keyfile=None,
            port=port,
            platform=platform,
            sudo_cmd=sudo_cmd,
            strict_host_check=strict_host_check,
        )

        _check_env()
        self.options = self._get_default_options()

        self.lock = threading.Lock()
        self.password_prompt = password_prompt if password_prompt is not None else self.default_password_prompt
        logger.debug('Logging in {}@{}'.format(username, host))
        timeout = timeout if timeout is not None else self.default_timeout

        self.conn = telnet_get_shell(host, username, password, port, timeout, original_prompt)
        atexit.register(self.close)

    def fmt_remote_path(self, path):
        return '{}@{}:{}'.format(self.username, self.host, path)

    def _get_default_options(self):
        if self.strict_host_check:
            options = {
                'StrictHostKeyChecking': 'yes',
            }
        else:
            options = {
                'StrictHostKeyChecking': 'no',
                'UserKnownHostsFile': '/dev/null',
            }
        return options

    def push(self, sources, dest, timeout=30):
        # Quote the destination as SCP would apply globbing too
        dest = self.fmt_remote_path(quote(dest))
        paths = list(sources) + [dest]
        return self._scp(paths, timeout)

    def pull(self, sources, dest, timeout=30):
        # First level of escaping for the remote shell
        sources = ' '.join(map(quote, sources))
        # All the sources are merged into one scp parameter
        sources = self.fmt_remote_path(sources)
        paths = [sources, dest]
        self._scp(paths, timeout)

    def _scp(self, paths, timeout=30):
        # NOTE: the version of scp in Ubuntu 12.04 occasionally (and bizarrely)
        # fails to connect to a device if port is explicitly specified using -P
        # option, even if it is the default port, 22. To minimize this problem,
        # only specify -P for scp if the port is *not* the default.
        port_string = '-P {}'.format(quote(str(self.port))) if (self.port and self.port != 22) else ''
        keyfile_string = '-i {}'.format(quote(self.keyfile)) if self.keyfile else ''
        options = " ".join(["-o {}={}".format(key, val)
                            for key, val in self.options.items()])
        paths = ' '.join(map(quote, paths))
        command = '{} {} -r {} {} {}'.format(scp,
                                                options,
                                                keyfile_string,
                                                port_string,
                                                paths)
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
                    except (ValueError, IndexError):
                        raise ValueError(
                            'Could not get exit code for "{}",\ngot: "{}"'\
                            .format(command, exit_code_text))
                    if exit_code:
                        cls = TargetTransientCalledProcessError if will_succeed else TargetStableCalledProcessError
                        raise cls(
                            exit_code,
                            command,
                            output,
                            None,
                        )
                return output
        except EOF:
            raise TargetNotRespondingError('Connection lost.')
        except TargetCalledProcessError:
            raise
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

    def _close(self):
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

    def wait_for_device(self, timeout=30):
        return

    def reboot_bootloader(self, timeout=30):
        raise NotImplementedError()

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

    def push(self, sources, dest, timeout=None):
        """
        Push a file to the gem5 device using VirtIO

        The file to push to the device is copied to the temporary directory on
        the host, before being copied within the simulation to the destination.
        Checks, in the form of 'ls' with error code checking, are performed to
        ensure that the file is copied to the destination.
        """
        # First check if the connection is set up to interact with gem5
        self._check_ready()

        for source in sources:
            filename = os.path.basename(source)
            logger.debug("Pushing {} to device.".format(source))
            logger.debug("gem5interactdir: {}".format(self.gem5_interact_dir))
            logger.debug("dest: {}".format(dest))
            logger.debug("filename: {}".format(filename))

            # We need to copy the file to copy to the temporary directory
            self._move_to_temp_dir(source)

            # Back to the gem5 world
            filename = quote(self.gem5_input_dir + filename)
            self._gem5_shell("ls -al {}".format(filename))
            self._gem5_shell("cat {} > {}".format(filename, quote(dest)))
            self._gem5_shell("sync")
            self._gem5_shell("ls -al {}".format(quote(dest)))
            self._gem5_shell("ls -al {}".format(quote(self.gem5_input_dir)))
            logger.debug("Push complete.")

    def pull(self, sources, dest, timeout=0): #pylint: disable=unused-argument
        """
        Pull a file from the gem5 device using m5 writefile

        The file is copied to the local directory within the guest as the m5
        writefile command assumes that the file is local. The file is then
        written out to the host system using writefile, prior to being moved to
        the destination on the host.
        """
        # First check if the connection is set up to interact with gem5
        self._check_ready()

        for source in sources:
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
                                      as_root=as_root,
                                      will_succeed=will_succeed)
        except TargetCalledProcessError:
            raise
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

    def _close(self):
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

    def wait_for_device(self, timeout=30):
        """
        Wait for Gem5 to be ready for interation with a timeout.
        """
        for _ in attempts(timeout):
            if self.ready:
                return
            time.sleep(1)
        raise TimeoutError('Gem5 is not ready for interaction')

    def reboot_bootloader(self, timeout=30):
        raise NotImplementedError()

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
        self.conn.send("echo \\*\\*sync\\*\\*\n")
        self.conn.expect(r"\*\*sync\*\*", timeout=self.default_timeout)
        self.conn.expect([self.conn.UNIQUE_PROMPT, self.conn.PROMPT], timeout=self.default_timeout)
        self.conn.expect([self.conn.UNIQUE_PROMPT, self.conn.PROMPT], timeout=self.default_timeout)

    def _gem5_util(self, command):
        """ Execute a gem5 utility command using the m5 binary on the device """
        if self.m5_path is None:
            raise TargetStableError('Path to m5 binary on simulated system  is not set!')
        self._gem5_shell('{} {}'.format(self.m5_path, command))

    def _gem5_shell(self, command, as_root=False, timeout=None, check_exit_code=True, sync=True, will_succeed=False):  # pylint: disable=R0912
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
            except (ValueError, IndexError):
                raise ValueError('Could not get exit code for "{}",\ngot: "{}"'.format(command, exit_code_text))
            else:
                if exit_code:
                    cls = TragetTransientCalledProcessError if will_succeed else TargetStableCalledProcessError
                    raise cls(
                        exit_code,
                        command,
                        output,
                        None,
                    )

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
