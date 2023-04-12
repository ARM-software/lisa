#    Copyright 2019 ARM Limited
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

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from weakref import WeakSet
from shlex import quote
from time import monotonic
import os
import signal
import socket
import subprocess
import threading
import time
import logging
import select
import fcntl

from devlib.utils.misc import InitCheckpoint

_KILL_TIMEOUT = 3


def _kill_pgid_cmd(pgid, sig, busybox):
    return '{} kill -{} -- -{}'.format(busybox, sig.value, pgid)

def _popen_communicate(bg, popen, input, timeout):
    try:
        stdout, stderr = popen.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        bg.cancel()
        raise

    ret = popen.returncode
    if ret:
        raise subprocess.CalledProcessError(
            ret,
            popen.args,
            stdout,
            stderr,
        )
    else:
        return (stdout, stderr)


class ConnectionBase(InitCheckpoint):
    """
    Base class for all connections.
    """
    def __init__(self):
        self._current_bg_cmds = WeakSet()
        self._closed = False
        self._close_lock = threading.Lock()
        self.busybox = None

    def cancel_running_command(self):
        bg_cmds = set(self._current_bg_cmds)
        for bg_cmd in bg_cmds:
            bg_cmd.cancel()

    @abstractmethod
    def _close(self):
        """
        Close the connection.

        The public :meth:`close` method makes sure that :meth:`_close` will
        only be called once, and will serialize accesses to it if it happens to
        be called from multiple threads at once.
        """

    def close(self):
        # Locking the closing allows any thread to safely call close() as long
        # as the connection can be closed from a thread that is not the one it
        # started its life in.
        with self._close_lock:
            if not self._closed:
                self._close()
                self._closed = True

    # Ideally, that should not be relied upon but that will improve the chances
    # of the connection being properly cleaned up when it's not in use anymore.
    def __del__(self):
        # Since __del__ will be called if an exception is raised in __init__
        # (e.g. we cannot connect), we only run close() when we are sure
        # __init__ has completed successfully.
        if self.initialized:
            self.close()


class BackgroundCommand(ABC):
    """
    Allows managing a running background command using a subset of the
    :class:`subprocess.Popen` API.

    Instances of this class can be used as context managers, with the same
    semantic as :class:`subprocess.Popen`.
    """
    @abstractmethod
    def send_signal(self, sig):
        """
        Send a POSIX signal to the background command's process group ID
        (PGID).

        :param signal: Signal to send.
        :type signal: signal.Signals
        """

    def kill(self):
        """
        Send SIGKILL to the background command.
        """
        self.send_signal(signal.SIGKILL)

    def cancel(self, kill_timeout=_KILL_TIMEOUT):
        """
        Try to gracefully terminate the process by sending ``SIGTERM``, then
        waiting for ``kill_timeout`` to send ``SIGKILL``.
        """
        if self.poll() is None:
            self._cancel(kill_timeout=kill_timeout)

    @abstractmethod
    def _cancel(self, kill_timeout):
        """
        Method to override in subclasses to implement :meth:`cancel`.
        """
        pass

    @abstractmethod
    def wait(self):
        """
        Block until the background command completes, and return its exit code.
        """

    def communicate(self, input=b'', timeout=None):
        """
        Block until the background command completes while reading stdout and stderr.
        Return ``tuple(stdout, stderr)``. If the return code is non-zero,
        raises a :exc:`subprocess.CalledProcessError` exception.
        """
        try:
            return self._communicate(input=input, timeout=timeout)
        finally:
            self.close()

    @abstractmethod
    def _communicate(self, input, timeout):
        pass

    @abstractmethod
    def poll(self):
        """
        Return exit code if the command has exited, None otherwise.
        """

    @property
    @abstractmethod
    def stdin(self):
        """
        File-like object connected to the background's command stdin.
        """

    @property
    @abstractmethod
    def stdout(self):
        """
        File-like object connected to the background's command stdout.
        """

    @property
    @abstractmethod
    def stderr(self):
        """
        File-like object connected to the background's command stderr.
        """

    @property
    @abstractmethod
    def pid(self):
        """
        Process Group ID (PGID) of the background command.

        Since the command is usually wrapped in shell processes for IO
        redirections, sudo etc, the PID cannot be assumed to be the actual PID
        of the command passed by the user. It's is guaranteed to be a PGID
        instead, which means signals sent to it as such will target all
        subprocesses involved in executing that command.
        """

    @abstractmethod
    def close(self):
        """
        Close all opened streams and then wait for command completion.

        :returns: Exit code of the command.

        .. note:: If the command is writing to its stdout/stderr, it might be
            blocked on that and die when the streams are closed.
        """

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class PopenBackgroundCommand(BackgroundCommand):
    """
    :class:`subprocess.Popen`-based background command.
    """

    def __init__(self, popen):
        self.popen = popen

    def send_signal(self, sig):
        return os.killpg(self.popen.pid, sig)

    @property
    def stdin(self):
        return self.popen.stdin

    @property
    def stdout(self):
        return self.popen.stdout

    @property
    def stderr(self):
        return self.popen.stderr

    @property
    def pid(self):
        return self.popen.pid

    def wait(self):
        return self.popen.wait()

    def _communicate(self, input, timeout):
        return _popen_communicate(self, self.popen, input, timeout)

    def poll(self):
        return self.popen.poll()

    def _cancel(self, kill_timeout):
        popen = self.popen
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        try:
            popen.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)

    def close(self):
        self.popen.__exit__(None, None, None)
        return self.popen.returncode

    def __enter__(self):
        self.popen.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.popen.__exit__(*args, **kwargs)


class ParamikoBackgroundCommand(BackgroundCommand):
    """
    :mod:`paramiko`-based background command.
    """
    def __init__(self, conn, chan, pid, as_root, cmd, stdin, stdout, stderr, redirect_thread):
        self.chan = chan
        self.as_root = as_root
        self.conn = conn
        self._pid = pid
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self.redirect_thread = redirect_thread
        self.cmd = cmd

    def send_signal(self, sig):
        # If the command has already completed, we don't want to send a signal
        # to another process that might have gotten that PID in the meantime.
        if self.poll() is not None:
            return
        # Use -PGID to target a process group rather than just the process
        # itself
        cmd = _kill_pgid_cmd(self.pid, sig, self.conn.busybox)
        self.conn.execute(cmd, as_root=self.as_root)

    @property
    def pid(self):
        return self._pid

    def wait(self):
        status = self.chan.recv_exit_status()
        # Ensure that the redirection thread is finished copying the content
        # from paramiko to the pipe.
        self.redirect_thread.join()
        return status

    def _communicate(self, input, timeout):
        stdout = self._stdout
        stderr = self._stderr
        stdin = self._stdin
        chan = self.chan

        # For some reason, file descriptors in the read-list of select() can
        # still end up blocking in .read(), so make the non-blocking to avoid a
        # deadlock. Since _communicate() will consume all input and all output
        # until the command dies, we can do whatever we want with the pipe
        # without affecting external users.
        for s in (stdout, stderr):
            fcntl.fcntl(s.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

        out = {stdout: [], stderr: []}
        ret = None
        can_send = True

        select_timeout = 1
        if timeout is not None:
            select_timeout = min(select_timeout, 1)

        def create_out():
            return (
                b''.join(out[stdout]),
                b''.join(out[stderr])
            )

        start = monotonic()

        while ret is None:
            # Even if ret is not None anymore, we need to drain the streams
            ret = self.poll()

            if timeout is not None and ret is None and monotonic() - start >= timeout:
                self.cancel()
                _stdout, _stderr = create_out()
                raise subprocess.TimeoutExpired(self.cmd, timeout, _stdout, _stderr)

            can_send &= (not chan.closed) & bool(input)
            wlist = [chan] if can_send else []

            if can_send and chan.send_ready():
                try:
                    n = chan.send(input)
                # stdin might have been closed already
                except OSError:
                    can_send = False
                    chan.shutdown_write()
                else:
                    input = input[n:]
                    if not input:
                        # Send EOF on stdin
                        chan.shutdown_write()

            rs, ws, _ = select.select(
                [x for x in (stdout, stderr) if not x.closed],
                wlist,
                [],
                select_timeout,
            )

            for r in rs:
                chunk = r.read()
                if chunk:
                    out[r].append(chunk)

        _stdout, _stderr = create_out()

        if ret:
            raise subprocess.CalledProcessError(
                ret,
                self.cmd,
                _stdout,
                _stderr,
            )
        else:
            return (_stdout, _stderr)

    def poll(self):
        # Wait for the redirection thread to finish, otherwise we would
        # indicate the caller that the command is finished and that the streams
        # are safe to drain, but actually the redirection thread is not
        # finished yet, which would end up in lost data.
        if self.redirect_thread.is_alive():
            return None
        elif self.chan.exit_status_ready():
            return self.wait()
        else:
            return None

    def _cancel(self, kill_timeout):
        self.send_signal(signal.SIGTERM)
        # Check if the command terminated quickly
        time.sleep(10e-3)
        # Otherwise wait for the full timeout and kill it
        if self.poll() is None:
            time.sleep(kill_timeout)
            self.send_signal(signal.SIGKILL)
            self.wait()

    @property
    def stdin(self):
        return self._stdin

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr

    def close(self):
        for x in (self.stdin, self.stdout, self.stderr):
            if x is not None:
                x.close()

        exit_code = self.wait()
        thread = self.redirect_thread
        if thread:
            thread.join()

        return exit_code


class AdbBackgroundCommand(BackgroundCommand):
    """
    ``adb``-based background command.
    """

    def __init__(self, conn, adb_popen, pid, as_root):
        self.conn = conn
        self.as_root = as_root
        self.adb_popen = adb_popen
        self._pid = pid

    def send_signal(self, sig):
        self.conn.execute(
            _kill_pgid_cmd(self.pid, sig, self.conn.busybox),
            as_root=self.as_root,
        )

    @property
    def stdin(self):
        return self.adb_popen.stdin

    @property
    def stdout(self):
        return self.adb_popen.stdout

    @property
    def stderr(self):
        return self.adb_popen.stderr

    @property
    def pid(self):
        return self._pid

    def wait(self):
        return self.adb_popen.wait()

    def _communicate(self, input, timeout):
        return _popen_communicate(self, self.adb_popen, input, timeout)


    def poll(self):
        return self.adb_popen.poll()

    def _cancel(self, kill_timeout):
        self.send_signal(signal.SIGTERM)
        try:
            self.adb_popen.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            self.send_signal(signal.SIGKILL)
            self.adb_popen.kill()

    def close(self):
        self.adb_popen.__exit__(None, None, None)
        return self.adb_popen.returncode

    def __enter__(self):
        self.adb_popen.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.adb_popen.__exit__(*args, **kwargs)


class TransferManagerBase(ABC):

    def _pull_dest_size(self, dest):
        if os.path.isdir(dest):
            return sum(
                os.stat(os.path.join(dirpath, f)).st_size
	            for dirpath, _, fnames in os.walk(dest)
	            for f in fnames
            )
        else:
            return os.stat(dest).st_size
        return 0

    def _push_dest_size(self, dest):
        cmd = '{} du -s {}'.format(quote(self.conn.busybox), quote(dest))
        out = self.conn.execute(cmd)
        try:
            return int(out.split()[0])
        except ValueError:
            return 0

    def __init__(self, conn, poll_period, start_transfer_poll_delay, total_timeout):
        self.conn = conn
        self.poll_period = poll_period
        self.total_timeout = total_timeout
        self.start_transfer_poll_delay = start_transfer_poll_delay

        self.logger = logging.getLogger('FileTransfer')
        self.managing = threading.Event()
        self.transfer_started = threading.Event()
        self.transfer_completed = threading.Event()
        self.transfer_aborted = threading.Event()

        self.monitor_thread = None
        self.sources = None
        self.dest = None
        self.direction = None

    @abstractmethod
    def _cancel(self):
        pass

    def cancel(self, reason=None):
        msg = 'Cancelling file transfer {} -> {}'.format(self.sources, self.dest)
        if reason is not None:
            msg += ' due to \'{}\''.format(reason)
        self.logger.warning(msg)
        self.transfer_aborted.set()
        self._cancel()

    @abstractmethod
    def isactive(self):
        pass

    @contextmanager
    def manage(self, sources, dest, direction):
        try:
            self.sources, self.dest, self.direction = sources, dest, direction
            m_thread = threading.Thread(target=self._monitor)

            self.transfer_completed.clear()
            self.transfer_aborted.clear()
            self.transfer_started.set()

            m_thread.start()
            yield self
        except BaseException:
            self.cancel(reason='exception during transfer')
            raise
        finally:
            self.transfer_completed.set()
            self.transfer_started.set()
            m_thread.join()
            self.transfer_started.clear()
            self.transfer_completed.clear()
            self.transfer_aborted.clear()

    def _monitor(self):
        start_t = monotonic()
        self.transfer_completed.wait(self.start_transfer_poll_delay)
        while not self.transfer_completed.wait(self.poll_period):
            if not self.isactive():
                self.cancel(reason='transfer inactive')
            elif monotonic() - start_t > self.total_timeout:
                self.cancel(reason='transfer timed out')


class PopenTransferManager(TransferManagerBase):

    def __init__(self, conn, poll_period=30, start_transfer_poll_delay=30, total_timeout=3600):
        super().__init__(conn, poll_period, start_transfer_poll_delay, total_timeout)
        self.transfer = None
        self.last_sample = None

    def _cancel(self):
        if self.transfer:
            self.transfer.cancel()
            self.transfer = None
            self.last_sample = None

    def isactive(self):
        size_fn = self._push_dest_size if self.direction == 'push' else self._pull_dest_size
        curr_size = size_fn(self.dest)
        self.logger.debug('Polled file transfer, destination size {}'.format(curr_size))
        active = True if self.last_sample is None else curr_size > self.last_sample
        self.last_sample = curr_size
        return active

    def set_transfer_and_wait(self, popen_bg_cmd):
        self.transfer = popen_bg_cmd
        self.last_sample = None
        ret = self.transfer.wait()

        if ret and not self.transfer_aborted.is_set():
            raise subprocess.CalledProcessError(ret, self.transfer.popen.args)
        elif self.transfer_aborted.is_set():
            raise TimeoutError(self.transfer.popen.args)


class SSHTransferManager(TransferManagerBase):

    def __init__(self, conn, poll_period=30, start_transfer_poll_delay=30, total_timeout=3600):
        super().__init__(conn, poll_period, start_transfer_poll_delay, total_timeout)
        self.transferer = None
        self.progressed = False
        self.transferred = None
        self.to_transfer = None

    def _cancel(self):
        self.transferer.close()

    def isactive(self):
        progressed = self.progressed
        self.progressed = False
        msg = 'Polled transfer: {}% [{}B/{}B]'
        pc = format((self.transferred / self.to_transfer) * 100, '.2f')
        self.logger.debug(msg.format(pc, self.transferred, self.to_transfer))
        return progressed

    @contextmanager
    def manage(self, sources, dest, direction, transferer):
        with super().manage(sources, dest, direction):
            try:
                self.progressed = False
                self.transferer = transferer  # SFTPClient or SCPClient
                yield self
            except socket.error as e:
                if self.transfer_aborted.is_set():
                    self.transfer_aborted.clear()
                    method = 'SCP' if self.conn.use_scp else 'SFTP'
                    raise TimeoutError('{} {}: {} -> {}'.format(method, self.direction, sources, self.dest))
                else:
                    raise e

    def progress_cb(self, *args):
        if self.transfer_started.is_set():
            self.progressed = True
            if len(args) == 3:  # For SCPClient callbacks
                self.transferred = args[2]
                self.to_transfer = args[1]
            elif len(args) == 2:  # For SFTPClient callbacks
                self.transferred = args[0]
                self.to_transfer = args[1]
