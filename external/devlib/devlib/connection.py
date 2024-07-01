#    Copyright 2024 ARM Limited
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
from contextlib import contextmanager, nullcontext
from shlex import quote
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import logging
import select
import fcntl

from devlib.utils.misc import InitCheckpoint, memoized

_KILL_TIMEOUT = 3


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
    def __init__(
        self,
        poll_transfers=False,
        start_transfer_poll_delay=30,
        total_transfer_timeout=3600,
        transfer_poll_period=30,
    ):
        self._current_bg_cmds = set()
        self._closed = False
        self._close_lock = threading.Lock()
        self.busybox = None
        self.logger = logging.getLogger('Connection')

        self.transfer_manager = TransferManager(
            self,
            start_transfer_poll_delay=start_transfer_poll_delay,
            total_transfer_timeout=total_transfer_timeout,
            transfer_poll_period=transfer_poll_period,
        ) if poll_transfers else NoopTransferManager()


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

        def finish_bg():
            bg_cmds = set(self._current_bg_cmds)
            n = len(bg_cmds)
            if n:
                self.logger.debug(f'Canceling {n} background commands before closing connection')
            for bg_cmd in bg_cmds:
                bg_cmd.cancel()

        # Locking the closing allows any thread to safely call close() as long
        # as the connection can be closed from a thread that is not the one it
        # started its life in.
        with self._close_lock:
            if not self._closed:
                finish_bg()
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

    def __init__(self, conn, data_dir, cmd, as_root):
        self.conn = conn
        self._data_dir = data_dir
        self.as_root = as_root
        self.cmd = cmd

        # Poll currently opened background commands on that connection to make
        # them deregister themselves if they are completed. This avoids
        # accumulating terminated commands and therefore leaking associated
        # resources if the user is not careful and does not use the context
        # manager API.
        for bg_cmd in set(conn._current_bg_cmds):
            try:
                bg_cmd.poll()
            # We don't want anything to fail here because of another command
            except Exception:
                pass

        conn._current_bg_cmds.add(self)

    @classmethod
    def from_factory(cls, conn, cmd, as_root, make_init_kwargs):
        cmd, data_dir = cls._with_data_dir(conn, cmd)
        return cls(
            conn=conn,
            data_dir=data_dir,
            cmd=cmd,
            as_root=as_root,
            **make_init_kwargs(cmd),
        )

    def _deregister(self):
        try:
            self.conn._current_bg_cmds.remove(self)
        except KeyError:
            pass

    @property
    def _pid_file(self):
        return str(Path(self._data_dir, 'pid'))

    @property
    @memoized
    def _targeted_pid(self):
        """
        PID of the process pointed at by ``devlib-signal-target`` command.
        """
        path = quote(self._pid_file)
        busybox = quote(self.conn.busybox)

        def execute(cmd):
            return self.conn.execute(cmd, as_root=self.as_root)

        while self.poll() is None:
            try:
                pid = execute(f'{busybox} cat {path}')
            except subprocess.CalledProcessError:
                time.sleep(0.01)
            else:
                if pid.endswith('\n'):
                    return int(pid.strip())
                else:
                    # We got a partial write in the PID file
                    continue

        raise ValueError(f'The background commmand did not use devlib-signal-target wrapper to designate which command should be the target of signals')

    @classmethod
    def _with_data_dir(cls, conn, cmd):
        busybox = quote(conn.busybox)
        data_dir = conn.execute(f'{busybox} mktemp -d').strip()
        cmd = f'_DEVLIB_BG_CMD_DATA_DIR={data_dir} exec {busybox} sh -c {quote(cmd)}'
        return cmd, data_dir

    def _cleanup_data_dir(self):
        path = quote(self._data_dir)
        busybox = quote(self.conn.busybox)
        cmd = f'{busybox} rm -r {path} || true'
        self.conn.execute(cmd, as_root=self.as_root)

    def send_signal(self, sig):
        """
        Send a POSIX signal to the background command's process group ID
        (PGID).

        :param signal: Signal to send.
        :type signal: signal.Signals
        """

        def execute(cmd):
            return self.conn.execute(cmd, as_root=self.as_root)

        def send(sig):
            busybox = quote(self.conn.busybox)
            # If the command has already completed, we don't want to send a
            # signal to another process that might have gotten that PID in the
            # meantime.
            if self.poll() is None:
                if sig in (signal.SIGTERM, signal.SIGQUIT, signal.SIGKILL):
                    # Use -PGID to target a process group rather than just the
                    # process itself. This will work in any condition and will
                    # not require cooperation from the command.
                    execute(f'{busybox} kill -{sig.value} -{self.pid}')
                else:
                    # Other signals require cooperation from the shell command
                    # so that it points to a specific process using
                    # devlib-signal-target
                    pid = self._targeted_pid
                    execute(f'{busybox} kill -{sig.value} {pid}')
        try:
            return send(sig)
        finally:
            # Deregister if the command has finished
            self.poll()

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
        try:
            if self.poll() is None:
                return self._cancel(kill_timeout=kill_timeout)
        finally:
            self._deregister()

    @abstractmethod
    def _cancel(self, kill_timeout):
        """
        Method to override in subclasses to implement :meth:`cancel`.
        """
        pass

    @abstractmethod
    def _wait(self):
        pass

    def wait(self):
        """
        Block until the background command completes, and return its exit code.
        """
        try:
            return self._wait()
        finally:
            self._deregister()

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
    def _poll(self):
        pass

    def poll(self):
        """
        Return exit code if the command has exited, None otherwise.
        """
        retcode = self._poll()
        if retcode is not None:
            self._deregister()
        return retcode

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
    def _close(self):
        pass

    def close(self):
        """
        Close all opened streams and then wait for command completion.

        :returns: Exit code of the command.

        .. note:: If the command is writing to its stdout/stderr, it might be
            blocked on that and die when the streams are closed.
        """
        try:
            return self._close()
        finally:
            self._deregister()
            self._cleanup_data_dir()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class PopenBackgroundCommand(BackgroundCommand):
    """
    :class:`subprocess.Popen`-based background command.
    """

    def __init__(self, conn, data_dir, cmd, as_root, popen):
        super().__init__(
            conn=conn,
            data_dir=data_dir,
            cmd=cmd,
            as_root=as_root,
        )
        self.popen = popen

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

    def _wait(self):
        return self.popen.wait()

    def _communicate(self, input, timeout):
        return _popen_communicate(self, self.popen, input, timeout)

    def _poll(self):
        return self.popen.poll()

    def _cancel(self, kill_timeout):
        popen = self.popen
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        try:
            popen.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)

    def _close(self):
        self.popen.__exit__(None, None, None)
        return self.popen.returncode

    def __enter__(self):
        super().__enter__()
        self.popen.__enter__()
        return self


class ParamikoBackgroundCommand(BackgroundCommand):
    """
    :mod:`paramiko`-based background command.
    """
    def __init__(self, conn, data_dir, cmd, as_root, chan, pid, stdin, stdout, stderr, redirect_thread):
        super().__init__(
            conn=conn,
            data_dir=data_dir,
            cmd=cmd,
            as_root=as_root,
        )

        self.chan = chan
        self._pid = pid
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self.redirect_thread = redirect_thread

    @property
    def pid(self):
        return self._pid

    def _wait(self):
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

        start = time.monotonic()

        while ret is None:
            # Even if ret is not None anymore, we need to drain the streams
            ret = self.poll()

            if timeout is not None and ret is None and time.monotonic() - start >= timeout:
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

    def _poll(self):
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

    def _close(self):
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

    def __init__(self, conn, data_dir, cmd, as_root, adb_popen, pid):
        super().__init__(
            conn=conn,
            data_dir=data_dir,
            cmd=cmd,
            as_root=as_root,
        )
        self.adb_popen = adb_popen
        self._pid = pid

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

    def _wait(self):
        return self.adb_popen.wait()

    def _communicate(self, input, timeout):
        return _popen_communicate(self, self.adb_popen, input, timeout)

    def _poll(self):
        return self.adb_popen.poll()

    def _cancel(self, kill_timeout):
        self.send_signal(signal.SIGTERM)
        try:
            self.adb_popen.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            self.send_signal(signal.SIGKILL)
            self.adb_popen.kill()

    def _close(self):
        self.adb_popen.__exit__(None, None, None)
        return self.adb_popen.returncode

    def __enter__(self):
        super().__enter__()
        self.adb_popen.__enter__()
        return self


class TransferManager:
    def __init__(self, conn, transfer_poll_period=30, start_transfer_poll_delay=30, total_transfer_timeout=3600):
        self.conn = conn
        self.transfer_poll_period = transfer_poll_period
        self.total_transfer_timeout = total_transfer_timeout
        self.start_transfer_poll_delay = start_transfer_poll_delay

        self.logger = logging.getLogger('FileTransfer')

    @contextmanager
    def manage(self, sources, dest, direction, handle):
        excep = None
        stop_thread = threading.Event()

        def monitor():
            nonlocal excep

            def cancel(reason):
                self.logger.warning(
                    f'Cancelling file transfer {sources} -> {dest} due to: {reason}'
                )
                handle.cancel()

            start_t = time.monotonic()
            stop_thread.wait(self.start_transfer_poll_delay)
            while not stop_thread.wait(self.transfer_poll_period):
                if not handle.isactive():
                    cancel(reason='transfer inactive')
                elif time.monotonic() - start_t > self.total_transfer_timeout:
                    cancel(reason='transfer timed out')
                    excep = TimeoutError(f'{direction}: {sources} -> {dest}')

        m_thread = threading.Thread(target=monitor, daemon=True)
        try:
            m_thread.start()
            yield self
        finally:
            stop_thread.set()
            m_thread.join()
            if excep is not None:
                raise excep


class NoopTransferManager:
    def manage(self, *args, **kwargs):
        return nullcontext(self)


class TransferHandleBase(ABC):
    def __init__(self, manager):
        self.manager = manager

    @property
    def logger(self):
        return self.manager.logger

    @abstractmethod
    def isactive(self):
        pass

    @abstractmethod
    def cancel(self):
        pass


class PopenTransferHandle(TransferHandleBase):
    def __init__(self, popen, dest, direction, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if direction == 'push':
            sample_size = self._push_dest_size
        elif direction == 'pull':
            sample_size = self._pull_dest_size
        else:
            raise ValueError(f'Unknown direction: {direction}')

        self.sample_size = lambda: sample_size(dest)

        self.popen = popen
        self.last_sample = 0

    @staticmethod
    def _pull_dest_size(dest):
        if os.path.isdir(dest):
            return sum(
                os.stat(os.path.join(dirpath, f)).st_size
	            for dirpath, _, fnames in os.walk(dest)
	            for f in fnames
            )
        else:
            return os.stat(dest).st_size

    def _push_dest_size(self, dest):
        conn = self.manager.conn
        cmd = '{} du -s -- {}'.format(quote(conn.busybox), quote(dest))
        out = conn.execute(cmd)
        return int(out.split()[0])

    def cancel(self):
        self.popen.terminate()

    def isactive(self):
        try:
            curr_size = self.sample_size()
        except Exception as e:
            self.logger.debug(f'File size polling failed: {e}')
            return True
        else:
            self.logger.debug(f'Polled file transfer, destination size: {curr_size}')
            if curr_size:
                active = curr_size > self.last_sample
                self.last_sample = curr_size
                return active
            # If the file is empty it will never grow in size, so we assume
            # everything is going well.
            else:
                return True


class SSHTransferHandle(TransferHandleBase):

    def __init__(self, handle, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SFTPClient or SSHClient
        self.handle = handle

        self.progressed = False
        self.transferred = 0
        self.to_transfer = 0

    def cancel(self):
        self.handle.close()

    def isactive(self):
        progressed = self.progressed
        if progressed:
            self.progressed = False
            pc = (self.transferred / self.to_transfer) * 100
            self.logger.debug(
                f'Polled transfer: {pc:.2f}% [{self.transferred}B/{self.to_transfer}B]'
            )
        return progressed

    def progress_cb(self, transferred, to_transfer):
        self.progressed = True
        self.transferred = transferred
        self.to_transfer = to_transfer
