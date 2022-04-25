#    Copyright 2013-2018 ARM Limited
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

import subprocess

class DevlibError(Exception):
    """Base class for all Devlib exceptions."""

    def __init__(self, *args):
        message = args[0] if args else None
        self._message = message

    @property
    def message(self):
        try:
            msg = self._message
        except AttributeError:
            msg = None

        if msg is None:
            return str(self)
        else:
            return self._message


class DevlibStableError(DevlibError):
    """Non transient target errors, that are not subject to random variations
    in the environment and can be reliably linked to for example a missing
    feature on a target."""
    pass


class DevlibTransientError(DevlibError):
    """Exceptions inheriting from ``DevlibTransientError`` represent random
    transient events that are usually related to issues in the environment, as
    opposed to programming errors, for example network failures or
    timeout-related exceptions. When the error could come from
    indistinguishable transient or non-transient issue, it can generally be
    assumed that the configuration is correct and therefore, a transient
    exception is raised."""
    pass


class TargetError(DevlibError):
    """An error has occured on the target"""
    pass


class TargetTransientError(TargetError, DevlibTransientError):
    """Transient target errors that can happen randomly when everything is
    properly configured."""
    pass


class TargetStableError(TargetError, DevlibStableError):
    """Non-transient target errors that can be linked to a programming error or
    a configuration issue, and is not influenced by non-controllable parameters
    such as network issues."""
    pass


class TargetCalledProcessError(subprocess.CalledProcessError, TargetError):
    """Exception raised when a command executed on the target fails."""
    def __str__(self):
        msg = super().__str__()
        def decode(s):
            try:
                s = s.decode()
            except AttributeError:
                s = str(s)

            return s.strip()

        if self.stdout is not None and self.stderr is None:
            out = ['OUTPUT: {}'.format(decode(self.output))]
        else:
            out = [
                'STDOUT: {}'.format(decode(self.output)) if self.output is not None else '',
                'STDERR: {}'.format(decode(self.stderr)) if self.stderr is not None else '',
            ]

        return '\n'.join((
            msg,
            *out,
        ))


class TargetStableCalledProcessError(TargetCalledProcessError, TargetStableError):
    """Variant of :exc:`devlib.exception.TargetCalledProcessError` that indicates a stable error"""
    pass


class TargetTransientCalledProcessError(TargetCalledProcessError, TargetTransientError):
    """Variant of :exc:`devlib.exception.TargetCalledProcessError` that indicates a transient error"""
    pass


class TargetNotRespondingError(TargetTransientError):
    """The target is unresponsive."""
    pass


class HostError(DevlibError):
    """An error has occured on the host"""
    pass


# pylint: disable=redefined-builtin
class TimeoutError(DevlibTransientError):
    """Raised when a subprocess command times out. This is basically a ``DevlibError``-derived version
    of ``subprocess.CalledProcessError``, the thinking being that while a timeout could be due to
    programming error (e.g. not setting long enough timers), it is often due to some failure in the
    environment, and there fore should be classed as a "user error"."""

    def __init__(self, command, output):
        super(TimeoutError, self).__init__('Timed out: {}'.format(command))
        self.command = command
        self.output = output

    def __str__(self):
        return '\n'.join([self.message, 'OUTPUT:', self.output or ''])


class WorkerThreadError(DevlibError):
    """
    This should get raised  in the main thread if a non-WAError-derived
    exception occurs on a worker/background thread. If a WAError-derived
    exception is raised in the worker, then it that exception should be
    re-raised on the main thread directly -- the main point of this is to
    preserve the backtrace in the output, and backtrace doesn't get output for
    WAErrors.

    """

    def __init__(self, thread, exc_info):
        self.thread = thread
        self.exc_info = exc_info
        orig = self.exc_info[1]
        orig_name = type(orig).__name__
        message = 'Exception of type {} occured on thread {}:\n'.format(orig_name, thread)
        message += '{}\n{}: {}'.format(get_traceback(self.exc_info), orig_name, orig)
        super(WorkerThreadError, self).__init__(message)


class KernelConfigKeyError(KeyError, IndexError, DevlibError):
    """
    Exception raised when a kernel config option cannot be found.

    It inherits from :exc:`IndexError` for backward compatibility, and
    :exc:`KeyError` to behave like a regular mapping.
    """
    pass


def get_traceback(exc=None):
    """
    Returns the string with the traceback for the specifiec exc
    object, or for the current exception exc is not specified.

    """
    import io, traceback, sys  # pylint: disable=multiple-imports
    if exc is None:
        exc = sys.exc_info()
    if not exc:
        return None
    tb = exc[2]
    sio = io.StringIO()
    traceback.print_tb(tb, file=sio)
    del tb  # needs to be done explicitly see: http://docs.python.org/2/library/sys.html#sys.exc_info
    return sio.getvalue()
