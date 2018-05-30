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

class DevlibError(Exception):
    """Base class for all Devlib exceptions."""
    @property
    def message(self):
        if self.args:
            return self.args[0]
        return str(self)


class TargetError(DevlibError):
    """An error has occured on the target"""
    pass


class TargetNotRespondingError(DevlibError):
    """The target is unresponsive."""
    pass


class HostError(DevlibError):
    """An error has occured on the host"""
    pass


class TimeoutError(DevlibError):
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


def get_traceback(exc=None):
    """
    Returns the string with the traceback for the specifiec exc
    object, or for the current exception exc is not specified.

    """
    import io, traceback, sys
    if exc is None:
        exc = sys.exc_info()
    if not exc:
        return None
    tb = exc[2]
    sio = io.BytesIO()
    traceback.print_tb(tb, file=sio)
    del tb  # needs to be done explicitly see: http://docs.python.org/2/library/sys.html#sys.exc_info
    return sio.getvalue()
