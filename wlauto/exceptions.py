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


from wlauto.utils.misc import get_traceback, TimeoutError  # NOQA pylint: disable=W0611


class WAError(Exception):
    """Base class for all Workload Automation exceptions."""
    pass


class NotFoundError(WAError):
    """Raised when the specified item is not found."""
    pass


class ValidationError(WAError):
    """Raised on failure to validate an extension."""
    pass


class DeviceError(WAError):
    """General Device error."""
    pass


class DeviceNotRespondingError(WAError):
    """The device is not responding."""

    def __init__(self, device):
        super(DeviceNotRespondingError, self).__init__('Device {} is not responding.'.format(device))


class WorkloadError(WAError):
    """General Workload error."""
    pass


class HostError(WAError):
    """Problem with the host on which WA is running."""
    pass


class ModuleError(WAError):
    """
    Problem with a module.

    .. note:: Modules for specific extension types should raise execeptions
              appropriate to that extension. E.g. a ``Device`` module should raise
              ``DeviceError``. This is intended for situation where a module is
              unsure (and/or doesn't care) what its owner is.

    """
    pass


class InstrumentError(WAError):
    """General Instrument error."""
    pass


class ResultProcessorError(WAError):
    """General ResultProcessor error."""
    pass


class ResourceError(WAError):
    """General Resolver error."""
    pass


class CommandError(WAError):
    """Raised by commands when they have encountered an error condition
    during execution."""
    pass


class ToolError(WAError):
    """Raised by tools when they have encountered an error condition
    during execution."""
    pass


class LoaderError(WAError):
    """Raised when there is an error loading an extension or
    an external resource. Apart form the usual message, the __init__
    takes an exc_info parameter which should be the result of
    sys.exc_info() for the original exception (if any) that
    caused the error."""

    def __init__(self, message, exc_info=None):
        super(LoaderError, self).__init__(message)
        self.exc_info = exc_info

    def __str__(self):
        if self.exc_info:
            orig = self.exc_info[1]
            orig_name = type(orig).__name__
            if isinstance(orig, WAError):
                reason = 'because of:\n{}: {}'.format(orig_name, orig)
            else:
                reason = 'because of:\n{}\n{}: {}'.format(get_traceback(self.exc_info), orig_name, orig)
            return '\n'.join([self.message, reason])
        else:
            return self.message


class ConfigError(WAError):
    """Raised when configuration provided is invalid. This error suggests that
    the user should modify their config and try again."""
    pass


class WorkerThreadError(WAError):
    """
    This should get raised  in the main thread if a non-WAError-derived exception occurs on
    a worker/background thread. If a WAError-derived exception is raised in the worker, then
    it that exception should be re-raised on the main thread directly -- the main point of this is
    to preserve the backtrace in the output, and backtrace doesn't get output for WAErrors.

    """

    def __init__(self, thread, exc_info):
        self.thread = thread
        self.exc_info = exc_info
        orig = self.exc_info[1]
        orig_name = type(orig).__name__
        message = 'Exception of type {} occured on thread {}:\n'.format(orig_name, thread)
        message += '{}\n{}: {}'.format(get_traceback(self.exc_info), orig_name, orig)
        super(WorkerThreadError, self).__init__(message)
