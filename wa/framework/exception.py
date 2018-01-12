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
from devlib.exception import (DevlibError, HostError, TimeoutError,
                              TargetError, TargetNotRespondingError)

from wa.utils.misc import get_traceback


class WAError(Exception):
    """Base class for all Workload Automation exceptions."""
    pass


class NotFoundError(WAError):
    """Raised when the specified item is not found."""
    pass


class ValidationError(WAError):
    """Raised on failure to validate an extension."""
    pass


class WorkloadError(WAError):
    """General Workload error."""
    pass


class JobError(WAError):
    """Job execution error."""
    pass


class InstrumentError(WAError):
    """General Instrument error."""
    pass


class OutputProcessorError(WAError):
    """General OutputProcessor error."""
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


class ConfigError(WAError):
    """Raised when configuration provided is invalid. This error suggests that
    the user should modify their config and try again."""
    pass


class SerializerSyntaxError(Exception):
    """
    Error loading a serialized structure from/to a file handle.
    """

    def __init__(self, message, line=None, column=None):
        super(SerializerSyntaxError, self).__init__(message)
        self.line = line
        self.column = column

    def __str__(self):
        linestring = ' on line {}'.format(self.line) if self.line else ''
        colstring = ' in column {}'.format(self.column) if self.column else ''
        message = 'Syntax Error{}: {}'
        return message.format(''.join([linestring, colstring]), self.message)


class PluginLoaderError(WAError):
    """Raised when there is an error loading an extension or
    an external resource. Apart form the usual message, the __init__
    takes an exc_info parameter which should be the result of
    sys.exc_info() for the original exception (if any) that
    caused the error."""

    def __init__(self, message, exc_info=None):
        super(PluginLoaderError, self).__init__(message)
        self.exc_info = exc_info

    def __str__(self):
        if self.exc_info:
            orig = self.exc_info[1]
            orig_name = type(orig).__name__
            if isinstance(orig, WAError):
                reason = 'because of:\n{}: {}'.format(orig_name, orig)
            else:
                text = 'because of:\n{}\n{}: {}'
                reason = text.format(get_traceback(self.exc_info), orig_name, orig)
            return '\n'.join([self.message, reason])
        else:
            return self.message


class WorkerThreadError(WAError):
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
        text = 'Exception of type {} occured on thread {}:\n{}\n{}: {}'
        message = text.format(orig_name, thread, get_traceback(self.exc_info),
                              orig_name, orig)
        super(WorkerThreadError, self).__init__(message)

