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


from devlib.utils.misc import TimeoutError  # NOQA pylint: disable=W0611


class DevlibError(Exception):
    """Base class for all Workload Automation exceptions."""
    pass


class TargetError(DevlibError):
    """An error has occured on the target"""
    pass


class TargetNotRespondingError(DevlibError):
    """The target is unresponsive."""

    def __init__(self, target):
        super(TargetNotRespondingError, self).__init__('Target {} is not responding.'.format(target))


class HostError(DevlibError):
    """An error has occured on the host"""
    pass

