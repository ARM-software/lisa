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

from wlauto.core import instrumentation


def instrument_is_installed(instrument):
    """Returns ``True`` if the specified instrument is installed, and ``False``
    other wise. The insturment maybe specified either as a name or a subclass (or
    instance of subclass) of :class:`wlauto.core.Instrument`."""
    return instrumentation.is_installed(instrument)


def instrument_is_enabled(instrument):
    """Returns ``True`` if the specified instrument is installed and is currently
    enabled, and ``False`` other wise. The insturment maybe specified either
    as a name or a subclass (or instance of subclass) of
    :class:`wlauto.core.Instrument`."""
    return instrumentation.is_enabled(instrument)


def clear_instrumentation():
    instrumentation.installed = []
