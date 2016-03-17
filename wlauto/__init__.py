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

from wlauto.core.config.core import settings  # NOQA
from wlauto.core.device_manager import DeviceManager, RuntimeParameter, CoreParameter  # NOQA
from wlauto.core.command import Command  # NOQA
from wlauto.core.workload import Workload  # NOQA
from wlauto.core.plugin import Parameter, Artifact, Alias  # NOQA
import wlauto.core.pluginloader as PluginLoader  # NOQA
from wlauto.core.instrumentation import Instrument  # NOQA
from wlauto.core.result import ResultProcessor, IterationResult  # NOQA
from wlauto.core.resource import ResourceGetter, Resource, GetterPriority, NO_ONE  # NOQA
from wlauto.core.exttype import get_plugin_type  # NOQA Note: MUST be imported after other core imports.

from wlauto.common.resources import File, PluginAsset, Executable
from wlauto.common.android.resources import ApkFile, JarFile
from wlauto.common.android.workload import (UiAutomatorWorkload, ApkWorkload, AndroidBenchmark,  # NOQA
                                    AndroidUiAutoBenchmark, GameWorkload)  # NOQA

from wlauto.core.version import get_wa_version

__version__ = get_wa_version()
