#    Copyright 2018 ARM Limited
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

from wa.framework import pluginloader, signal
from wa.framework.command import Command, ComplexCommand, SubCommand
from wa.framework.configuration import settings
from wa.framework.configuration.core import Status
from wa.framework.exception import (CommandError, ConfigError, HostError, InstrumentError,  # pylint: disable=redefined-builtin
                                    JobError, NotFoundError, OutputProcessorError,
                                    PluginLoaderError, ResourceError, TargetError,
                                    TargetNotRespondingError, TimeoutError, ToolError,
                                    ValidationError, WAError, WorkloadError, WorkerThreadError)
from wa.framework.instrument import (Instrument, extremely_slow, very_slow, slow, normal, fast,
                                     very_fast, extremely_fast, hostside)
from wa.framework.output import RunOutput, discover_wa_outputs
from wa.framework.output_processor import OutputProcessor
from wa.framework.plugin import Plugin, Parameter, Alias
from wa.framework.resource import (NO_ONE, JarFile, ApkFile, ReventFile, File,
                                   Executable)
from wa.framework.target.descriptor import (TargetDescriptor, TargetDescription,
                                            create_target_description, add_description_for_target)
from wa.framework.workload import (Workload, ApkWorkload, ApkUiautoWorkload,
                                   ApkReventWorkload, UIWorkload, UiautoWorkload,
                                   PackageHandler, ReventWorkload, TestPackageHandler)


from wa.framework.version import get_wa_version, get_wa_version_with_commit

__version__ = get_wa_version()
__full_version__ = get_wa_version_with_commit()
