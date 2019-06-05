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

from devlib.target import Target, LinuxTarget, AndroidTarget, LocalLinuxTarget, ChromeOsTarget
from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.exception import DevlibError, DevlibTransientError, DevlibStableError, TargetError, TargetTransientError, TargetStableError, TargetNotRespondingError, HostError

from devlib.module import Module, HardRestModule, BootModule, FlashModule
from devlib.module import get_module, register_module

from devlib.platform import Platform
from devlib.platform.arm import TC2, Juno, JunoEnergyInstrument
from devlib.platform.gem5 import Gem5SimulationPlatform

from devlib.instrument import Instrument, InstrumentChannel, Measurement, MeasurementsCsv
from devlib.instrument import MEASUREMENT_TYPES, INSTANTANEOUS, CONTINUOUS
from devlib.instrument.daq import DaqInstrument
from devlib.instrument.energy_probe import EnergyProbeInstrument
from devlib.instrument.arm_energy_probe import ArmEnergyProbeInstrument
from devlib.instrument.frames import GfxInfoFramesInstrument, SurfaceFlingerFramesInstrument
from devlib.instrument.hwmon import HwmonInstrument
from devlib.instrument.monsoon import MonsoonInstrument
from devlib.instrument.netstats import NetstatsInstrument
from devlib.instrument.gem5power import Gem5PowerInstrument
from devlib.instrument.baylibre_acme import (
        BaylibreAcmeNetworkInstrument,
        BaylibreAcmeXMLInstrument,
        BaylibreAcmeLocalInstrument,
        BaylibreAcmeInstrument,
)

from devlib.derived import DerivedMeasurements, DerivedMetric
from devlib.derived.energy import DerivedEnergyMeasurements
from devlib.derived.fps import DerivedGfxInfoStats, DerivedSurfaceFlingerStats

from devlib.trace.ftrace import FtraceCollector
from devlib.trace.perf import PerfCollector
from devlib.trace.serial_trace import SerialTraceCollector
from devlib.trace.dmesg import DmesgCollector

from devlib.host import LocalConnection
from devlib.utils.android import AdbConnection
from devlib.utils.ssh import SshConnection, TelnetConnection, Gem5Connection

from devlib.utils.version import (get_devlib_version as __get_devlib_version,
                                  get_commit as __get_commit)


__version__ = __get_devlib_version()

__commit = __get_commit()
if __commit:
    __full_version__ = '{}+{}'.format(__version__, __commit)
else:
    __full_version__ = __version__
