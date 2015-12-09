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


# pylint: disable=W0613,E1101,W0201
import os
import re
import csv

from wlauto import Instrument, settings, Parameter
from wlauto.instrumentation import instrument_is_installed
from wlauto.exceptions import ConfigError
from wlauto.utils.types import boolean


NUMBER_OF_CCI_PMU_COUNTERS = 4
DEFAULT_EVENTS = ['0x63', '0x6A', '0x83', '0x8A']
DEFAULT_PERIOD = 10  # in jiffies

CPL_BASE = '/sys/kernel/debug/cci_pmu_logger/'
CPL_CONTROL_FILE = CPL_BASE + 'control'
CPL_PERIOD_FILE = CPL_BASE + 'period_jiffies'

DRIVER = 'pmu_logger.ko'

REGEX = re.compile(r'(\d+(?:\.\d+)?):\s+bprint:.*Cycles:\s*(\S+)\s*Counter_0:\s*(\S+)\s*Counter_1:\s*(\S+)\s*Counter_2:\s*(\S+)\s*Counter_3:\s*(\S+)')


class CciPmuLogger(Instrument):

    name = "cci_pmu_logger"
    description = """
    This instrument allows collecting CCI counter data.

    It relies on the pmu_logger.ko kernel driver, the source for which is
    included with Workload Automation (see inside ``wlauto/external`` directory).
    You will need to build this against your specific kernel. Once compiled, it needs
    to be placed in the dependencies directory (usually ``~/.workload_uatomation/dependencies``).

    .. note:: When compling pmu_logger.ko for a new hardware platform, you may need to
              modify CCI_BASE inside pmu_logger.c to contain the base address of where
              CCI is mapped in memory on your device.

    This instrument relies on ``trace-cmd`` instrument to also be enabled. You should enable
    at least ``'bprint'`` trace event.

    """

    parameters = [
        Parameter('events', kind=list, default=DEFAULT_EVENTS,
                  global_alias='cci_pmu_events',
                  description="""
                  A list of strings, each representing an event to be counted. The length
                  of the list cannot exceed the number of PMU counters available (4 in CCI-400).
                  If this is not specified, shareable read transactions and snoop hits on both
                  clusters will be counted by default.  E.g. ``['0x63', '0x83']``.
                  """),
        Parameter('event_labels', kind=list, default=[],
                  global_alias='cci_pmu_event_labels',
                  description="""
                  A list of labels to be used when reporting PMU counts. If specified,
                  this must be of the same length as ``cci_pmu_events``. If not specified,
                  events will be labeled "event_<event_number>".
                  """),
        Parameter('period', kind=int, default=10,
                  global_alias='cci_pmu_period',
                  description='The period (in jiffies) between counter reads.'),
        Parameter('install_module', kind=boolean, default=True,
                  global_alias='cci_pmu_install_module',
                  description="""
                  Specifies whether pmu_logger has been compiled as a .ko module that needs
                  to be installed by the instrument. (.ko binary must be in {}). If this is set
                  to ``False``, it will be assumed that pmu_logger has been compiled into the kernel,
                  or that it has been installed prior to the invocation of WA.
                  """.format(settings.dependencies_directory)),
    ]

    def on_run_init(self, context):
        if self.install_module:
            self.device_driver_file = self.device.path.join(self.device.working_directory, DRIVER)
            host_driver_file = os.path.join(settings.dependencies_directory, DRIVER)
            self.device.push_file(host_driver_file, self.device_driver_file)

    def setup(self, context):
        if self.install_module:
            self.device.execute('insmod {}'.format(self.device_driver_file), check_exit_code=False)
        self.device.set_sysfile_value(CPL_PERIOD_FILE, self.period)
        for i, event in enumerate(self.events):
            counter = CPL_BASE + 'counter{}'.format(i)
            self.device.set_sysfile_value(counter, event, verify=False)

    def start(self, context):
        self.device.set_sysfile_value(CPL_CONTROL_FILE, 1, verify=False)

    def stop(self, context):
        self.device.set_sysfile_value(CPL_CONTROL_FILE, 1, verify=False)

    # Doing result processing inside teardown because need to make sure that
    # trace-cmd has processed its results and generated the trace.txt
    def teardown(self, context):
        trace_file = os.path.join(context.output_directory, 'trace.txt')
        rows = [['timestamp', 'cycles'] + self.event_labels]
        with open(trace_file) as fh:
            for line in fh:
                match = REGEX.search(line)
                if match:
                    rows.append([
                        float(match.group(1)),
                        int(match.group(2), 16),
                        int(match.group(3), 16),
                        int(match.group(4), 16),
                        int(match.group(5), 16),
                        int(match.group(6), 16),
                    ])
        output_file = os.path.join(context.output_directory, 'cci_counters.txt')
        with open(output_file, 'wb') as wfh:
            writer = csv.writer(wfh)
            writer.writerows(rows)
        context.add_iteration_artifact('cci_counters', path='cci_counters.txt', kind='data',
                                       description='CCI PMU counter data.')

        # summary metrics
        sums = map(sum, zip(*(r[1:] for r in rows[1:])))
        labels = ['cycles'] + self.event_labels
        for label, value in zip(labels, sums):
            context.result.add_metric('cci ' + label, value, lower_is_better=True)

        # actual teardown
        if self.install_module:
            self.device.execute('rmmod pmu_logger', check_exit_code=False)

    def validate(self):
        if not instrument_is_installed('trace-cmd'):
            raise ConfigError('To use cci_pmu_logger, trace-cmd instrument must also be enabled.')
        if not self.event_labels:  # pylint: disable=E0203
            self.event_labels = ['event_{}'.format(e) for e in self.events]
        elif len(self.events) != len(self.event_labels):
            raise ConfigError('cci_pmu_events and cci_pmu_event_labels must be of the same length.')
        if len(self.events) > NUMBER_OF_CCI_PMU_COUNTERS:
            raise ConfigError('The number cci_pmu_counters must be at most {}'.format(NUMBER_OF_CCI_PMU_COUNTERS))
