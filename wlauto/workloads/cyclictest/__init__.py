#    Copyright 2012-2015 ARM Limited
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
# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init

import os
import time

from wlauto import settings, Workload, Executable, Parameter
from wlauto.exceptions import ConfigError, WorkloadError
from wlauto.utils.types import boolean

TXT_RESULT_NAME = 'cyclictest_result.txt'
RESULT_INTERPRETATION = {
    'T': 'Thread',
    'P': 'Priority',
    'C': 'Clock',
}


class Cyclictest(Workload):

    name = 'cyclictest'
    description = """
    Measures the amount of time that passes between when a timer expires and
    when the thread which set the timer actually runs.

    Cyclic test works by taking a time snapshot just prior to waiting for a specific
    time interval (t1), then taking another time snapshot after the timer
    finishes (t2), then comparing the theoretical wakeup time with the actual
    wakeup time (t2 -(t1 + sleep_time)). This value is the latency for that
    timers wakeup.

    """

    parameters = [
        Parameter('clock', allowed_values=['monotonic', 'realtime'], default='realtime',
                  description=('specify the clock to be used during the test.')),
        Parameter('duration', kind=int, default=30,
                  description=('Specify the length for the test to run in seconds.')),
        Parameter('quiet', kind=boolean, default=True,
                  description=('Run the tests quiet and print only a summary on exit.')),
        Parameter('thread', kind=int, default=8,
                  description=('Set the number of test threads')),
        Parameter('latency', kind=int, default=1000000,
                  description=('Write the value to /dev/cpu_dma_latency')),
        Parameter('extra_parameters', kind=str, default="",
                  description=('Any additional command line parameters to append to the '
                               'existing parameters above. A list can be found at '
                               'https://rt.wiki.kernel.org/index.php/Cyclictest or '
                               'in the help page ``cyclictest -h``')),
        Parameter('clear_file_cache', kind=boolean, default=True,
                  description=('Clear file caches before starting test')),
        Parameter('screen_off', kind=boolean, default=True,
                  description=('If true it will turn the screen off so that onscreen '
                               'graphics do not effect the score. This is predominantly '
                               'for devices without a GPU')),

    ]

    def setup(self, context):
        self.cyclictest_on_device = 'cyclictest'
        self.cyclictest_result = os.path.join(self.device.working_directory, TXT_RESULT_NAME)
        self.cyclictest_command = '{} --clock={} --duration={}s --thread={} --latency={} {} {} > {}'
        self.device_binary = None

        if not self.device.is_rooted:
            raise WorkloadError("This workload requires a device with root premissions to run")

        host_binary = context.resolver.get(Executable(self, self.device.abi, 'cyclictest'))
        self.device_binary = self.device.install(host_binary)

        self.cyclictest_command = self.cyclictest_command.format(self.device_binary,
                                                                 0 if self.clock == 'monotonic' else 1,
                                                                 self.duration,
                                                                 self.thread,
                                                                 self.latency,
                                                                 "--quiet" if self.quiet else "",
                                                                 self.extra_parameters,
                                                                 self.cyclictest_result)

        if self.clear_file_cache:
            self.device.execute('sync')
            self.device.set_sysfile_value('/proc/sys/vm/drop_caches', 3)

        if self.device.platform == 'android':
            if self.screen_off and self.device.is_screen_on:
                self.device.execute('input keyevent 26')

    def run(self, context):
        self.device.execute(self.cyclictest_command, self.duration * 2, as_root=True)

    def update_result(self, context):
        self.device.pull_file(self.cyclictest_result, context.output_directory)

        # Parsing the output
        # Standard Cyclictest Output:
        # T: 0 (31974) P:95 I:1000 C:4990 Min:9 Act:37 Avg:31 Max:59
        with open(os.path.join(context.output_directory, TXT_RESULT_NAME)) as f:
            for line in f:
                if line.find('C:') is not -1:
                    # Key = T: 0 (31974) P:95 I:1000
                    # Remaing = 49990 Min:9 Act:37 Avg:31 Max:59
                    # sperator = C:
                    (key, sperator, remaing) = line.partition('C:')

                    index = key.find('T')
                    key = key.replace(key[index], RESULT_INTERPRETATION['T'])
                    index = key.find('P')
                    key = key.replace(key[index], RESULT_INTERPRETATION['P'])

                    index = sperator.find('C')
                    sperator = sperator.replace(sperator[index], RESULT_INTERPRETATION['C'])

                    metrics = (sperator + remaing).split()
                    # metrics is now in the from of ['Min:', '9', 'Act:', '37', 'Avg:', '31' , 'Max', '59']
                    for i in range(0, len(metrics), 2):
                        full_key = key + ' ' + metrics[i][:-1]
                        value = int(metrics[i + 1])
                        context.result.add_metric(full_key, value, 'microseconds')

    def teardown(self, context):
        if self.device.platform == 'android':
            if self.screen_off:
                self.device.ensure_screen_is_on()
        self.device.execute('rm -f {}'.format(self.cyclictest_result))
