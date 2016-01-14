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
import itertools


from wlauto import Instrument, Executable, Parameter
from wlauto.exceptions import ConfigError
from wlauto.utils.misc import ensure_file_directory_exists as _f
from wlauto.utils.types import list_or_string, list_of_strs

PERF_COMMAND_TEMPLATE = '{} stat {} {} sleep 1000 > {} 2>&1 '

DEVICE_RESULTS_FILE = '/data/local/perf_results.txt'
HOST_RESULTS_FILE_BASENAME = 'perf.txt'

PERF_COUNT_REGEX = re.compile(r'^(CPU\d+)?\s*(\d+)\s*(.*?)\s*(\[\s*\d+\.\d+%\s*\])?\s*$')


class PerfInstrument(Instrument):

    name = 'perf'
    description = """
    Perf is a Linux profiling with performance counters.

    Performance counters are CPU hardware registers that count hardware events
    such as instructions executed, cache-misses suffered, or branches
    mispredicted. They form a basis for profiling applications to trace dynamic
    control flow and identify hotspots.

    pref accepts options and events. If no option is given the default '-a' is
    used. For events, the default events are migrations and cs. They both can
    be specified in the config file.

    Events must be provided as a list that contains them and they will look like
    this ::

        perf_events = ['migrations', 'cs']

    Events can be obtained by typing the following in the command line on the
    device ::

        perf list

    Whereas options, they can be provided as a single string as following ::

        perf_options = '-a -i'

    Options can be obtained by running the following in the command line ::

        man perf-record
    """

    parameters = [
        Parameter('events', kind=list_of_strs, default=['migrations', 'cs'],
                  global_alias='perf_events',
                  constraint=(lambda x: x, 'must not be empty.'),
                  description="""Specifies the events to be counted."""),
        Parameter('optionstring', kind=list_or_string, default='-a',
                  global_alias='perf_options',
                  description="""Specifies options to be used for the perf command. This
                  may be a list of option strings, in which case, multiple instances of perf
                  will be kicked off -- one for each option string. This may be used to e.g.
                  collected different events from different big.LITTLE clusters.
                  """),
        Parameter('labels', kind=list_of_strs, default=None,
                  global_alias='perf_labels',
                  description="""Provides labels for pref output. If specified, the number of
                  labels must match the number of ``optionstring``\ s.
                  """),
        Parameter('force_install', kind=bool, default=False,
                  description="""
                  always install perf binary even if perf is already present on the device.
                  """),
    ]

    def on_run_init(self, context):
        binary = context.resolver.get(Executable(self, self.device.abi, 'perf'))
        if self.force_install:
            self.binary = self.device.install(binary)
        else:
            self.binary = self.device.install_if_needed(binary)
        self.commands = self._build_commands()

    def setup(self, context):
        self._clean_device()

    def start(self, context):
        for command in self.commands:
            self.device.kick_off(command)

    def stop(self, context):
        as_root = self.device.platform == 'android'
        self.device.killall('sleep', as_root=as_root)

    def update_result(self, context):
        for label in self.labels:
            device_file = self._get_device_outfile(label)
            host_relpath = os.path.join('perf', os.path.basename(device_file))
            host_file = _f(os.path.join(context.output_directory, host_relpath))
            self.device.pull_file(device_file, host_file)
            context.add_iteration_artifact(label, kind='raw', path=host_relpath)
            with open(host_file) as fh:
                in_results_section = False
                for line in fh:
                    if 'Performance counter stats' in line:
                        in_results_section = True
                        fh.next()  # skip the following blank line
                    if in_results_section:
                        if not line.strip():  # blank line
                            in_results_section = False
                            break
                        else:
                            line = line.split('#')[0]  # comment
                            match = PERF_COUNT_REGEX.search(line)
                            if match:
                                classifiers = {}
                                cpu = match.group(1)
                                if cpu is not None:
                                    classifiers['cpu'] = int(cpu.replace('CPU', ''))
                                count = int(match.group(2))
                                metric = '{}_{}'.format(label, match.group(3))
                                context.result.add_metric(metric, count, classifiers=classifiers)

    def teardown(self, context):  # pylint: disable=R0201
        self._clean_device()

    def validate(self):
        if isinstance(self.optionstring, list):
            self.optionstrings = self.optionstring
        else:
            self.optionstrings = [self.optionstring]
        if isinstance(self.events[0], list):  # we know events are non-empty due to param constraint pylint: disable=access-member-before-definition
            self.events = self.events
        else:
            self.events = [self.events]
        if not self.labels:  # pylint: disable=E0203
            self.labels = ['perf_{}'.format(i) for i in xrange(len(self.optionstrings))]
        if len(self.labels) != len(self.optionstrings):
            raise ConfigError('The number of labels must match the number of optstrings provided for perf.')

    def _build_commands(self):
        events = itertools.cycle(self.events)
        commands = []
        for opts, label in itertools.izip(self.optionstrings, self.labels):
            commands.append(self._build_perf_command(opts, events.next(), label))
        return commands

    def _clean_device(self):
        for label in self.labels:
            filepath = self._get_device_outfile(label)
            self.device.delete_file(filepath)

    def _get_device_outfile(self, label):
        return self.device.path.join(self.device.working_directory, '{}.out'.format(label))

    def _build_perf_command(self, options, events, label):
        event_string = ' '.join(['-e {}'.format(e) for e in events])
        command = PERF_COMMAND_TEMPLATE.format(self.binary,
                                               options or '',
                                               event_string,
                                               self._get_device_outfile(label))
        return command
