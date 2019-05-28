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


import os
import re
from past.builtins import basestring, zip

from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.trace import TraceCollector
from devlib.utils.misc import ensure_file_directory_exists as _f


PERF_COMMAND_TEMPLATE = '{} stat {} {} sleep 1000 > {} 2>&1 '

PERF_COUNT_REGEX = re.compile(r'^(CPU\d+)?\s*(\d+)\s*(.*?)\s*(\[\s*\d+\.\d+%\s*\])?\s*$')

DEFAULT_EVENTS = [
    'migrations',
    'cs',
]


class PerfCollector(TraceCollector):
    """
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

        man perf-stat
    """

    def __init__(self, target,
                 events=None,
                 optionstring=None,
                 labels=None,
                 force_install=False):
        super(PerfCollector, self).__init__(target)
        self.events = events if events else DEFAULT_EVENTS
        self.force_install = force_install
        self.labels = labels

        # Validate parameters
        if isinstance(optionstring, list):
            self.optionstrings = optionstring
        else:
            self.optionstrings = [optionstring]
        if self.events and isinstance(self.events, basestring):
            self.events = [self.events]
        if not self.labels:
            self.labels = ['perf_{}'.format(i) for i in range(len(self.optionstrings))]
        if len(self.labels) != len(self.optionstrings):
            raise ValueError('The number of labels must match the number of optstrings provided for perf.')

        self.binary = self.target.get_installed('perf')
        if self.force_install or not self.binary:
            self.binary = self._deploy_perf()

        self.commands = self._build_commands()

    def reset(self):
        self.target.killall('perf', as_root=self.target.is_rooted)
        for label in self.labels:
            filepath = self._get_target_outfile(label)
            self.target.remove(filepath)

    def start(self):
        for command in self.commands:
            self.target.kick_off(command)

    def stop(self):
        self.target.killall('perf', signal='SIGINT',
                            as_root=self.target.is_rooted)
        # perf doesn't transmit the signal to its sleep call so handled here:
        self.target.killall('sleep', as_root=self.target.is_rooted)
        # NB: we hope that no other "important" sleep is on-going

    # pylint: disable=arguments-differ
    def get_trace(self, outdir):
        for label in self.labels:
            target_file = self._get_target_outfile(label)
            host_relpath = os.path.basename(target_file)
            host_file = _f(os.path.join(outdir, host_relpath))
            self.target.pull(target_file, host_file)

    def _deploy_perf(self):
        host_executable = os.path.join(PACKAGE_BIN_DIRECTORY,
                                       self.target.abi, 'perf')
        return self.target.install(host_executable)

    def _build_commands(self):
        commands = []
        for opts, label in zip(self.optionstrings, self.labels):
            commands.append(self._build_perf_command(opts, self.events, label))
        return commands

    def _get_target_outfile(self, label):
        return self.target.get_workpath('{}.out'.format(label))

    def _build_perf_command(self, options, events, label):
        event_string = ' '.join(['-e {}'.format(e) for e in events])
        command = PERF_COMMAND_TEMPLATE.format(self.binary,
                                               options or '',
                                               event_string,
                                               self._get_target_outfile(label))
        return command
