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
import time
from past.builtins import basestring, zip

from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.trace import TraceCollector
from devlib.utils.misc import ensure_file_directory_exists as _f


PERF_COMMAND_TEMPLATE = '{binary} {command} {options} {events} sleep 1000 > {outfile} 2>&1 '
PERF_REPORT_COMMAND_TEMPLATE= '{binary} report {options} -i {datafile} > {outfile} 2>&1 '
PERF_RECORD_COMMAND_TEMPLATE= '{binary} record {options} {events} -o {outfile}' 

PERF_DEFAULT_EVENTS = [
    'cpu-migrations',
    'context-switches',
]

SIMPLEPERF_DEFAULT_EVENTS = [
    'raw-cpu-cycles',
    'raw-l1-dcache',
    'raw-l1-dcache-refill',
    'raw-br-mis-pred',
    'raw-instruction-retired',
]

DEFAULT_EVENTS = {'perf':PERF_DEFAULT_EVENTS, 'simpleperf':SIMPLEPERF_DEFAULT_EVENTS}

class PerfCollector(TraceCollector):
    """
    Perf is a Linux profiling with performance counters.
    Simpleperf is an Android profiling tool with performance counters.

    It is highly recomended to use perf_type = simpleperf when using this instrument
    on android devices, since it recognises android symbols in record mode and is much more stable
    when reporting record .data files. For more information see simpleperf documentation at:
    https://android.googlesource.com/platform/system/extras/+/master/simpleperf/doc/README.md

    Performance counters are CPU hardware registers that count hardware events
    such as instructions executed, cache-misses suffered, or branches
    mispredicted. They form a basis for profiling applications to trace dynamic
    control flow and identify hotspots.

    pref accepts options and events. If no option is given the default '-a' is
    used. For events, the default events are migrations and cs for perf and raw-cpu-cycles,
    raw-l1-dcache, raw-l1-dcache-refill, raw-instructions-retired. They both can
    be specified in the config file.

    Events must be provided as a list that contains them and they will look like
    this ::

        perf_events = ['migrations', 'cs']

    Events can be obtained by typing the following in the command line on the
    device ::

        perf list
        simpleperf list

    Whereas options, they can be provided as a single string as following ::

        perf_options = '-a -i'

    Options can be obtained by running the following in the command line ::

        man perf-stat
    """

    def __init__(self, 
                 target,
                 perf_type='perf',
                 command='stat',
                 events=None,
                 optionstring=None,
                 report_options=None,
                 labels=None,
                 force_install=False):
        super(PerfCollector, self).__init__(target)
        self.force_install = force_install
        self.labels = labels
        self.report_options = report_options

        # Validate parameters
        if isinstance(optionstring, list):
            self.optionstrings = optionstring
        else:
            self.optionstrings = [optionstring]
        if perf_type in ['perf', 'simpleperf']:
            self.perf_type = perf_type
        else:
            raise ValueError('Invalid perf type: {}, must be perf or simpleperf'.format(perf_type))
        if not events:
            self.events = DEFAULT_EVENTS[self.perf_type]
        else:
            self.events = events
        if isinstance(self.events, basestring):
            self.events = [self.events]
        if not self.labels:
            self.labels = ['perf_{}'.format(i) for i in range(len(self.optionstrings))]
        if len(self.labels) != len(self.optionstrings):
            raise ValueError('The number of labels must match the number of optstrings provided for perf.')
        if command in ['stat', 'record']:
            self.command = command
        else:
            raise ValueError('Unsupported perf command, must be stat or record')

        self.binary = self.target.get_installed(self.perf_type)
        if self.force_install or not self.binary:
            self.binary = self._deploy_perf()

        self._validate_events(self.events)

        self.commands = self._build_commands()

    def reset(self):
        self.target.killall(self.perf_type, as_root=self.target.is_rooted)
        self.target.remove(self.target.get_workpath('TemporaryFile*'))
        for label in self.labels:
            filepath = self._get_target_file(label, 'data')
            self.target.remove(filepath)
            filepath = self._get_target_file(label, 'rpt')
            self.target.remove(filepath)

    def start(self):
        for command in self.commands:
            self.target.kick_off(command)

    def stop(self):
        self.target.killall(self.perf_type, signal='SIGINT',
                            as_root=self.target.is_rooted)
        # perf doesn't transmit the signal to its sleep call so handled here:
        self.target.killall('sleep', as_root=self.target.is_rooted)
        # NB: we hope that no other "important" sleep is on-going

    # pylint: disable=arguments-differ
    def get_trace(self, outdir):
        for label in self.labels:
            if self.command == 'record':
                self._wait_for_data_file_write(label, outdir)
                self._pull_target_file_to_host(label, 'rpt', outdir)
            else:
                self._pull_target_file_to_host(label, 'out', outdir)

    def _deploy_perf(self):
        host_executable = os.path.join(PACKAGE_BIN_DIRECTORY,
                                       self.target.abi, self.perf_type)
        return self.target.install(host_executable)

    def _get_target_file(self, label, extension):
        return self.target.get_workpath('{}.{}'.format(label, extension))

    def _build_commands(self):
        commands = []
        for opts, label in zip(self.optionstrings, self.labels):
            if self.command == 'stat':
                commands.append(self._build_perf_stat_command(opts, self.events, label))
            else:
                commands.append(self._build_perf_record_command(opts, label))
        return commands

    def _build_perf_stat_command(self, options, events, label):
        event_string = ' '.join(['-e {}'.format(e) for e in events])
        command = PERF_COMMAND_TEMPLATE.format(binary = self.binary,
                                               command = self.command,
                                               options = options or '',
                                               events = event_string,
                                               outfile = self._get_target_file(label, 'out'))
        return command

    def _build_perf_report_command(self, report_options, label):
        command = PERF_REPORT_COMMAND_TEMPLATE.format(binary=self.binary,
                                                      options=report_options or '',
                                                      datafile=self._get_target_file(label, 'data'),
                                                      outfile=self._get_target_file(label, 'rpt'))
        return command

    def _build_perf_record_command(self, options, label):
        event_string = ' '.join(['-e {}'.format(e) for e in self.events])
        command = PERF_RECORD_COMMAND_TEMPLATE.format(binary=self.binary,
                                                      options=options or '',
                                                      events=event_string,
                                                      outfile=self._get_target_file(label, 'data'))
        return command

    def _pull_target_file_to_host(self, label, extension, outdir):
        target_file = self._get_target_file(label, extension)
        host_relpath = os.path.basename(target_file)
        host_file = _f(os.path.join(outdir, host_relpath))
        self.target.pull(target_file, host_file)

    def _wait_for_data_file_write(self, label, outdir):
        data_file_finished_writing = False
        max_tries = 80
        current_tries = 0
        while not data_file_finished_writing:
            files = self.target.execute('cd {} && ls'.format(self.target.get_workpath('')))
            # Perf stores data in tempory files whilst writing to data output file. Check if they have been removed.
            if 'TemporaryFile' in files and current_tries <= max_tries:
                time.sleep(0.25)
                current_tries += 1
            else:
                if current_tries >= max_tries:
                    self.logger.warning('''writing {}.data file took longer than expected, 
                                        file may not have written correctly'''.format(label))
                data_file_finished_writing = True
        report_command = self._build_perf_report_command(self.report_options, label)
        self.target.execute(report_command)

    def _validate_events(self, events):
        available_events_string = self.target.execute('{} list'.format(self.perf_type))
        available_events = available_events_string.splitlines()
        for available_event in available_events:
            if available_event == '':
                continue
            if 'OR' in available_event:
                available_events.append(available_event.split('OR')[1]) 
            available_events[available_events.index(available_event)] = available_event.split()[0].strip()
        # Raw hex event codes can also be passed in that do not appear on perf/simpleperf list, prefixed with 'r'
        raw_event_code_regex = re.compile(r"^r(0x|0X)?[A-Fa-f0-9]+$")
        for event in events:
            if event in available_events or re.match(raw_event_code_regex, event):
                continue
            else:
                raise ValueError('Event: {} is not in available event list for {}'.format(event, self.perf_type))
