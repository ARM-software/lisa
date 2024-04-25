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


# pylint: disable=unused-argument
import csv
import os
import re

from devlib.collector.perf import PerfCollector

from wa import Instrument, Parameter, ConfigError
from wa.utils.types import list_or_string, list_of_strs, numeric

PERF_COUNT_REGEX = re.compile(r'^(CPU\d+)?\s*(\d+)\s*(.*?)\s*(\[\s*\d+\.\d+%\s*\])?\s*$')


class PerfInstrument(Instrument):

    name = 'perf'
    description = """
    Perf is a Linux profiling tool with performance counters.
    Simpleperf is an Android profiling tool with performance counters.

    It is highly recomended to use perf_type = simpleperf when using this instrument
    on android devices since it recognises android symbols in record mode and is much more stable
    when reporting record .data files. For more information see simpleperf documentation at:
    https://android.googlesource.com/platform/system/extras/+/master/simpleperf/doc/README.md

    Performance counters are CPU hardware registers that count hardware events
    such as instructions executed, cache-misses suffered, or branches
    mispredicted. They form a basis for profiling applications to trace dynamic
    control flow and identify hotspots.

    perf accepts options and events. If no option is given the default '-a' is
    used. For events, the default events for perf are migrations and cs. The default
    events for simpleperf are raw-cpu-cycles, raw-l1-dcache, raw-l1-dcache-refill, raw-instructions-retired.
    They both can be specified in the config file.

    Events must be provided as a list that contains them and they will look like
    this ::

        (for perf_type = perf ) perf_events = ['migrations', 'cs']
        (for perf_type = simpleperf) perf_events = ['raw-cpu-cycles', 'raw-l1-dcache']


    Events can be obtained by typing the following in the command line on the
    device ::

        perf list
        simpleperf list

    Whereas options, they can be provided as a single string as following ::

        perf_options = '-a -i'
        perf_options = '--app com.adobe.reader'

    Options can be obtained by running the following in the command line ::

        man perf-stat
    """

    parameters = [
        Parameter('perf_type', kind=str, allowed_values=['perf', 'simpleperf'], default='perf',
                  global_alias='perf_type', description="""Specifies which type of perf binaries
                  to install. Use simpleperf for collecting perf data on android systems."""),
        Parameter('command', kind=str, default='stat', allowed_values=['stat', 'record'],
                  global_alias='perf_command', description="""Specifies which perf command to use. If in record mode
                  report command will also be executed and results pulled from target along with raw data
                  file"""),
        Parameter('events', kind=list_of_strs, global_alias='perf_events',
                  description="""Specifies the events to be counted."""),
        Parameter('optionstring', kind=list_or_string, default='-a',
                  global_alias='perf_options',
                  description="""Specifies options to be used for the perf command. This
                  may be a list of option strings, in which case, multiple instances of perf
                  will be kicked off -- one for each option string. This may be used to e.g.
                  collected different events from different big.LITTLE clusters. In order to
                  profile a particular application process for android with simpleperf use
                  the --app option e.g. --app com.adobe.reader
                  """),
        Parameter('report_option_string', kind=str, global_alias='perf_report_options', default=None,
                  description="""Specifies options to be used to gather report when record command
                  is used. It's highly recommended to use perf_type simpleperf when running on
                  android devices as reporting options are unstable with perf"""),
        Parameter('run_report_sample', kind=bool, default=False, description="""If true, run
                  'perf/simpleperf report-sample'. It only works with the record command."""),
        Parameter('report_sample_options', kind=str, default=None,
                  description="""Specifies options to pass to report-samples when run_report_sample
                  is true."""),
        Parameter('labels', kind=list_of_strs, default=None,
                  global_alias='perf_labels',
                  description="""Provides labels for perf/simpleperf output for each optionstring.
                  If specified, the number of labels must match the number of ``optionstring``\ s.
                  """),
        Parameter('force_install', kind=bool, default=False,
                  description="""
                  always install perf binary even if perf is already present on the device.
                  """),
        Parameter('validate_pmu_events', kind=bool, default=True,
                  description="""
                  Query the hardware capabilities to verify the specified PMU events.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(PerfInstrument, self).__init__(target, **kwargs)
        self.collector = None
        self.outdir = None

    def validate(self):
        if self.report_option_string and (self.command != "record"):
            raise ConfigError("report_option_string only works with perf/simpleperf record. Set command to record or remove report_option_string")
        if self.report_sample_options and (self.command != "record"):
            raise ConfigError("report_sample_options only works with perf/simpleperf record. Set command to record or remove report_sample_options")
        if self.run_report_sample and (self.command != "record"):
            raise ConfigError("run_report_sample only works with perf/simpleperf record. Set command to record or remove run_report_sample")

    def initialize(self, context):
        if self.report_sample_options:
            self.run_report_sample = True

        self.collector = PerfCollector(self.target,
                                       self.perf_type,
                                       self.command,
                                       self.events,
                                       self.optionstring,
                                       self.report_option_string,
                                       self.run_report_sample,
                                       self.report_sample_options,
                                       self.labels,
                                       self.force_install,
                                       self.validate_pmu_events)

    def setup(self, context):
        self.outdir = os.path.join(context.output_directory, self.perf_type)
        self.collector.set_output(self.outdir)
        self.collector.reset()

    def start(self, context):
        self.collector.start()

    def stop(self, context):
        self.collector.stop()

    def update_output(self, context):
        self.logger.info('Extracting reports from target...')
        self.collector.get_data()

        if self.perf_type == 'perf':
            self._process_perf_output(context)
        else:
            self._process_simpleperf_output(context)

    def teardown(self, context):
        self.collector.reset()

    def _process_perf_output(self, context):
        if self.command == 'stat':
            self._process_perf_stat_output(context)
        elif self.command == 'record':
            self._process_perf_record_output(context)

    def _process_simpleperf_output(self, context):
        if self.command == 'stat':
            self._process_simpleperf_stat_output(context)
        elif self.command == 'record':
            self._process_simpleperf_record_output(context)

    def _process_perf_stat_output(self, context):
        for host_file in os.listdir(self.outdir):
            label = host_file.split('.out')[0]
            host_file_path = os.path.join(self.outdir, host_file)
            context.add_artifact(label, host_file_path, 'raw')
            with open(host_file_path) as fh:
                in_results_section = False
                for line in fh:
                    if 'Performance counter stats' in line:
                        in_results_section = True
                        next(fh)  # skip the following blank line
                    if not in_results_section:
                        continue
                    if not line.strip():  # blank line
                        in_results_section = False
                        break
                    else:
                        self._add_perf_stat_metric(line, label, context)

    @staticmethod
    def _add_perf_stat_metric(line, label, context):
        line = line.split('#')[0]  # comment
        match = PERF_COUNT_REGEX.search(line)
        if not match:
            return
        classifiers = {}
        cpu = match.group(1)
        if cpu is not None:
            classifiers['cpu'] = int(cpu.replace('CPU', ''))
        count = int(match.group(2))
        metric = '{}_{}'.format(label, match.group(3))
        context.add_metric(metric, count, classifiers=classifiers)

    def _process_perf_record_output(self, context):
        for host_file in os.listdir(self.outdir):
            label, ext = os.path.splitext(host_file)
            context.add_artifact(label, os.path.join(self.outdir, host_file), 'raw')
            column_headers = []
            column_header_indeces = []
            event_type = ''
            if ext == '.rpt':
                with open(os.path.join(self.outdir, host_file)) as fh:
                    for line in fh:
                        words = line.split()
                        if not words:
                            continue
                        event_type = self._get_report_event_type(words, event_type)
                        column_headers = self._get_report_column_headers(column_headers, words, 'perf')
                        for column_header in column_headers:
                            column_header_indeces.append(line.find(column_header))
                        self._add_report_metric(column_headers,
                                                column_header_indeces,
                                                line,
                                                words,
                                                context,
                                                event_type,
                                                label)

    @staticmethod
    def _get_report_event_type(words, event_type):
        if words[0] != '#':
            return event_type
        if len(words) == 6 and words[4] == 'event':
            event_type = words[5]
            event_type = event_type.strip("'")
        return event_type

    def _process_simpleperf_stat_output(self, context):
        labels = []
        for host_file in os.listdir(self.outdir):
            labels.append(host_file.split('.out')[0])
        for opts, label in zip(self.optionstring, labels):
            stat_file = os.path.join(self.outdir, '{}{}'.format(label, '.out'))
            if '--csv' in opts:
                self._process_simpleperf_stat_from_csv(stat_file, context, label)
            else:
                self._process_simpleperf_stat_from_raw(stat_file, context, label)

    @staticmethod
    def _process_simpleperf_stat_from_csv(stat_file, context, label):
        with open(stat_file) as csv_file:
            readCSV = csv.reader(csv_file, delimiter=',')
            line_num = 0
            for row in readCSV:
                if 'Performance counter statistics' not in row and 'Total test time' not in row:
                    classifiers = {}
                    if '%' in row:
                        classifiers['scaled from(%)'] = row[len(row) - 2].replace('(', '').replace(')', '').replace('%', '')
                    context.add_metric('{}_{}'.format(label, row[1]), row[0], 'count', classifiers=classifiers)
                line_num += 1

    @staticmethod
    def _process_simpleperf_stat_from_raw(stat_file, context, label):
        with open(stat_file) as fh:
            for line in fh:
                if '#' in line and not line.startswith('#'):
                    units = 'count'
                    if "(ms)" in line:
                        line = line.replace("(ms)", "")
                        units = 'ms'
                    tmp_line = line.split('#')[0]
                    tmp_line = line.strip()
                    count, metric = tmp_line.split(' ')[0], tmp_line.split(' ')[2]
                    count = float(count) if "." in count else int(count.replace(',', ''))
                    classifiers = {}
                    if '%' in line:
                        scaled_percentage = line.split('(')[1].strip().replace(')', '').replace('%', '')
                        classifiers['scaled from(%)'] = int(scaled_percentage)
                    metric = '{}_{}'.format(label, metric)
                    context.add_metric(metric, count, units, classifiers=classifiers)

    def _process_simpleperf_record_output(self, context):
        for host_file in os.listdir(self.outdir):
            label, ext = os.path.splitext(host_file)
            context.add_artifact(label, os.path.join(self.outdir, host_file), 'raw')
            if ext != '.rpt':
                continue
            column_headers = []
            column_header_indeces = []
            event_type = ''
            with open(os.path.join(self.outdir, host_file)) as fh:
                for line in fh:
                    words = line.split()
                    if not words:
                        continue
                    if words[0] == 'Event:':
                        event_type = words[1]
                    column_headers = self._get_report_column_headers(column_headers,
                                                                     words,
                                                                     'simpleperf')
                    for column_header in column_headers:
                        column_header_indeces.append(line.find(column_header))
                    self._add_report_metric(column_headers,
                                            column_header_indeces,
                                            line,
                                            words,
                                            context,
                                            event_type,
                                            label)

    @staticmethod
    def _get_report_column_headers(column_headers, words, perf_type):
        if 'Overhead' not in words:
            return column_headers
        if perf_type == 'perf':
            words.remove('#')
        column_headers = words
        # Concatonate Shared Objects header
        if 'Shared' in column_headers:
            shared_index = column_headers.index('Shared')
            column_headers[shared_index:shared_index + 2] = ['{} {}'.format(column_headers[shared_index],
                                                                            column_headers[shared_index + 1])]
        return column_headers

    @staticmethod
    def _add_report_metric(column_headers, column_header_indeces, line, words, context, event_type, label):
        if '%' not in words[0]:
            return
        classifiers = {}
        for i in range(1, len(column_headers)):
            classifiers[column_headers[i]] = line[column_header_indeces[i]:column_header_indeces[i + 1]].strip()

        context.add_metric('{}_{}_Overhead'.format(label, event_type),
                           numeric(words[0].strip('%')),
                           'percent',
                           classifiers=classifiers)
