#    Copyright 2015 ARM Limited
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
# pylint: disable=access-member-before-definition,attribute-defined-outside-init,unused-argument
import os

import pandas as pd

from wa import Instrument, Parameter, Executable
from wa.framework import signal
from wa.framework.exception import ConfigError, InstrumentError
from wa.utils.trace_cmd import TraceCmdParser
from wa.utils.types import list_or_string


class FilePoller(Instrument):
    name = 'file_poller'
    description = """
    Polls the given files at a set sample interval. The values are output in CSV format.

    This instrument places a file called poller.csv in each iterations result directory.
    This file will contain a timestamp column which will be in uS, the rest of the columns
    will be the contents of the polled files at that time.

    This instrument will strip any commas or new lines for the files' values
    before writing them.
    """

    parameters = [
        Parameter('sample_interval', kind=int, default=1000,
                  description="""The interval between samples in mS."""),
        Parameter('files', kind=list_or_string, mandatory=True,
                  description="""A list of paths to the files to be polled"""),
        Parameter('labels', kind=list_or_string,
                  description="""
                  A list of lables to be used in the CSV output for the
                  corresponding files. This cannot be used if a `*` wildcard is
                  used in a path.
                  """),
        Parameter('align_with_ftrace', kind=bool, default=False,
                  description="""
                  Insert a marker into ftrace that aligns with the first
                  timestamp. During output processing, extract the marker
                  and use it's timestamp to adjust the timestamps in the collected
                  csv so that they align with ftrace.
                  """),
        Parameter('as_root', kind=bool, default=False,
                  description="""
                  Whether or not the poller will be run as root. This should be
                  used when the file you need to poll can only be accessed by root.
                  """),
    ]

    def validate(self):
        if not self.files:
            raise ConfigError('You must specify atleast one file to poll')
        if self.labels and any(['*' in f for f in self.files]):
            raise ConfigError('You cannot used manual labels with `*` wildcards')

    def initialize(self, context):
        if not self.target.is_rooted and self.as_root:
            raise ConfigError('The target is not rooted, cannot run poller as root.')
        host_poller = context.resolver.get(Executable(self, self.target.abi,
                                                      "poller"))
        target_poller = self.target.install(host_poller)

        expanded_paths = []
        for path in self.files:
            if "*" in path:
                for p in self.target.list_directory(path):
                    expanded_paths.append(p)
            else:
                expanded_paths.append(path)
        self.files = expanded_paths
        if not self.labels:
            self.labels = self._generate_labels()

        self.target_output_path = self.target.path.join(self.target.working_directory, 'poller.csv')
        self.target_log_path = self.target.path.join(self.target.working_directory, 'poller.log')
        marker_option = ''
        if self.align_with_ftrace:
            marker_option = '-m'
            signal.connect(self._adjust_timestamps, signal.AFTER_JOB_OUTPUT_PROCESSED)
        self.command = '{} -t {} {} -l {} {} > {} 2>{}'.format(target_poller,
                                                               self.sample_interval * 1000,
                                                               marker_option,
                                                               ','.join(self.labels),
                                                               ' '.join(self.files),
                                                               self.target_output_path,
                                                               self.target_log_path)

    def start(self, context):
        self.target.kick_off(self.command, as_root=self.as_root)

    def stop(self, context):
        self.target.killall('poller', signal='TERM', as_root=self.as_root)

    def update_output(self, context):
        host_output_file = os.path.join(context.output_directory, 'poller.csv')
        self.target.pull(self.target_output_path, host_output_file)
        context.add_artifact('poller-output', host_output_file, kind='data')

        host_log_file = os.path.join(context.output_directory, 'poller.log')
        self.target.pull(self.target_log_path, host_log_file)
        context.add_artifact('poller-log', host_log_file, kind='log')

        with open(host_log_file) as fh:
            for line in fh:
                if 'ERROR' in line:
                    raise InstrumentError(line.strip())
                if 'WARNING' in line:
                    self.logger.warning(line.strip())

    def teardown(self, context):
        self.target.remove(self.target_output_path)
        self.target.remove(self.target_log_path)

    def _generate_labels(self):
        # Split paths into their parts
        path_parts = [f.split(self.target.path.sep) for f in self.files]
        # Identify which parts differ between at least two of the paths
        differ_map = [len(set(x)) > 1 for x in zip(*path_parts)]

        # compose labels from path parts that differ
        labels = []
        for pp in path_parts:
            label_parts = [p for i, p in enumerate(pp[:-1])
                           if i >= len(differ_map) or differ_map[i]]
            label_parts.append(pp[-1])  # always use file name even if same for all
            labels.append('-'.join(label_parts))
        return labels

    def _adjust_timestamps(self, context):
        output_file = context.get_artifact_path('poller-output')
        message = 'Adjusting timestamps inside "{}" to align with ftrace'
        self.logger.debug(message.format(output_file))

        trace_txt = context.get_artifact_path('trace-cmd-txt')
        trace_parser = TraceCmdParser(filter_markers=False)
        marker_timestamp = None
        for event in trace_parser.parse(trace_txt):
            if event.name == 'print' and 'POLLER_START' in event.text:
                marker_timestamp = event.timestamp
                break

        if marker_timestamp is None:
            raise InstrumentError('Did not see poller marker in ftrace')

        df = pd.read_csv(output_file)
        df.time -= df.time[0]
        df.time += marker_timestamp
        df.to_csv(output_file, index=False)
