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

from wa import Instrument, Parameter, Executable
from wa.framework.exception import ConfigError, InstrumentError
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
                  description="""A list of lables to be used in the CSV output for
                                 the corresponding files. This cannot be used if
                                 a `*` wildcard is used in a path."""),
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
        self.command = '{} -t {} -l {} {} > {} 2>{}'.format(target_poller,
                                                            self.sample_interval * 1000,
                                                            ','.join(self.labels),
                                                            ' '.join(self.files),
                                                            self.target_output_path,
                                                            self.target_log_path)

    def start(self, context):
        self.target.kick_off(self.command, as_root=self.as_root)

    def stop(self, context):
        self.target.killall('poller', signal='TERM', as_root=self.as_root)

    def update_result(self, context):
        host_output_file = os.path.join(context.output_directory, 'poller.csv')
        self.target.pull(self.target_output_path, host_output_file)
        context.add_artifact('poller_output', host_output_file, kind='data')
        host_log_file = os.path.join(context.output_directory, 'poller.log')
        self.target.pull(self.target_log_path, host_log_file)
        context.add_artifact('poller_log', host_log_file, kind='log')

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
