#    Copyright 2014-2015 ARM Limited
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
import sys
import shutil

import wlauto
from wlauto import Command, settings
from wlauto.core.agenda import Agenda
from wlauto.core.execution import Executor
from wlauto.utils.log import add_log_file
from wlauto.core.configuration import RunConfiguration
from wlauto.core import pluginloader


class RunCommand(Command):

    name = 'run'
    description = 'Execute automated workloads on a remote device and process the resulting output.'

    def initialize(self, context):
        self.parser.add_argument('agenda', metavar='AGENDA',
                                 help="""
                                 Agenda for this workload automation run. This defines which
                                 workloads will be executed, how many times, with which
                                 tunables, etc.  See example agendas in {} for an example of
                                 how this file should be structured.
                                 """.format(os.path.dirname(wlauto.__file__)))
        self.parser.add_argument('-d', '--output-directory', metavar='DIR', default=None,
                                 help="""
                                 Specify a directory where the output will be generated. If
                                 the directory already exists, the script will abort unless -f
                                 option (see below) is used, in which case the contents of the
                                 directory will be overwritten. If this option is not specified,
                                 then {} will be used instead.
                                 """.format(settings.default_output_directory))
        self.parser.add_argument('-f', '--force', action='store_true',
                                 help="""
                                 Overwrite output directory if it exists. By default, the script
                                 will abort in this situation to prevent accidental data loss.
                                 """)
        self.parser.add_argument('-i', '--id', action='append', dest='only_run_ids', metavar='ID',
                                 help="""
                                 Specify a workload spec ID from an agenda to run. If this is
                                 specified, only that particular spec will be run, and other
                                 workloads in the agenda will be ignored. This option may be
                                 used to specify multiple IDs.
                                 """)
        self.parser.add_argument('--disable', action='append', dest='instruments_to_disable',
                                 metavar='INSTRUMENT', help="""
                                 Specify an instrument to disable from the command line. This
                                 equivalent to adding "~{metavar}" to the instrumentation list in
                                 the agenda. This can be used to temporarily disable a troublesome
                                 instrument for a particular run without introducing permanent
                                 change to the config (which one might then forget to revert).
                                 This option may be specified multiple times.
                                 """)

    def execute(self, args):  # NOQA
        output_directory = self.set_up_output_directory(args)
        add_log_file(os.path.join(output_directory, "run.log"))
        config = RunConfiguration(pluginloader)

        if os.path.isfile(args.agenda):
            agenda = Agenda(args.agenda)
            settings.agenda = args.agenda
            shutil.copy(args.agenda, config.meta_directory)
        else:
            self.logger.debug('{} is not a file; assuming workload name.'.format(args.agenda))
            agenda = Agenda()
            agenda.add_workload_entry(args.agenda)

        for filepath in settings.config_paths:
            config.load_config(filepath)

        if args.instruments_to_disable:
            if 'instrumentation' not in agenda.config:
                agenda.config['instrumentation'] = []
            for itd in args.instruments_to_disable:
                self.logger.debug('Updating agenda to disable {}'.format(itd))
                agenda.config['instrumentation'].append('~{}'.format(itd))

        basename = 'config_'
        for file_number, path in enumerate(settings.config_paths, 1):
            file_ext = os.path.splitext(path)[1]
            shutil.copy(path, os.path.join(config.meta_directory,
                                           basename + str(file_number) + file_ext))

        executor = Executor(config)
        executor.execute(agenda, selectors={'ids': args.only_run_ids})

    def set_up_output_directory(self, args):
        if args.output_directory:
            output_directory = args.output_directory
        else:
            output_directory = settings.default_output_directory
        self.logger.debug('Using output directory: {}'.format(output_directory))
        if os.path.exists(output_directory):
            if args.force:
                self.logger.info('Removing existing output directory.')
                shutil.rmtree(os.path.abspath(output_directory))
            else:
                self.logger.error('Output directory {} exists.'.format(output_directory))
                self.logger.error('Please specify another location, or use -f option to overwrite.\n')
                sys.exit(1)

        self.logger.info('Creating output directory.')
        os.makedirs(output_directory)
        return output_directory
