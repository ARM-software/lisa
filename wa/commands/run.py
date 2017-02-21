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

from wa import Command, settings
from wa.framework import log
from wa.framework.agenda import Agenda
from wa.framework.output import RunOutput


class RunCommand(Command):

    name = 'run'
    description = """
    Execute automated workloads on a remote device and process the resulting output.
    """

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
                                 """.format(settings.output_directory))
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
        try:
            executor = Executor(args.output_directory, args.force)
        except RuntimeError:
            self.logger.error('Output directory {} exists.'.format(args.output_directory))
            self.logger.error('Please specify another location, or use -f option to overwrite.\n')
            return 2
        for path in settings.get_config_paths():
            executor.load_config(path)
        executor.load_agenda(args.agenda)
        for itd in args.instruments_to_disable:
            self.logger.debug('Globally disabling instrument "{}" (from command line option)'.format(itd))
            executor.disable_instrument(itd)
        executor.initialize()
        executor.execute(selectors={'ids': args.only_run_ids})
        executor.finalize()
