#    Copyright 2014-2018 ARM Limited
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

from wa import Command
from wa import discover_wa_outputs
from wa.framework.configuration.core import Status
from wa.framework.exception import CommandError
from wa.framework.output import RunOutput
from wa.framework.output_processor import ProcessorManager
from wa.utils import log


class ProcessContext(object):

    def __init__(self):
        self.run_output = None
        self.target_info = None
        self.job_output = None

    def add_augmentation(self, aug):
        pass


class ProcessCommand(Command):

    name = 'process'
    description = 'Process the output from previously run workloads.'

    def initialize(self, context):
        self.parser.add_argument('directory', metavar='DIR',
                                 help="""
                                 Specify a directory containing the data
                                 from a previous run to be processed.
                                 """)
        self.parser.add_argument('-p', '--processor', action='append',
                                 dest='additional_processors', metavar='OutputProcessor',
                                 help="""
                                 Specify an output processor to add from the
                                 command line. This can be used to run a
                                 processor that is not normally used without
                                 introducing permanent change to the config
                                 (which one might then forget to revert). This
                                 option may be specified multiple times.
                                 """)
        self.parser.add_argument('-f', '--force', action='store_true',
                                 help="""
                                 Run processors that have already been run. By
                                 default these will be skipped. Also, forces
                                 processing of in-progress runs.
                                 """)
        self.parser.add_argument('-r', '--recursive', action='store_true',
                                 help="""
                                 Walk the specified directory to process
                                 all of the previous runs contained within
                                 instead of just processing the root.
                                 """)

    def execute(self, config, args):  # pylint: disable=arguments-differ,too-many-branches,too-many-statements
        process_directory = os.path.expandvars(args.directory)
        self.logger.debug('Using process directory: {}'.format(process_directory))
        if not os.path.exists(process_directory):
            msg = 'Path `{}` does not exist, please specify a valid path.'
            raise CommandError(msg.format(process_directory))
        if not args.recursive:
            output_list = [RunOutput(process_directory)]
        else:
            output_list = list(discover_wa_outputs(process_directory))

        pc = ProcessContext()
        for run_output in output_list:
            if run_output.status < Status.OK and not args.force:
                msg = 'Skipping {} as it has not completed -- {}'
                self.logger.info(msg.format(run_output.basepath, run_output.status))
                continue

            pc.run_output = run_output
            pc.target_info = run_output.target_info

            if not args.recursive:
                self.logger.info('Installing output processors')
            else:
                self.logger.info('Install output processors for run in path `{}`'
                                 .format(run_output.basepath))

            logfile = os.path.join(run_output.basepath, 'process.log')
            i = 0
            while os.path.exists(logfile):
                i += 1
                logfile = os.path.join(run_output.basepath, 'process-{}.log'.format(i))
            log.add_file(logfile)

            pm = ProcessorManager(loader=config.plugin_cache)
            for proc in config.get_processors():
                pm.install(proc, pc)
            if args.additional_processors:
                for proc in args.additional_processors:
                    # Do not add any processors that are already present since
                    # duplicate entries do not get disabled.
                    try:
                        pm.get_output_processor(proc)
                    except ValueError:
                        pm.install(proc, pc)

            pm.validate()
            pm.initialize(pc)

            for job_output in run_output.jobs:
                if job_output.status < Status.OK or job_output.status in [Status.SKIPPED, Status.ABORTED]:
                    msg = 'Skipping job {} {} iteration {} -- {}'
                    self.logger.info(msg.format(job_output.id, job_output.label,
                                                job_output.iteration, job_output.status))
                    continue

                pc.job_output = job_output
                pm.enable_all()
                if not args.force:
                    for augmentation in job_output.spec.augmentations:
                        try:
                            pm.disable(augmentation)
                        except ValueError:
                            pass

                msg = 'Processing job {} {} iteration {}'
                self.logger.info(msg.format(job_output.id, job_output.label,
                                            job_output.iteration))
                pm.process_job_output(pc)
                pm.export_job_output(pc)

                job_output.write_result()

            pm.enable_all()
            if not args.force:
                for augmentation in run_output.augmentations:
                    try:
                        pm.disable(augmentation)
                    except ValueError:
                        pass

            self.logger.info('Processing run')
            pm.process_run_output(pc)
            pm.export_run_output(pc)
            pm.finalize(pc)

            run_output.write_info()
            run_output.write_result()
            self.logger.info('Done.')
